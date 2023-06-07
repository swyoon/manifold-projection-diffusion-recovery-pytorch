"""
    Author: Eric Y.U. Jin
    Implementation of Pytorch Datasets for
    acoustic anomaly detection.
    List of Implemented datasets
    - DCASE2020 Track 2 Dataset
"""
import os
import re
import sys
import torch
import librosa
import random
import numpy as np
import itertools
import pickle


from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import get_shuffled_idx

basic_acoustics_configs = {
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 128,
    "power": 2.0,  # magnitude
    "frames": 5,
}
valid_foldernames = [
    "fan", "pump", "slider", "valve",
    "ToyCar", "ToyConveyor",
]


def file_to_vector_array(
    wav_file: str,
    n_mels: int = 64,
    frames: int = 5,
    step: int = 1,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Description:
    - convert file_name to a vector array.
    file_name : str
        - target .wav file
    return : torch.Tensor(dtype: float)
        - shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    # > mono (bool)?
    # >> When load a multi channels file and this param True,
    # >> the returned data will be merged for mono data
    y, sr = librosa.load(wav_file, sr=None, mono=False)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = (
        20.0 / power * np.log10(
            mel_spectrogram + sys.float_info.epsilon
        )
    )

    # 04 calculate total vector size
    vector_array_size = (
        (len(log_mel_spectrogram[0, :]) - frames) // step
    ) + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    # https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
    _lms = torch.from_numpy(log_mel_spectrogram)  # share memory
    vector_array = _lms.unfold(
        dimension=1, size=vector_array_size, step=1
    ).permute(1, 0, 2)

    return vector_array.flatten(start_dim=0, end_dim=1).T


class DCASEDataset(Dataset):
    """
        DCASE Challenge Track 2 Dataset
        > One dataset is generated for one machine!
    """
    def __init__(
        self,
        # [Required]
        # 1. where to find data
        path_to_dataset: str,
        machine: str,
        # 2. train-set or validation step?
        split_type: str,
        # [Optional]
        **kwargs
    ):
        """
        Args:
            - Required:
                - path_to_dataset (str): path to DCASE2020 dataset
                    - ex) {...}/dev_data/
                - machine (str): type of machine to inspect
                    - ex) ToyCar, valve, pump
                - split_type (str): training / validation / test
            - Optional (kwargs):
                - [COMMON] reload (bool)
                    - load data from scratch and save again
                - [COMMON] frames_to_concat (int):
                    - Number of time frames to concat and use for inference.
                - [TRAIN] is_reject_ids (bool): Whether to reject list of ids
                  or use them
                - [TRAIN] designate_ids (dict): Required when training on specified ID's:
                    - Params:
                        - enable (bool): Whether to activate mode
                        - ids (list): List of id's to use
                    - Relation to "is_reject_ids":
                        - use ex) is_reject_ids = False, ["id_01", "id_05"]
                        - reject ex) is_reject_ids = True, ["id_02", "id_08"]

                - [TEST] machine_id (str): Specific ID of machine to inspect
                    - should be in format "id_0X" (X is integer)
                    - ex) "id_00"
        """
        # Load data
        # > check directory
        root_dir = os.path.join(
            path_to_dataset,
            machine,
        )
        assert os.path.isdir(root_dir), (
            f"Check designated directory: {root_dir}"
        )

        # check data type
        self.is_ood = (
            True if "ood" in split_type else False
        )

        # normalize? (default: False)
        self.enable_normalize = kwargs["normalize_dict"].get("enable", False)

        # get sampling (mel-spectrogram) params
        sfft_hop = kwargs["sfft_hop"]
        frames_to_concat = kwargs["frames_to_concat"]
        step = kwargs["step"]

        # get files from mixed ids
        if "is_reject_ids" in kwargs.keys():
            _is_reject_ids = kwargs.get("is_reject_ids", False)
        else:
            _is_reject_ids = False
        if "designate_ids" in kwargs.keys():
            _designate_id_mode = kwargs["designate_ids"]["enable"]
            if _designate_id_mode:
                assert "ids" in kwargs["designate_ids"].keys()
                _designate_ids = kwargs["designate_ids"]["ids"]
                if self.enable_normalize:
                    assert "ids_normref" in kwargs["designate_ids"].keys()
                    _designate_normrefs = kwargs["designate_ids"]["ids_normref"]
            else:
                _designate_ids = None
        else:
            _designate_id_mode = False
            _designate_ids = None
        # get files
        is_train = True if split_type == "training" else False
        files = self.get_wav_files_mixedid(
            root_dir,
            _is_reject_ids,
            _designate_ids,
            is_train,
        )
   
        # create tensor of spectograms
        # (num_files * num_spects, n_mels * n_features)
        # create temp directory if not exist
        # all default dirs are generated from the params
        default_save_dir = os.path.join(
            ".dcase_temp",
            f"sffthop-{sfft_hop}_frames-{frames_to_concat}_frameskipstep-{step}",
        )
        if not _designate_id_mode:
            # all id's are used
            dataset_save_dir = os.path.join(
                default_save_dir,
                "all_ids",
            )
        else:
            # only specified id's are used
            dataset_save_dir = os.path.join(
                default_save_dir,
                "_".join(_designate_ids),
            )
        temp_file_name = os.path.join(
            dataset_save_dir,
            f'{machine}_{split_type}.pt',
        )
        os.makedirs(
            dataset_save_dir,
            exist_ok=True
        )

        if os.path.isfile(temp_file_name) and not kwargs.get("reload", False):
            _sample_src = torch.load(temp_file_name)
            print(f'array loaded from {temp_file_name}')
        else:
            _sample_src = torch.vstack(
                [
                    file_to_vector_array(
                        wav_file=targ_file,
                        n_mels=basic_acoustics_configs["n_mels"],
                        frames=kwargs["frames_to_concat"],
                        step=kwargs["step"],
                        n_fft=basic_acoustics_configs["n_fft"],
                        hop_length=kwargs["sfft_hop"],
                        power=basic_acoustics_configs["power"],
                    ) for targ_file in tqdm(
                        files,
                        desc=f'creating <{machine}> data',
                    )
                ]
            )
            torch.save(_sample_src, temp_file_name)
            print(f'array saved to {temp_file_name}')

        # normalize data if needed (default: False)
        if self.enable_normalize:
            # Assign .dcase_temp/<machine>_scaler.pkl as default
            if not _designate_id_mode:
                scaler_save_dir = os.path.join(
                    default_save_dir,
                    "all_ids",
                )
            else:
                scaler_save_dir = os.path.join(
                    default_save_dir,
                    "_".join(_designate_normrefs)
                )
            os.makedirs(
                scaler_save_dir,
                exist_ok=True
            )
            scaler_path_default = os.path.join(
                scaler_save_dir,
                f"{machine}_scaler.pkl"
            )
            if split_type == "training":
                # save vmax, vmin of trainset
                # _scaler = MinMaxScaler(feature_range=(0., 1.), clip=True)
                _scaler = StandardScaler()
                _scaler.fit(_sample_src)
                # Assign .dcase_temp/<machine>_scaler.pkl as default
                save_path = kwargs["normalize_dict"].get(
                    "scaler_pkl", scaler_path_default
                )
                if save_path is None:
                    save_path = scaler_path_default
                print("Save path: ", save_path)
                os.makedirs(
                    os.path.split(save_path)[0], exist_ok=True
                )
                with open(save_path, 'wb') as f:
                    pickle.dump(_scaler, f)
                print("Statistics saved in >> ", save_path)
            else:
                # validation case
                # Assign .dcase_temp/<machine>_scaler.pkl as default
                load_path = kwargs["normalize_dict"].get(
                    "scaler_pkl", scaler_path_default
                )
                if load_path is None:
                    load_path = scaler_path_default
                assert os.path.exists(load_path), (
                    f"Designated {load_path} does not exist!"
                )
                print("Loading stats from >> ", load_path)
                # instantiate normalizer (default: False)
                with open(
                    load_path,
                    'rb',
                ) as f:
                    _scaler = pickle.load(f)
            # Scale sample using trained Scaler
            _sample_src = _scaler.transform(_sample_src)
            _sample_src = torch.from_numpy(_sample_src).to(dtype=torch.float)
            _sample_src.clamp_(-3., 3.)
            # sample = _sample_src
            # m = torch.stack([sample[:,:128],
            #                 sample[:,128:256],
            #                 sample[:, 256:384],
            #                 sample[:, 384:512]]).mean(dim=0)
            # m = torch.tile(m, (1,5))
            # _sample_src = (sample - m).clamp(-3., 3)

        self.sample_src = _sample_src

    def get_wav_files_mixedid(
        self, root_dir: str, is_reject: bool, designate_ids: list, is_train: bool = False,
    ):
        _full_root = os.path.join(
            root_dir,
            "train" if is_train else "test",
        )
        if designate_ids is not None:
            if not is_reject:
                _list = [
                    os.path.join(
                        _full_root,
                        file,
                    ) for file in os.listdir(_full_root) if (
                        file.endswith('.wav')
                        and (
                            re.findall('id_[0-9][0-9]', file)[0]
                            in designate_ids
                        )
                    )
                ]
            else:
                _list = [
                    os.path.join(
                        _full_root,
                        file,
                    ) for file in os.listdir(_full_root) if (
                        file.endswith('.wav')
                        and (
                            re.findall('id_[0-9][0-9]', file)[0]
                            not in designate_ids
                        )
                    )
                ]
        else:
            _list = [
                os.path.join(
                    _full_root,
                    file,
                ) for file in os.listdir(_full_root) if (
                    file.endswith('.wav')
                )
            ]

        if self.is_ood:
            _list = [item for item in _list if "anomaly" in item]
        else:
            _list = [item for item in _list if "anomaly" not in item]
        assert len(_list) > 0, (
            "No files were detected."
            + "Please check designated directory: "
            + _full_root
        )
        return sorted(_list)

    def __len__(self):
        return len(self.sample_src)

    def __getitem__(self, idx):
        if self.is_ood:
            is_anomaly = 1
        else:
            is_anomaly = 0  # all data are normal for train-set
        return self.sample_src[idx, :], is_anomaly


class DCASETestDataset(Dataset):
    """
        DCASE Challenge Track 2 Train Set
        > One dataset is generated for one id of one machine!
    """
    def __init__(
        self,
        # required
        machine_id: str,
        root_dir: str,
        frames_to_concat: int,
        step: int,
        sfft_hop: int,
        normalize_dict: dict,
        reload: bool = False,
        split: str = "evaluation",
        designate_ids: dict = None,
    ):
        """
        Args:
            - machine_id (str): ID of the machine to inspect. ex) "id_01"
            - root_dir (string): Directory with the wav files.
            - frames_to_concat (int):
                - Number of time frames to concat and use for inference.
            - normalize_dict (dict): Configs for normalization
                - enable: True->normalize/False->leave as raw
                - scaler_pkl: designate .pkl file path...
                    - if train: to save trained scaler
                    - else: to load trained scaler
            - reload (bool): whether to reset data and scaler
            - split (str): data split
            - designate_ids (dict): Configs used when training on specific id's
                - enable (bool)
                - ids_normref (list): list of id's used for normalization (
                    w.r.t. train-set)
        """
        # check directory
        assert os.path.isdir(root_dir), (
            f"Check designated directory: {root_dir}"
        )
        machine_type = os.path.split(root_dir)[-1]
        self.split = split
        files = self.get_wav_files(
            machine_id=machine_id,
            root_dir=root_dir,
        )
        # parameters required for sampling spectrograms
        self.frames_to_concat = frames_to_concat
        self.sfft_hop = sfft_hop
        self.step = step

        # all default dirs are named using the params
        default_save_dir = os.path.join(
            ".dcase_temp",
            f"sffthop-{sfft_hop}_frames-{frames_to_concat}_frameskipstep-{step}",
        )
        os.makedirs(
            default_save_dir,
            exist_ok=True
        )

        # load wav files, extract spectrograms, save into list
        saved_data_name = os.path.join(
            default_save_dir,
            f"{machine_type}_{machine_id}_{split}.pkl",
        )
        if os.path.isfile(saved_data_name) and not reload:
            # if the list exists, just load from memory
            with open(saved_data_name, 'rb') as tmp_f:
                saved_data_dict = pickle.load(tmp_f)
            self.list_data = saved_data_dict["list_data"]
            self.list_labels = saved_data_dict["list_labels"]
            self.list_fnames = saved_data_dict["list_fnames"]
            print(f'data loaded from {saved_data_name}')
        else:
            # if not generate lists
            self.list_data, self.list_labels, self.list_fnames = (
                self.load_data(files)
            )
            # save data
            data_dict_to_save = {
                "list_data": self.list_data,
                "list_labels": self.list_labels,
                "list_fnames": self.list_fnames,
            }
            with open(saved_data_name, 'wb') as tmp_f:
                pickle.dump(data_dict_to_save, tmp_f)
            print(f'data saved to {saved_data_name}')
        # normalization options
        # enable? (default >> False)
        self.enable_normalize = normalize_dict.get("enable", False)
        if self.enable_normalize:
            # Assign .dcase_temp/<machine>_scaler.pkl as default
            if designate_ids is not None:
                if designate_ids["enable"]:
                    assert "ids_normref" in designate_ids.keys()
                    scaler_dir_default = os.path.join(
                        default_save_dir,
                        "_".join(designate_ids["ids_normref"]),
                    )
                else:
                    scaler_dir_default = os.path.join(
                        default_save_dir,
                        "all_ids",
                    )
            else:
                scaler_dir_default = os.path.join(
                    default_save_dir,
                    "all_ids",
                )
            scaler_path_default = os.path.join(
                scaler_dir_default,
                f"{machine_type}_scaler.pkl"
            )
            scaler_path = normalize_dict.get("scaler_pkl", None)
            if scaler_path is None:
                scaler_path = scaler_path_default
            with open(
                scaler_path,
                'rb',
            ) as f:
                self.n_scaler = pickle.load(f)

    def get_wav_files(self, machine_id: int, root_dir: str):
        split_folder = "test" if self.split == "evaluation" else "train"
        _full_root = os.path.join(
            root_dir,
            split_folder,
        )
        _list = [
            os.path.join(
                _full_root,
                file,
            ) for file in os.listdir(_full_root) if (
                file.endswith('.wav')
                and machine_id in file
            )
        ]
        assert len(_list) > 0, (
            "No files were detected."
            + "Please check designated directory: "
            + _full_root
        )
        return sorted(_list)

    def __len__(self):
        return len(self.list_data)

    def load_data(self, files):
        # load data
        list_x = []
        list_y = []
        list_fnames = []
        for file in files:
            sample = file_to_vector_array(
                wav_file=file,
                n_mels=basic_acoustics_configs["n_mels"],
                frames=self.frames_to_concat,
                step=self.step,
                n_fft=basic_acoustics_configs["n_fft"],
                hop_length=self.sfft_hop,
                power=basic_acoustics_configs["power"],
            )
            fname = os.path.basename(file)
            # get anomaly/normal answer
            if "anomaly" in fname:
                is_anomaly = 1
            else:
                is_anomaly = 0
            # append each items to list
            list_x.append(sample)
            list_y.append(is_anomaly)
            list_fnames.append(fname)
        return list_x, list_y, list_fnames

    def __getitem__(self, idx):
        sample = self.list_data[idx]
        if self.enable_normalize:
            sample = self.n_scaler.transform(sample)
            sample = torch.from_numpy(sample).to(dtype=torch.float)
            sample.clamp_(-3., 3.)
            # m = torch.stack([sample[:,:128],
            #                 sample[:,128:256],
            #                 sample[:, 256:384],
            #                 sample[:, 384:512]]).mean(dim=0)
            # m = torch.tile(m, (1,5))
            # sample = (sample - m).clamp(-3., 3)
        is_anomaly = self.list_labels[idx]
        fname = self.list_fnames[idx]
        return sample, is_anomaly, fname

    def get_collate_fn(self):
        def fn(batch):
            batch = list(zip(*batch))
            return (
                torch.stack(batch[0]).flatten(
                    start_dim=0, end_dim=1,
                ),
                torch.tensor(batch[1], dtype=torch.long),
                batch[2],
            )
        return fn


def compute_auc_per_wav_per_id(model, dl_wav_per_id, device, max_fpr=0.1,
                               return_score=False, return_only_score=False,
                               submodule=None):
    """
    model: nn.Module
    dl_wav_per_id: DataLoader with dataset of one machine id.
                   Each batch corr wav file corresponds to single wav file.
    device: device id (ex. 'cuda:1')
    """
    l_label = []
    l_score = []
    model.eval()
    if submodule is not None:
        model = getattr(model, submodule)

    for x, _is_anomaly, fname in tqdm(dl_wav_per_id):
        with torch.no_grad():
            x = x.to(device=device)
            score = torch.mean(model.predict(x)).item()
        l_label.append(_is_anomaly.item())
        l_score.append(score)
    l_label = np.array(l_label)
    l_score = np.array(l_score)
    if return_only_score:
        return l_score
    auc = roc_auc_score(l_label, l_score)
    pauc = roc_auc_score(l_label, l_score, max_fpr=max_fpr)
    if return_score:
        return auc, pauc, l_score, l_label
    return auc, pauc



"""
    The following code is experimental.
    It may be severly impaired, thus not recommended for usage.
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%% UUT %%%%%%%%%%%%%%%%%%%%%%%%%%
def wav_to_img(
    wav_file: str,
    window_length: int = 64,
    sfft_hop: int = 512,
    n_mels: int = 128,
    n_fft: int = 1024,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Description:
    - convert file_name to a vector array.
    file_name : str
        - target .wav file
    return : torch.Tensor(dtype: float)
        - shape = (dataset_size, feature_vector_length)
    """
    # 1. generate melspectrogram using librosa
    # > mono (bool)?
    # >> When load a multi channels file and this param True,
    # >> the returned data will be merged for mono data
    y, sr = librosa.load(wav_file, sr=None, mono=False)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=sfft_hop,
        n_mels=n_mels,
        power=power,
    )

    # 2. convert melspectrogram to log mel energy
    log_mel_spectrogram = (
        20.0 / power * np.log10(
            mel_spectrogram + sys.float_info.epsilon
        )
    )

    # 3. Drop wav files that are too short
    assert log_mel_spectrogram.shape[1] > window_length, (
        "Designated window_length {window_length} may be too long!"
    )

    data = np.expand_dims(
        log_mel_spectrogram.T,
        axis=0,
    )
    return data


class DCASEImgDataset(Dataset):
    def __init__(
        self,
        # [Required]
        # 1. where to find data
        path_to_dataset: str,
        machine: str,
        # 2. train-set or validation step?
        split_type: str,
        # [Optional]
        **kwargs
    ):
        """
        Args:
            - Required:
                - path_to_dataset (str): path to DCASE2020 dataset
                    - ex) {...}/dev_data/
                - machine (str): type of machine to inspect
                    - ex) ToyCar, valve, pump
                - split_type (str): training / validation / test
            - Optional (kwargs):
                - [COMMON] reset_saved_file (bool)
                    - load data from scratch and save again
                - [COMMON] window_length (int):
                    - Number of consecutive frames to use for inference
                    - decides the hieght of each loaded image
                    - default: 64
                - [COMMON] window_overlap (int):
                    - Number of overlapping frames between consecutive windows
                    - default: 56
                - [COMMON] sfft_hop (int):
                    - default: 32
                    - SFFT hop_length
                - [TRAIN] is_reject_ids (bool): Whether to reject list of ids
                  or use them
                - [TRAIN] designate_ids (list): ID's to use or reject
                    - use ex) is_reject_ids = False, ["id_01", "id_05"]
                    - reject ex) is_reject_ids = True, ["id_02", "id_08"]

                - [TEST] machine_id (str): Specific ID of machine to inspect
                    - should be in format "id_0X" (X is integer)
                    - ex) "id_00"
        """
        # 1. check input configs
        self.split_type = split_type
        # 2. check & load data
        # check whether existing data exists
        data_load_path = '.dcase_temp/img_data/'
        os.makedirs(data_load_path, exist_ok=True)
        saved_data = os.path.join(
            data_load_path,
            f'{machine}_data.pkl'
        )
        # saved data exist, load and exploit
        if (
            os.path.exists(saved_data)
            and not kwargs.get("reset_saved_file", False)
        ):
            with open(
                saved_data,
                'rb',
            ) as tmp_f:
                data_dict = pickle.load(tmp_f)
                self.array_data = data_dict["data"]
                self.list_labels = data_dict["labels"]
                self.window_length = data_dict["window_length"]
                self.window_overlap = data_dict["window_overlap"]
                self.sfft_hop = data_dict["sfft_hop"]
        # if no data had been saved, create and save
        else:
            if "is_reject_ids" in kwargs.keys():
                _is_reject_ids = kwargs.get("is_reject_ids", False)
            else:
                _is_reject_ids = False
            if "designate_ids" in kwargs.keys():
                _designate_ids = kwargs.get("designate_ids", None)
            else:
                _designate_ids = None
            # get files
            is_train = True if split_type == "training" else False
            root_dir = os.path.join(
                path_to_dataset,
                machine,
            )
            files = self.get_wav_files(
                root_dir,
                _is_reject_ids,
                _designate_ids,
                is_train,
            )
            self.window_length = kwargs.get("window_length", 64)
            self.window_overlap = kwargs.get("window_overlap", 56)
            self.sfft_hop = kwargs.get("sftt_hop", 32)
            (
                self.array_data,
                self.list_labels,
            ) = self.load_data(
                files,
                self.window_length,
                self.sfft_hop,
            )
            assert len(self.list_labels) == self.__len__()
            data_dict = {
                "data": self.array_data,
                "labels": self.list_labels,
                "window_length": self.window_length,
                "window_overlap": self.window_overlap,
                "sfft_hop": self.sfft_hop,
            }
            with open(saved_data, 'wb') as tmp_f:
                pickle.dump(
                    data_dict, tmp_f
                )
            print("Data saved in >> ", saved_data)
        # get number of windows per wav file
        self.n_windows = self.num_windows(self.array_data)

        # normalize data if needed (default: False)
        self.enable_normalize = kwargs["normalize_dict"].get("enable", False)
        if self.enable_normalize:
            # Assign .dcase_temp/<machine>_scaler.pkl as default
            scaler_path_default = os.path.join(
                ".dcase_temp/img_data/",
                f"{machine}_scaler.pkl"
            )
            if split_type == "training":
                # save vmax, vmin of trainset
                # _scaler = MinMaxScaler(feature_range=(0., 1.), clip=True)
                _scaler = StandardScaler()
                _N, _H, _W = self.array_data.shape
                _tmp_data = self.array_data.reshape(_N * _H, _W)
                _scaler.fit(_tmp_data)
                # Assign .dcase_temp/<machine>_scaler.pkl as default
                save_path = kwargs["normalize_dict"].get(
                    "scaler_pkl", scaler_path_default
                )
                if save_path is None:
                    save_path = scaler_path_default
                print("Save path: ", save_path)
                os.makedirs(
                    os.path.split(save_path)[0], exist_ok=True
                )
                with open(save_path, 'wb') as f:
                    pickle.dump(_scaler, f)
                print("Statistics saved in >> ", save_path)
            else:
                # validation case
                # Assign .dcase_temp/<machine>_scaler.pkl as default
                load_path = kwargs["normalize_dict"].get(
                    "scaler_pkl", scaler_path_default
                )
                if load_path is None:
                    load_path = scaler_path_default
                assert os.path.exists(load_path), (
                    f"Designated {load_path} does not exist!"
                )
                print("Loading stats from >> ", load_path)
                # instantiate normalizer (default: False)
                with open(
                    load_path,
                    'rb',
                ) as f:
                    _scaler = pickle.load(f)
            # declare scaler
            self.scaler = _scaler

    def __len__(self):
        return (
            self.num_windows(self.array_data)
            * len(self.array_data)
        )

    def __getitem__(self, idx):
        wav_idx = idx // self.n_windows
        window_idx = idx - wav_idx * self.n_windows
        _data = self.array_data[
            wav_idx,
            window_idx: (window_idx + self.window_length)
        ]
        _label = self.list_labels[idx]
        if self.enable_normalize:
            _data = self.scaler.transform(_data)
        _data = torch.from_numpy(
            np.expand_dims(_data, axis=0)
        ).to(dtype=torch.float)
        return _data, _label

    def get_wav_files(
        self, root_dir: str, is_reject: bool, designate_ids: list, is_train: bool = False,
    ):
        _full_root = os.path.join(
            root_dir,
            "train" if is_train else "test",
        )
        if designate_ids is not None:
            if not is_reject:
                _list = [
                    os.path.join(
                        _full_root,
                        file,
                    ) for file in os.listdir(_full_root) if (
                        file.endswith('.wav')
                        and (
                            re.findall('id_[0-9][0-9]', file)[0]
                            in designate_ids
                        )
                    )
                ]
            else:
                _list = [
                    os.path.join(
                        _full_root,
                        file,
                    ) for file in os.listdir(_full_root) if (
                        file.endswith('.wav')
                        and (
                            re.findall('id_[0-9][0-9]', file)[0]
                            not in designate_ids
                        )
                    )
                ]
        else:
            _list = [
                os.path.join(
                    _full_root,
                    file,
                ) for file in os.listdir(_full_root) if (
                    file.endswith('.wav')
                )
            ]

        if "ood" in self.split_type:
            _list = [item for item in _list if "anomaly" in item]
        else:
            _list = [item for item in _list if "anomaly" not in item]
        assert len(_list) > 0, (
            "No files were detected."
            + "Please check designated directory: "
            + _full_root
        )
        return sorted(_list)

    def num_windows(self, array_data: np.ndarray):
        assert array_data.ndim == 3
        return (
            (
                array_data.shape[-2] - self.window_length
            ) // (
                self.window_length - self.window_overlap
            ) + 1
        )

    def load_data(
        self,
        files: list,
        window_length: int = 64,
        sfft_hop: int = 32,
    ):
        list_arrays = []
        list_labels = []
        for file in tqdm(
            files,
            desc="Reading in files and creating data..."
        ):
            _out = wav_to_img(
                wav_file=file,
                window_length=window_length,
                sfft_hop=sfft_hop,
            )
            if "anomaly" in file:
                _list_labels = [1] * self.num_windows(_out)
            else:
                _list_labels = [0] * self.num_windows(_out)
            if len(list_labels) < 0:
                list_labels = _list_labels
            else:
                list_labels.extend(_list_labels)
            list_arrays.append(_out)
        return np.vstack(list_arrays), list_labels
# %%%%%%%%%%%%%%%%%%%%%%%%%% UUT %%%%%%%%%%%%%%%%%%%%%%%%%%
