import numpy as np
import copy
import json
import torch
from torch.utils import data
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop,\
                                   Pad, Normalize

from loader.inmemory_loader import InMemoryLoader, InMemoryDataset
from loader.basic_dataset import basic_dataset
from loader.leaveout_dataset import MNISTLeaveOut, CIFAR10LeaveOut
from loader.modified_dataset import Gray2RGB, MNIST_OOD, FashionMNIST_OOD, \
                                    CIFAR10_OOD, SVHN_OOD, Constant_OOD, \
                                    Noise_OOD, CIFAR100_OOD, CelebA_OOD, \
                                    NotMNIST, ConstantGray_OOD, ImageNet32
from loader.dtd import DTD
from loader.chimera_dataset import Chimera
import torchvision
from torchvision.datasets import FashionMNIST, Omniglot 
from augmentations import get_composed_augmentations
from augmentations.augmentations import ToGray, Invert, Fragment
# from loader.acoustic_dataset import (
#     DCASEDataset, DCASETestDataset
# )


OOD_SIZE = 32  # common image size for OOD detection experiments


def get_dataloader(data_dict, mode=None, mode_dict=None, data_aug=None, subset=None):
    """constructs DataLoader
    data_dict: data part of cfg

    mode: deprecated argument
    mode_dict: deprecated argument
    data_aug: deprecated argument
    subset: the list of indices in dataset

    Example data_dict
        dataset: FashionMNISTpad_OOD
        path: datasets
        shuffle: True
        batch_size: 128
        n_workers: 8
        split: training
        dequant:
          UniformDequantize: {}
    """

    # dataset loading
    aug = get_composed_augmentations(data_dict.get('augmentations', None))
    dequant = get_composed_augmentations(data_dict.get('dequant', None))
    dataset = get_dataset(data_dict, split_type=None, data_aug=aug, dequant=dequant)

    if subset is not None:
        dataset = data.Subset(dataset, subset)

    # dataloader loading
    if callable(getattr(dataset, 'get_collate_fn', None)):
        _collate_fn = dataset.get_collate_fn()
    else:
        _collate_fn = None
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        num_workers=data_dict["n_workers"],
        shuffle=data_dict.get('shuffle', False),
        pin_memory=False,
        collate_fn=(
            _collate_fn if _collate_fn is not None else None
        ),
    )

    return loader


def get_dataset(data_dict, split_type=None, data_aug=None, dequant=None):
    """
    split_type: deprecated argument
    """
    do_concat = any([k.startswith('concat') for k in data_dict.keys()])
    if do_concat:
        if data_aug is not None:
            return data.ConcatDataset([get_dataset(d, data_aug=data_aug) for k, d in data_dict.items() if k.startswith('concat')])
        elif dequant is not None:
            return data.ConcatDataset([get_dataset(d, dequant=dequant) for k, d in data_dict.items() if k.startswith('concat')])
        else: return data.ConcatDataset([get_dataset(d) for k, d in data_dict.items() if k.startswith('concat')])
    name = data_dict["dataset"]
    split_type = data_dict['split']
    data_path = data_dict["path"][split_type] if split_type in data_dict["path"] else data_dict["path"]

    # default tranform behavior. 
    original_data_aug = data_aug
    if data_aug == 'None':  # really do nothing
        data_aug = None
    elif data_aug is not None:
        #data_aug = Compose([data_aug, ToTensor()])
        data_aug = Compose([ToTensor(), data_aug])
    else:
        data_aug = ToTensor()

    if dequant is not None:  # dequantization should be applied last
        data_aug = Compose([data_aug, dequant])


    # datasets
    if name == 'MNISTLeaveOut':
        l_out_class = data_dict['out_class']
        dataset = MNISTLeaveOut(data_path, l_out_class=l_out_class, split=split_type, download=False,
                                transform=data_aug, holdout=data_dict.get('holdout', False))
    elif name == 'MNISTLeaveOut_pad':
        l_out_class = data_dict['out_class']
        l_aug = [
               Pad(2),
               data_aug]
        if data_dict.get('rgb', False):
            l_aug = [Gray2RGB()] + l_aug

        transform = Compose(l_aug)
        dataset = MNISTLeaveOut(data_path, l_out_class=l_out_class, split=split_type, download=False,
                                transform=transform)

    elif name == 'MNISTLeaveOutFragment':
        l_out_class = data_dict['out_class']
        fragment = data_dict['fragment']
        dataset = MNISTLeaveOut(data_path, l_out_class=l_out_class, split=split_type, download=False,
                                transform=Compose([ToTensor(),
                                                   Fragment(fragment)]))
    elif name == 'MNIST_OOD':
        size = data_dict.get('size', 28)
        if size == 28:
            l_transform = [ToTensor()]
        else:
            l_transform = [Gray2RGB(), Resize(OOD_SIZE), ToTensor()]
        dataset = MNIST_OOD(data_path, split=split_type, download=False,
                            transform=Compose(l_transform))
        dataset.img_size = (size, size)

    elif name == 'MNISTpad_OOD':
        dataset = MNIST_OOD(data_path, split=split_type, download=False,
                            transform=Compose([Gray2RGB(),
                                               Pad(2),
                                               ToTensor()]))
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == 'FashionMNIST_OOD':
        size = data_dict.get('size', 28)
        if size == 28:
            l_transform = [ToTensor()]
        else:
            l_transform = [Gray2RGB(), Resize(OOD_SIZE), ToTensor()]

        dataset = FashionMNIST_OOD(data_path, split=split_type, download=False,
                            transform=Compose(l_transform))
        dataset.img_size = (size, size)

    elif name == 'FashionMNISTpad_OOD':
        dataset = FashionMNIST_OOD(data_path, split=split_type, download=False,
                            transform=Compose([Gray2RGB(),
                                               Pad(2),
                                               ToTensor()]))
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == 'HalfMNIST':
        mnist = MNIST_OOD(data_path, split=split_type, download=False,
                            transform=ToTensor())
        dataset = Chimera(mnist, mode='horizontal_blank')
    elif name == 'ChimeraMNIST':
        mnist = MNIST_OOD(data_path, split=split_type, download=False,
                            transform=ToTensor())
        dataset = Chimera(mnist, mode='horizontal')
    elif name == 'CIFAR10_OOD':
        dataset = CIFAR10_OOD(data_path, split=split_type, download=False,
                              transform=data_aug)
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == 'CIFAR10LeaveOut':
        l_out_class = data_dict['out_class']
        seed = data_dict.get('seed', 1)
        dataset = CIFAR10LeaveOut(data_path, l_out_class=l_out_class, split=split_type, download=False,
                              transform=data_aug, seed=seed)

    elif name == 'CIFAR10_GRAY':
        dataset = CIFAR10_OOD(data_path, split=split_type, download=False,
                              transform=Compose([ToTensor(),
                                                 ToGray()]))
        dataset.img_size = (OOD_SIZE, OOD_SIZE)


    elif name == 'CIFAR100_OOD':
        dataset = CIFAR100_OOD(data_path, split=split_type, download=False,
                               transform=data_aug)
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == 'SVHN_OOD':
        dataset = SVHN_OOD(data_path, split=split_type, download=False,
                           transform=data_aug)
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == 'Constant_OOD':
        size = data_dict.get('size', OOD_SIZE)
        channel = data_dict.get('channel', 3)
        dataset = Constant_OOD(data_path, split=split_type, size=(size, size),
                               channel=channel,
                               transform=ToTensor())

    elif name == 'ConstantGray_OOD':
        size = data_dict.get('size', OOD_SIZE)
        channel = data_dict.get('channel', 3)
        dataset = ConstantGray_OOD(data_path, split=split_type, size=(size, size),
                               channel=channel,
                               transform=ToTensor())

    elif name == 'Noise_OOD':
        channel = data_dict.get('channel', 3)
        size = data_dict.get('size', OOD_SIZE)
        dataset = Noise_OOD(data_path, split=split_type,
                            transform=ToTensor(), channel=channel, size=(size, size))

    elif name == 'CelebA_OOD':
        size = data_dict.get('size', OOD_SIZE)
        l_aug = []
        l_aug.append(CenterCrop(140))
        l_aug.append(Resize(size))
        l_aug.append(ToTensor())
        if original_data_aug is not None:
            l_aug.append(original_data_aug)
        if dequant is not None:
            l_aug.append(dequant)
        data_aug = Compose(l_aug)
        dataset = CelebA_OOD(data_path, split=split_type,
                             transform=data_aug)
        dataset.img_size = (OOD_SIZE, OOD_SIZE)

    elif name == 'FashionMNIST':   # normal FashionMNIS
        dataset = FashionMNIST_OOD(data_path, split=split_type, download=False,
                                   transform=ToTensor())
        dataset.img_size = (28, 28)
    elif name == 'MNIST':   # normal  MNIST
        dataset = MNIST_OOD(data_path, split=split_type, download=False,
                            transform=ToTensor())
        dataset.img_size = (28, 28)
    elif name == 'NotMNIST':
        dataset = NotMNIST(data_path, split=split_type, transform=ToTensor())
        dataset.img_size = (28, 28)
    elif name == 'Omniglot':
        size = data_dict.get('size', OOD_SIZE)
        invert = data_dict.get('invert', True)  # invert pixel intensity: x -> 1 - x
        if split_type == 'training':
            background = True
        else:
            background = False

        if invert:
            tr = Compose([Resize(size), ToTensor(), Invert()])
        else:
            tr = Compose([Resize(size), ToTensor()])

        dataset = Omniglot(data_path, background=background, download=False,
                           transform=tr)
    elif name == 'KMNIST':
        from torchvision.datasets import KMNIST
        dataset = KMNIST(root=data_path, train=split_type == 'training', download=False, transform=ToTensor())
    elif name == 'EMNIST':
        from torchvision.datasets import EMNIST
        tr = Compose([
            lambda img: torchvision.transforms.functional.rotate(img, -90),
            lambda img: torchvision.transforms.functional.hflip(img),
            ToTensor()])
        dataset = EMNIST(root=data_path, split='letters', train=split_type == 'training', download=False,
                transform=tr)
    elif name == 'ImageNet32':
        train_split_ratio = data_dict.get('train_split_ratio', 0.8)
        seed = data_dict.get('seed', 1)
        dataset = ImageNet32(data_path, split=split_type, transform=ToTensor(), seed=seed,
                             train_split_ratio=train_split_ratio)
    elif name == "DCASE":
        seed = data_dict.get('seed', 1)
        dataset = DCASEDataset(
            # required
            path_to_dataset=data_path,
            machine=data_dict.get('machine_type', 'fan'),
            split_type=split_type,
            # optional
            frames_to_concat=data_dict.get('frames_to_concat', 5),
            step=data_dict.get('step', 1),
            sfft_hop=data_dict.get("sfft_hop", 32),
            is_reject_ids=data_dict.get('is_reject_ids', False),
            normalize_dict=data_dict['normalize_dict'],
            reload=data_dict.get('reload', False),
            # required for per-id training
            designate_ids=data_dict.get('designate_ids', None),
        )
    elif name == "DCASE_test":
        dataset = DCASETestDataset(
            machine_id=data_dict['machine_id'],
            root_dir=data_path,
            frames_to_concat=data_dict['frames_to_concat'],
            step=data_dict.get('step', 1),
            sfft_hop=data_dict.get("sfft_hop", 32),
            normalize_dict=data_dict['normalize_dict'],
            reload=data_dict.get('reload', False),
            split=data_dict.get('split', 'evaluation'),
            # required for per-id training
            designate_ids=data_dict.get('designate_ids', None),
        )
    elif name == "TensorDataset":
        # assumes that the data is stored in a pickle file with a dictionary
        # the key of the dictionary is the id of a dataset
        data_dict_ = data_dict.copy()
        # data_dict_.pop('name')
        d_data = torch.load(data_dict_['path'])
        tensor = d_data[data_dict_['key']]
        if data_dict_['key'] + '_targets' in d_data:
            print('loading targets')
            target = torch.tensor(d_data[data_dict_['key'] + '_targets'])# .unsqueeze(1)
        else:
            target = torch.zeros(tensor.shape[0])
        dataset = data.TensorDataset(tensor, target)
    elif name == "ExtractedFeature":
        data_dict_ = data_dict.copy()
        data_dict_.pop('dataset')
        dataset = ExtractedFeatureDataset(
                **data_dict_
                )
    elif name == 'dtd':
        data_dict_ = data_dict.copy()
        split = data_dict_.pop('split')
        size = data_dict.get("size", 224)
        transform = Compose([Resize(size), CenterCrop(size), ToTensor()])
        dataset = DTD(data_path, split=split, transform=transform)
    # elif name == "DCASE2D":
    #     seed = data_dict.get('seed', 1)
    #     dataset = DCASEImgDataset(
    #         # required
    #         path_to_dataset=data_path,
    #         machine=data_dict.get('machine_type', 'fan'),
    #         split_type=split_type,
    #         # optional
    #         reset_saved_file=data_dict.get("reset_saved_file", False),
    #         window_length=data_dict.get("window_length", 64),
    #         window_overlap=data_dict.get("window_overlap", 56),
    #         sfft_hop=data_dict.get("sfft_hop", 32),
    #         is_reject_ids=data_dict.get('is_reject_ids', False),
    #         designate_ids=data_dict.get('designate_ids', None),
    #         normalize_dict=data_dict['normalize_dict'],
    #     )
    # elif name == "DCASE2D_test":
    #     dataset = DCASETestImgDataset(
    #         machine_id=data_dict['machine_id'],
    #         root_dir=data_path,
    #         reset_saved_file=data_dict.get("reset_saved_file", False),
    #         window_length=data_dict.get("window_length", 64),
    #         window_overlap=data_dict.get("window_overlap", 56),
    #         sfft_hop=data_dict.get("sfft_hop", 32),
    #         normalize_dict=data_dict['normalize_dict'],
    #     )
    else:
        n_classes = data_dict["n_classes"]
        split = data_dict['split'][split_type]

        param_dict = copy.deepcopy(data_dict)
        param_dict.pop("dataset")
        param_dict.pop("path")
        param_dict.pop("n_classes")
        param_dict.pop("split")
        param_dict.update({"split_type": split_type})


        dataset_instance = _get_dataset_instance(name)
        dataset = dataset_instance(
            data_path,
            n_classes,
            split=split,
            augmentations=data_aug,
            is_transform=True,
            **param_dict,
        )

    return dataset


def _get_dataset_instance(name):
    """get_loader

    :param name:
    """
    return {
        "basic": basic_dataset,
        "inmemory": InMemoryDataset,
    }[name]



def np_to_loader(l_tensors, batch_size, num_workers, load_all=False, shuffle=False):
    '''Convert a list of numpy arrays to a torch.DataLoader'''
    if load_all:
        dataset = data.TensorDataset(*[torch.Tensor(X).cuda() for X in l_tensors])
        num_workers = 0
        return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    else:
        dataset = data.TensorDataset(*[torch.Tensor(X) for X in l_tensors])
        return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=False)


import os
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torchvision.transforms import Resize, Normalize, Compose, InterpolationMode
from torchvision.transforms.functional import hflip
from tqdm import tqdm



class ExtractedFeatureDataset:
    def __init__(self,
            model_name,
            split,
            path,
            filename,
            dataset_dict=None,
            pretrained=True,
            override=False,
            extract_hflip=False,
            extract_device='cuda:0',
            extract_center_crop=False,
            unittest=False,
            bound=None,
            **kwargs):
        """
        Dataset for feature extracted from a pretrained network
        dataset: torchvision dataset class
        model_name: name of the pretrained model
        split: split of the dataset
        unittest: a flag for unittest
        """
        self.model_name = model_name
        self.split = split
        self.extract_hflip = extract_hflip
        self.extract_device = extract_device
        self.extract_center_crop = extract_center_crop
        self.unittest = unittest
        self.bound = bound
        dataset_name = dataset_dict['dataset']
        if filename is not None:
            self.file_path = os.path.join(path,
                    'extracted_feature',
                    f'{dataset_name}_{model_name}_{filename}.pkl')
        else:
            if extract_hflip:
                self.file_path = os.path.join(path,
                        'extracted_feature',
                        f'{dataset_name}_{model_name}_hflip.pkl')
            else:
                self.file_path = os.path.join(path,
                        'extracted_feature',
                        f'{dataset_name}_{model_name}.pkl')

        # read saved file if exists
        if not os.path.exists(self.file_path) or override:
            # create directory if not exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            self._build(dataset_dict)
        data = torch.load(self.file_path)
        self.data = data[self.split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.bound is not None:
            x = (x - self.bound[0]) / (self.bound[1] - self.bound[0])
            x = 2 * x - 1
            x = torch.clamp(x, -1, 1)
        return x, 0

    def _build(self, dataset_dict):
        print(f'{self.file_path} does not exist, extracting features...')

        # prepare pre-processor
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        model.to(self.extract_device)
        model.eval()

        if self.extract_center_crop:
            # let's use what is provided by timm
            # this first resizes an image to 256 then center-crops to 224.
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
        else:
            # this uses the whole image without crop
            transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC),
                                 ToTensor(),
                            Normalize([0.4850, 0.4560, 0.4060],
                                [0.2290, 0.2240, 0.2250])])

        dataset_dict_ = dataset_dict.copy()
        batch_size = dataset_dict_.pop('batch_size')
        n_workers = dataset_dict_.pop('n_workers')
        d_data = {}
        for split in ['training', 'validation', 'evaluation']:
            dataset_dict_['split'] = split
            ds = get_dataset(dataset_dict_)
            ds.transform = transform
            dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=n_workers,)

            l_emb = []
            for xx, _ in tqdm(dl):
                with torch.no_grad():
                    o = model(xx.to(self.extract_device)).detach().cpu()

                l_emb.append(o)

                if self.extract_hflip:
                    xx = hflip(xx)
                    with torch.no_grad():
                        o = model(xx.to(self.extract_device)).detach().cpu()
                    l_emb.append(o)

                if self.unittest:
                   break 

            d_data[split] = torch.cat(l_emb)
        torch.save(d_data, self.file_path)
        print(f'Features are saved to {self.file_path}')


