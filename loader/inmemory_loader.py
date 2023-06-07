import numpy as np
import torch
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils import data
import collections
import pickle


class InMemoryDataset(data.Dataset):
    def __init__(
        self,
        root,
        n_classes,
        split="train",
        split_type="training",
        augmentations=None,
        is_transform=False,
        label="pre-encoded",
        mode="Segmentation",
        resize_factor=1,
        gray=False,
        flatten=False,
    ):
        """
        read pre-processed data.
        assume that the data is already processed with ToTensor()
        """
        self.root = root
        self.n_classes = n_classes
        self.split = split
        self.split_type = split_type
        self.augmentations = augmentations
        self.is_transform = is_transform
        self.label = label
        self.mode = mode
        self.flatten = flatten

        # unused
        self.resize_factor = resize_factor
        self.gray = gray

        path = pjoin(self.root, 'InMemory', self.mode, f'{split}_{label}.pt')
        self.data, self.label = pickle.load(open(path, 'rb')) 

        self.files = collections.defaultdict(list)
        path = pjoin(self.root, "ImageSets", self.mode, split + ".txt")
        file_list = tuple(open(path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list

        self.img_size = (int(self.data[0].shape[2] / self.resize_factor), int(self.data[0].shape[2] / self.resize_factor))
        self.resize = torch.nn.AdaptiveAvgPool2d(output_size=(self.img_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.resize(self.data[index])
        if self.flatten:
            img = img.view((-1,))
        lbl = self.label[index]
        return img, lbl


class InMemoryLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        idx = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(idx)
        data = zip(*[dataset[i] for i in tqdm(idx)])
        self.data = [torch.stack(d) for d in data]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == len(self):
            raise StopIteration

        batch_start = self.i * self.batch_size
        batch_end = min((self.i + 1) * self.batch_size, len(self.dataset))

        self.i += 1
        return [d[batch_start:batch_end] for d in self.data]

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
