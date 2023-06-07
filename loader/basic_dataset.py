import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms


class basic_dataset(data.Dataset):

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
    ):
        """
        gray: If True, read images in gray scale (1-channel). If False, gray scale images are converted to a 3 channel
              image with each channel has identical content.
        """
        self.root = root
        self.n_classes = n_classes
        self.split = split
        self.split_type = split_type
        self.augmentations = augmentations
        self.is_transform = is_transform
        self.label = label
        self.mode = mode

        self.resize_factor = resize_factor
        self.gray = gray

        self.files = collections.defaultdict(list)
        path = pjoin(self.root, "ImageSets", self.mode, split + ".txt")
        file_list = tuple(open(path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list

        self.tf = transforms.ToTensor()

        # average sample images to derive img_size
        img_width = []
        img_height = []
        for i in range(len(self.files[self.split])):
            img_path = self.get_file_path(pjoin(self.root, "JPEGImages"), self.files[self.split][i])
            img = Image.open(img_path)
            img_s = img.size
            img_width.append(img_s[0])
            img_height.append(img_s[1])
            if i > 20:
                break
        self.img_size = (int(np.mean(img_width) / self.resize_factor), int(np.mean(img_height) / self.resize_factor))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # find file name
        img_name = self.files[self.split][index]
        img_path = self.get_file_path(pjoin(self.root, "JPEGImages"), img_name)
        lbl_path = self.get_file_path(pjoin(self.root, self.mode + "Class", self.label), img_name)

        # open image
        img = Image.open(img_path)
        if self.resize_factor > 1:
            img = img.resize(self.img_size)
        if (len(img.getbands()) < 3) and not self.gray:  # img is a grey-scale
            img = img.convert('RGB')
        if (len(img.getbands()) == 3) and self.gray:
            img = img.getchannel("R")  # select the first channel

        if self.mode == "Classification":
            f_lbl = open(lbl_path, 'r')
            lbl = int(f_lbl.read())

            if self.n_classes == 2:
                lbl = 1 if (lbl > 1) else lbl

            if self.augmentations is not None:
                img = self.augmentations(img)

        elif self.mode == "Segmentation":
            lbl = Image.open(lbl_path).convert('L')
            if self.n_classes == 2:
                lbl_array = np.array(lbl)
                lbl_array[lbl_array > 1] = 1
                lbl = Image.fromarray(lbl_array)

            if self.resize_factor > 1:
                lbl = lbl.resize(self.img_size)

            if self.augmentations is not None:
                img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def get_file_path(self, file_dir, file_name):
        dic_file = {fname.split('.')[0]: fname for fname in os.listdir(file_dir)}
        file_basename = dic_file[file_name]
        file_path = pjoin(file_dir, file_basename)

        return file_path

    def transform(self, img, lbl):
        img = img.resize(self.img_size)
        img = self.tf(img)

        if self.mode == "Classification":
            lbl = torch.tensor(lbl).long()

        elif self.mode == "Segmentation":
            lbl = lbl.resize(self.img_size)
            lbl = torch.from_numpy(np.array(lbl)).long()
            lbl[lbl == 255] = 0

        return img, lbl
