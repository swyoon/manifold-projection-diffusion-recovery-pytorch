import os
from torch.utils.data import Dataset
from PIL import Image


class DTD(Dataset):
    """describe texture dataset
    https://www.robots.ox.ac.uk/~vgg/data/dtd/"""
    def __init__(self, root, split, transform):
        super().__init__()
        self.image_dir = os.path.join(root, 'dtd', 'dtd', 'images')
        self.label_dir = os.path.join(root, 'dtd', 'dtd', 'labels')

        # read image list
        if split == 'training':
            split = 'train'
        elif split == 'evaluation':
            split = 'test'
        else:
            raise ValueError('split must be training or evaluation')

        # find all text files starting with split in label_dir
        image_name_files = [s for s in os.listdir(self.label_dir) if s.startswith(split)]
        self.image_list = []
        for image_name_file in image_name_files:
            with open(os.path.join(self.label_dir, image_name_file), 'r') as f:
                self.image_list += [l.strip() for l in f.readlines()]

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, 0

