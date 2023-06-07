import numpy as np
from loader import get_dataset, get_dataloader
from torch.utils import data
import yaml
import pytest
from skimage import io
import pickle
import torch
import os


def test_get_dataloader():
    cfg = {'dataset': 'FashionMNISTpad_OOD',
           'path': 'datasets',
           'shuffle': True,
           'n_workers': 0,
           'batch_size': 1,
           'split': 'training'}
    dl = get_dataloader(cfg)



# skip if data doesn't exist
@pytest.mark.skipif(not os.path.exists('datasets/dtd'), reason="DTD dataset not found")
def test_dtd():
    data_cfg = {'dataset': 'dtd',
                'path': 'datasets',
                'shuffle': True,
                'n_workers': 0,
                'batch_size': 1,
                'split': 'training',
                'size': 32}
    ds = get_dataset(data_cfg)
    ds[0]
    len(ds)
