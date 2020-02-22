import os
import sys
from shutil import copyfile
import h5py
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ClassArch.dataset import ModelNet40Dataset
from ClassArch.model.PointNet import PointNet
from ClassArch.utils import data_utils


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='data/'
)



cfg = parser.parse_args()
print(cfg)

test_transforms = transforms.Compose([
    data_utils.PointcloudToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_test = ModelNet40Dataset(num_points=2500, root=cfg.dataset, split='test', transforms=test_transforms)

    model_loc = './PointNet/PointNet_best'
    # model_loc = './epoch_save_model/PointNet_ckpt_'
    model = torch.load(model_loc + '.pt')
    model = model.cuda()
    model.eval()

    label = []
    correct = 0
    not_correct = 0

    for i in range(840):
        X, y = ds_test.__getitem__(i)
        X = X.view(1, 3, 2048).cuda()
        pred, _, _ = model(X)
        pred = (pred == pred.max()).nonzero().flatten()[-1]

        
        # print(pred)
        # print(y)
        
        if pred == y.cuda():
            correct += 1
        else:
            not_correct += 1

        print(f'Accuracy: {correct / (not_correct+correct)}')

        
