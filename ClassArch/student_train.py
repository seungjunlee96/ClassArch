import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from ClassArch.dataset import ModelNet40Dataset
from ClassArch.model.PointNet import PointNet
from ClassArch.pointnet_trainer import PointNetTrainer
from ClassArch.utils import data_utils
from ClassArch.accuracy_metric import accuracy


def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Creating Directory." + dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=32
)
parser.add_argument(
    '--epoch', type=int, default=40
)
parser.add_argument(
    '--lr', type=float, default=0.003
)
parser.add_argument(
    '--dataset', type=str, default='data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)
parser.add_argument(
    '--save_model', type=str, default='./save_model/'
)

parser.add_argument(
    '--base_model', type=str, default='pointnet'
)

cfg = parser.parse_args()
print(cfg)

model_dict = {
    'pointnet': (PointNet(k=40), F.nll_loss, accuracy, torch.optim.Adam, PointNetTrainer),
}

train_transforms = transforms.Compose([
    data_utils.PointcloudToTensor()
])

test_transforms = transforms.Compose([
    data_utils.PointcloudToTensor()
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    ds_train = ModelNet40Dataset(num_points=2500, root=cfg.dataset, transforms=train_transforms)
    ds_test = ModelNet40Dataset(num_points=2500, root=cfg.dataset, transforms=test_transforms , split = 'test')
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    print("DATA LOADED")

    model, criterion, metric, optimizer, trainer = model_dict[cfg.base_model]
    optimizer = optimizer(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    print(model, criterion, optimizer)

    trainer = trainer(model, criterion, optimizer, scheduler, metric, device, None)
    fit = trainer.fit(dl_train, dl_test, num_epochs=cfg.epoch, checkpoints=cfg.save_model+model.__class__.__name__+'.pt')

    create_folder(model.__class__.__name__)
    torch.save(model.state_dict(), os.path.join(model.__class__.__name__, 'final_state_dict.pt'))
    torch.save(model, os.path.join(model.__class__.__name__, 'final.pt'))
