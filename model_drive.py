import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import pickle
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

tr = transforms.Compose([
        #transforms.RandomResizedCrop(size=256),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
])

class GardenData(torch.utils.data.Dataset):
    def __init__(self, dat):
        self.dat = dat

    def loader(self,path):
        with open(path, "rb") as handle:
            out = pickle.load(handle)
        return out

    def __getitem__(self, idx):
        row = self.dat.iloc[idx]

        out = {'throttle' : row.throttle,
        'steer' : row.steer,
        'image' : tr(self.loader(row.image_path)),
        'label' : to_cat(row.throttle, row.steer)}
        return out
    def __len__(self):
        return len(self.dat)

def to_cat(throttle, steer):
    if throttle==1:
        return 0
    if throttle==-1:
        return 1
    elif steer == -1:
        return 2
    elif steer == 1:
        return 3