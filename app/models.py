import torch
import torch.nn as nn
import random
import datetime
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import pickle
import cv2
import PIL
import utils
### SUPPORTED ACTIONS
idx2action = {
    0 : '<UNK>',
    1 : 'stop',
    2 : 'backandturn',
    3 : 'forward',
    }
action2idx = {val : key for key, val in idx2action.items()}

### TRANSFORMS
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

tr_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
])

tr_depth = transforms.Compose([
        lambda x: x.astype("int32"),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        lambda x: x/5000.
        #transforms.Normalize(mean_nums, std_nums)
])

### DATALOADERS ###
class GardenData(torch.utils.data.Dataset):
    def __init__(self, dat, device="cpu"):
        self.dat = dat
        self.device = device

    def __getitem__(self, idx):
        row = self.dat.iloc[idx]

        data_dict = utils.loader(row.data_path)
        out ={}
        for key in ['acceleration']:
            out[key] = data_dict[key].astype("float")
        out['camera'] = tr_rgb(data_dict['camera'])
        out['depth_frame'] = tr_depth(data_dict['depth_frame'])
        out['action'] = action2idx.get(row.action,0)

        return out

    def __len__(self):
        return len(self.dat)

class DepthAvgNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def step(self, state):
        avg_dist = state['depth_frame'].mean()
        if avg_dist > 200:
            out = {'left' : 1, 'right' : 1, 'action' : "forward"}
        else:
            out = {'action' : "backandturn"}
        return out
class GardenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.idx2action = idx2action
        self.num_actions = len(self.idx2action)
        
    @torch.no_grad()
    def step(self, state):
        if state is None:
            return {}
        else:
            img = tr_depth(state['depth_frame']).unsqueeze(0)
            actionvec = self.forward(img).squeeze()

            actionvec[2] = actionvec[2]*1.3 # Increase stopandturn by threshold

            action = self.idx2action[actionvec.argmax().item()]
            logging.info(f"AI says: {action}")
            if action == "forward":
                out = {'left' : 1, 'right' : 1}
            else:
                out = {}
            out['action'] = action
            return out
        
class ConvNet(GardenNet):
    def __init__(self, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.channels = [1,16,32,64]
        self.layers = nn.ModuleList()
        for i in range(1,len(self.channels)):
            self.layers.append(
                nn.Conv2d(self.channels[i-1],self.channels[i], kernel_size=5, stride=2)
                )

            self.layers.append(
                nn.MaxPool2d(3, stride=2)
            )
            
        self.fc = nn.Linear(256, self.num_actions)

        #self.load_state_dict(torch.load("trained_parameters/conv_parameters.pt", map_location=torch.device('cpu')))
        logging.info("Loaded conv par")
    def forward(self, img):
        x = img
        for layer in self.layers:
            x = layer(x)
            #print(x.size())
        x = x.flatten(1)
        #print(x.size())
        x = self.fc(x)
        return x
        

class GreenNet(GardenNet):
    def __init__(self, load_parameters=False):
        super().__init__()
        self.crop_y_min = 100
        self.th=0.1
        self.fc = nn.Linear(1,len(idx2action))

        
    def forward(self, img):
        bottom_half_col = img[:,:,self.crop_y_min:,:].mean(2).mean(2)
        bottom_half_col.size()
        bottom_half_col = bottom_half_col- bottom_half_col.mean(1).unsqueeze(1)
        green = bottom_half_col[:,1]
        blue = bottom_half_col[:,2]
        
        
        out = torch.zeros((len(img), self.num_actions))
        forward = green * (green>blue)
        out[:,0] = self.th
        out[:,1] = random.random()*0.11
        out[:,2]= forward
        return out
    
class ColourNet(GardenNet):
    def __init__(self, load_parameters=False):
        super().__init__()
        self.crop_y_min = 200
        self.fc = nn.Linear(3,len(idx2action))

        
    def forward(self, img):
        bottom_half_col = img[:,:,self.crop_y_min:,:].mean(2).mean(2)
        out = self.fc(bottom_half_col)
        return out