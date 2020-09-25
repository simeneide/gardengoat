#%%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import pickle
import matplotlib.pyplot as plt
import cv2
import utils
#%%
data_dir = "../data/data/"
tubs = os.listdir(data_dir)

episode = max(tubs)
data_dir = f"{data_dir}/{episode}"
print(episode)
#%%
files = os.listdir(data_dir)
files.sort()

event_files = [f for f in files if "event" in f]

#%%
L = []
for filename in event_files:
    try:
        path = f"{data_dir}/{filename}"
        l = utils.loader(path)
        L.append(l)
    except:
        logging.warning(f"Failed to load {path}")
#%%
dat = pd.DataFrame(L)
logging.info(f"Loaded {len(dat)} events into dataframe.")

# extract_actions
dat['action_dict'] = dat['action']
dat['active_option'] = dat.action_dict.map(lambda d: d.get("active_option", False)=="BackAndTurn")
dat['action'] = dat.action_dict.map(lambda a: a.get('action'))
dat = dat[(dat.action!="favicon.ico")]
dat = dat[(dat.action!="stop")]

# Get full path to data (No need to just take basename here in next iteration)
dat['data_path'] = dat['data_path'].map(lambda s: f"{data_dir}/{os.path.basename(s)}")

# remove all images that are well within the active option. 
# we dont want to learn here...
keep = dat['active_option'].rolling(5, center=True).mean() <1.0
dat = dat[keep]
dat = dat.dropna().sort_values("step").reset_index(drop=True)
logging.info(f"After filters we have {len(dat)} events in dataframe.")

dat['action'].hist()
#dat.head()
#%% LOAD IMAGE TO TEST
#plt.imshow(cv2.imread(dat.image_path[1]))

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import models
# %%
dataset = models.GardenData(dat, device="cuda")
train_len = int(len(dataset)*0.9)
d = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len])
datasets = {'train' : d[0], 'valid' : d[1]}
dataloaders = {phase : torch.utils.data.DataLoader(dataset=ds, batch_size = 32, num_workers=0, shuffle=True)
        for phase, ds in datasets.items()}
dataset_sizes = {name : len(dl.dataset) for name, dl in dataloaders.items()}

logging.info("Dataset sizes:")
logging.info(dataset_sizes)

#depth = dataset.__getitem__(2)['depth_frame']


#%%
from torch import nn
model = models.ConvNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#%%
num_epochs=100
for ep in range(num_epochs):
    logging.info(" ")
    logging.info("-"*30)
    logging.info(f"EPISODE {ep}")
    for phase, dl in dataloaders.items():
        stats = {'num_obs' : 0, 'loss' : 0, 'corrects' : 0, 
        'pred_class' : torch.zeros((len(model.idx2action)), device=device)}
        for batch in dl:
            batch = {key : val.to(device) for key, val in batch.items()}
            
            yhat = model(batch['depth_frame'])
            loss = criterion(yhat, batch['action'])
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_class = torch.argmax(yhat,1)
                id, cnts = pred_class.unique(return_counts=True)
                stats['pred_class'][id] += cnts
                stats['loss'] += loss
                stats['corrects'] += (pred_class == batch['action']).sum().float()
                stats['num_obs'] += len(batch['action'])
        logging.info(f"{phase}: loss={stats['loss']/stats['num_obs']:.3f}, accuracy={stats['corrects']/stats['num_obs']:.2f}   cl={(stats['pred_class']/stats['num_obs']).cpu().numpy().round(3)}")
# %%
with torch.no_grad():
    scores = model(batch['depth_frame']).cpu()
    scores[:,2] = scores[:,2]*1.3
# %%
plt.hist(scores.detach().cpu().t())
#%%
plt.hist(scores.argmax(1).detach().cpu())
#%%
plt.plot(scores)
#%%
plt.plot(batch['action'].detach().cpu())
plt.plot(scores[:,3]/scores[:,2])
#%% THRESHOLD
import numpy as np
np.quantile(scores[:,3]/scores[:,2], 0.2)
# %%
torch.save(model.state_dict(),"trained_parameters/conv_parameters.pt")
# %%
