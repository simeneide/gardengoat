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
#%%
tubs = os.listdir("data")

episode = 'tub_2020-09-14.10:47:48' #max(tubs)
data_dir = f"data/{episode}"
files = os.listdir(data_dir)
files.sort()

event_files = [f for f in files if "event" in f]

#%%

def loader(path):
    with open(path, "rb") as handle:
        dat = pickle.load(handle)
    return dat

L = []
for filename in event_files:
    try:
        path = f"{data_dir}/{filename}"
        l = loader(path)
        L.append(l)
    except:
        logging.warning(f"Failed to load {path}")
#%%
dat = pd.DataFrame(L)
logging.info(f"Loaded {len(dat)} events into dataframe.")
dat.head()
#dat['image_path'] = dat.image_path.map(lambda s: f"/root/{s}")
#%%

# extract_actions
dat['action_dict'] = dat['action']
dat['action'] = dat.action.map(lambda a: a.get('action'))
dat = dat.dropna().sort_values("step").reset_index(drop=True)
dat.head()
#%% LOAD IMAGE TO TEST
#plt.imshow(cv2.imread(dat.image_path[1]))

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import models
# %%
dataset = models.GardenData(dat, device="cuda")
dl = torch.utils.data.DataLoader(dataset=dataset, batch_size = 32, num_workers=0)
dataloaders = {'train' : dl}
dataset_sizes = {name : len(dl.dataset) for name, dl in dataloaders.items()}
#%%
from torch import nn
model = models.ConvNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dataloaders['train'].dataset.dat['action'].hist()
#%%
num_epochs=100
for ep in range(num_epochs):
    phase = "train"
    stats = {'num_obs' : 0, 'loss' : 0, 'corrects' : 0}
    for batch in dataloaders[phase]:
        batch = {key : val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()
        yhat = model(batch['image'])
        loss = criterion(yhat, batch['action'])
        if phase == 'train':
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            
            pred_class = torch.argmax(yhat,1)
            stats['loss'] += loss
            stats['corrects'] += (pred_class == batch['action']).sum().float()
            stats['num_obs'] += len(batch['action'])
    logging.info(f"Episode {ep}: loss={stats['loss']/stats['num_obs']:.3f}, accuracy={stats['corrects']/stats['num_obs']:.2f}")
# %%
import time
torch.save(model, "model.pickle")
# %%
