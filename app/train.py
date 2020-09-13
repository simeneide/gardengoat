#%%
import os
tubs = os.listdir("data")

episode = max(tubs)
data_dir = f"data/{episode}"
files = os.listdir(data_dir)
files.sort()

event_files = [f for f in files if "event" in f]

#%%
import pandas as pd
import logging
import pickle
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
dat.head()
len(dat)
# extract_actions
dat['throttle'] = dat.action.map(lambda a: a['throttle'])
dat['steer'] = dat.action.map(lambda a: a['steer'])

dat = dat.dropna().reset_index(drop=True)
dat.head()
#plt.imshow(dat['image'][0])
#%%
import matplotlib.pyplot as plt
plt.imshow(loader(dat.image_path[1]))


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import model_drive
# %%
dataset = model_drive.GardenData(dat)
dl = torch.utils.data.DataLoader(dataset=dataset, batch_size = 16)
dataloaders = {'train' : dl}
dataset_sizes = {name : len(dl.dataset) for name, dl in dataloaders.items()}
batch = next(iter(dl))
#%% TRANSFORM

#%% BUILD MODEL
from torch import nn
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0

            # Here's where the training happens
            print('Iterating through data...')

            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)

                # We need to zero the gradients, don't forget it
                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model

# %%
import time
import copy
train_model(model = model, criterion = criterion, optimizer =optimizer, scheduler=scheduler, num_epochs=1)

torch.save(model, "model.pickle")
# %%
