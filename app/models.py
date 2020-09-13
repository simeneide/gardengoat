import torch
import torch.nn as nn
import random
import datetime
import logging
import time
class GardenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.idx2action = {0 : 'stop',
                          1 : 'backandturn',
                          2 : 'forward',
                          }
        self.num_actions = len(self.idx2action)
        
    @torch.no_grad()
    def step(self, state):
        if state is None:
            return {}
        else:
            img = torch.tensor(state['camera']/255.).unsqueeze(0)
            actionvec = self.forward(img)

            action = self.idx2action[actionvec.argmax().item()]
            logging.info(f"AI says: {action}")
            if action == "forward":
                out = {'left' : 1, 'right' : 1}
            else:
                out = {}
            out['action'] = action
            return out
        
        

class GreenNet(GardenNet):
    def __init__(self, load_parameters=False):
        super().__init__()
        self.crop_y_min = 100
        self.th=0.1
        self.fc = nn.Linear(1,4)

        
    def forward(self, img):
        bottom_half_col = img[:,:,self.crop_y_min:,:].mean(2).mean(2)
        bottom_half_col.size()
        bottom_half_col = bottom_half_col- bottom_half_col.mean(1)
        green = bottom_half_col[:,1]
        blue = bottom_half_col[:,2]
        
        
        out = torch.zeros((len(img), self.num_actions))
        forward = green * (green>blue)
        out[:,0] = self.th
        out[:,1] = random.random()*0.11
        out[:,2]= forward
        return out
    
    
