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
                          3 : 'left',
                          4 : 'backward'}
        
        # Backandturn parameters:
        self.starttime_backandturn = False
        self.back_time = None
        self.rot_time = None
        self.start_backandturn()
    def forward(self, img):
        pass
    
    def start_backandturn(self, back_time=None, rot_time=None):
        
        self.starttime_backandturn = datetime.datetime.now()
        if back_time is None:
            self.back_time = random.randint(1,3)
        if rot_time is None:
            self.rot_time = random.randint(3,5)
        logging.info(f"Starting backandturn manouvre..: back={self.back_time}, turn={self.rot_time}")
        return self.action_backandturn()
    
    def action_backandturn(self):
        """ A special move that override other moves once started. """
        time_since_started = (datetime.datetime.now()-self.starttime_backandturn).seconds
        if time_since_started < self.back_time:
            return {'left' : -1, 'right' : -1}
        elif time_since_started < (self.back_time + self.rot_time):
            return {'left' : -1, 'right' : 1}
        else:
            logging.info("Done with backandturn move.")
            self.starttime_backandturn=False
            return {'left' : 0, 'right' : 0}
        
    @torch.no_grad()
    def step(self, state):
        if state is None:
            return {}
        elif self.starttime_backandturn:
            torch.tensor(state.get("camera")).float().mean()
            return self.action_backandturn()
        else:
            img = torch.tensor(state['camera']/255.).unsqueeze(0)
            actionvec = self.forward(img)

            action = self.idx2action[actionvec.argmax().item()]
            print(action, actionvec)
            if action == "forward":
                return {'left' : 1, 'right' : 1}
            elif action == "backandturn":
                return self.start_backandturn()
            else:
                return {}
        
        

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
        green = bottom_half_col[:,1]# - bottom_half_col[:,2]
        blue = bottom_half_col[:,2]
        
        
        out = torch.zeros((len(img), 4))
        forward = green * (green>blue)
        out[:,0] = self.th
        out[:,1]= forward
        return out
    
    
