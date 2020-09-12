import torch
import torch.nn as nn

class GreenNet(nn.Module):
    def __init__(self, load_parameters=False):
        super().__init__()
        self.crop_y_min = 100
        self.th=0.1
        self.fc = nn.Linear(1,4)
        self.idx2action = {0 : 'stop',
                          1 : 'forward',
                          2 : 'backward',
                          3 : 'left'}
        
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
    
    @torch.no_grad()
    def step(self, state):
        if state is None:
            return {}
        img = torch.tensor(state['camera']/255.).unsqueeze(0)
        actionvec = self.forward(img)
        
        action = self.idx2action[actionvec.argmax().item()]
        print(action, actionvec)
        if action == "forward":
            return {'left' : 1, 'right' : 1}
        else:
            return {}