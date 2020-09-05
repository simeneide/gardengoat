import torch
import torch.nn as nn

class GreenNet(nn.Module):
    def __init__(self, load_parameters=False):
        super().__init__()
        self.crop_y_min = 100
        self.th=0.1
        self.fc = nn.Linear(1,4)
        
    def forward(self, img):
        bottom_half_col = img[:,:,self.crop_y_min:,:].mean(2).mean(2)
        bottom_half_col.size()
        bottom_half_col = bottom_half_col- bottom_half_col.mean(1) #light = bottom_half_col.mean(1)
        green = bottom_half_col[:,1]# - bottom_half_col[:,2]
        blue = bottom_half_col[:,2]
        print(green,blue)
        forward = ((green>self.th) & (green>blue))
        out = torch.zeros((len(img), 4))
        out[forward,0] = 1.0
        out[forward==False,2] = 1.0
        #out = self.fc(bottom_half_green)
        return out