import torch
import torch.nn as nn

class GreenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_y_min = 150
        self.th=0.2
        self.fc = nn.Linear(1,4)
    def forward(self, img):
        bottom_half_col = img[:,:,self.crop_y_min:,:].mean(2).mean(2)
        bottom_half_col.size()
        green = bottom_half_col[:,1] - bottom_half_col[:,2]

        out = torch.zeros((len(img), 4))
        out[(green>self.th),0] = 1.0
        out[(green<=self.th),1] = 1.0
        #out = self.fc(bottom_half_green)
        return out