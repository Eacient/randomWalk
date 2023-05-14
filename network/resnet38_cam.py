import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

import network.resnet38d

class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)


    def forward(self, x):
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        cam = self.fc8(d['conv6'])
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        return None, cam
