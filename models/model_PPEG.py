from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        # reshape to 2D
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x_pos = self.proj(cnn_feat) + cnn_feat + \
                self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x_pos = x_pos.flatten(2).transpose(1, 2)
        return x_pos
        
class PPEGFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.ppeg = PPEG(dim)
        self.fusion = TransLayer(norm_layer=RMSNorm, dim=dim)
        
    def forward(self, x1, x2):
        # 1. Calculate required square dimensions
        N = x1.shape[1]
        H = W = int(np.ceil(np.sqrt(N)))
        
        # 2. Pad sequence to square shape (before calling PPEG)
        add_length = H * W - N
        x1_padded = torch.cat([x1, x1[:,:add_length,:]], dim=1)
        
        # 3. Apply PPEG (keep it concise)
        x1_pos = self.ppeg(x1_padded, H, W)
        
        # 4. Directly use position-encoded features for fusion, then truncate
        return self.fusion(torch.cat([x1_pos, x2], dim=1))[:, :N, :]