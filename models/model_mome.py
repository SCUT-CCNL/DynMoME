import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *
import admin_torch
import sys

#import Gating Network
from models.model_Gating import *


class SNNFusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class DropX2Fusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()

    def forward(self, x1, x2):
        return x1

class MoME(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256,):
        super().__init__()
        self.TransFusion = TransFusion(norm_layer, dim)
        self.BottleneckTransFusion = BottleneckTransFusion(n_bottlenecks, norm_layer, dim)
        self.SNNFusion = SNNFusion(norm_layer, dim)
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        self.routing_network = MLP_Gate(4, dim=dim)
        self.routing_dict = {
            0: self.TransFusion,
            1: self.BottleneckTransFusion,
            2: self.SNNFusion,
            3: self.DropX2Fusion,
        }

    def forward(self, x1, x2, hard=False):
        logits, y_soft = self.routing_network(x1, x2, hard)
        corresponding_net_id = None
        if hard:
            corresponding_net_id = torch.argmax(logits, dim=1).item()
            x = self.routing_dict[corresponding_net_id](x1, x2)
        else:
            x = torch.zeros_like(x1)
            
            for branch_id, branch in self.routing_dict.items():
                x += branch(x1, x2)
            '''
            # use y_soft to do weighted sum
            for branch_id, branch in self.routing_dict.items():
                x += y_soft[:, branch_id] * branch(x1, x2)
            '''
        return x, corresponding_net_id
    
class MoMETransformer(nn.Module):
    def __init__(self, n_bottlenecks, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25, soft_mode=False):
        super(MoMETransformer, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}
        #self.size_dict_WSI = {"small": [256, 512, 512], "big": [1024, 512, 384]} #brca condensed-special
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}

        #route mode: true for sofr
        self.soft_mode = soft_mode
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### MoMEs
        self.MoME_genom1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_patho1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])

        if not self.soft_mode:
            self.MoME_genom2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
            self.MoME_patho2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])

        ### Classifier
        self.multi_layer1 = TransLayer(dim=size[2])
        self.cls_multimodal = torch.rand((1, size[2])).cuda()
        self.classifier = nn.Linear(size[2], n_classes)

        ###grade_classifier
        self.classifier_grade = nn.Linear(size[2], 3)
        self.act_grad = nn.Softmax(dim=1)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        
        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        h_path_bag = h_path_bag.unsqueeze(0)
        h_omic_bag = h_omic_bag.unsqueeze(0)

        h_path_bag, corresponding_net_id_patho1 = self.MoME_patho1(h_path_bag, h_omic_bag, hard=not self.soft_mode)
        h_omic_bag, corresponding_net_id_genom1 = self.MoME_genom1(h_omic_bag, h_path_bag, hard=not self.soft_mode)

        if not self.soft_mode:
            h_path_bag, corresponding_net_id_patho2 = self.MoME_patho2(h_path_bag, h_omic_bag, hard=not self.soft_mode)
            h_omic_bag, corresponding_net_id_genom2 = self.MoME_genom2(h_omic_bag, h_path_bag, hard=not self.soft_mode)

        h_path_bag = h_path_bag.squeeze()
        h_omic_bag = h_omic_bag.squeeze()
        
        if h_path_bag.dim() == 1:
            h_path_bag = h_path_bag.unsqueeze(0)

        h_multi = torch.cat([self.cls_multimodal, h_path_bag, h_omic_bag], dim=0).unsqueeze(0)
        h = self.multi_layer1(h_multi)[:,0,:]


        ### Survival Layer
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        ### Grade Layer
        #hazards_grade = self.classifier_grade(h).unsqueeze(0)
        hazards_grade = self.classifier_grade(h)
        hazards_grade = self.act_grad(hazards_grade)

        attention_scores = {}

        expert_choices = {"corresponding_net_id_patho1": corresponding_net_id_patho1,
                          "corresponding_net_id_genom1": corresponding_net_id_genom1,
                          "corresponding_net_id_fuse": -1}
        
        return hazards, S, Y_hat, attention_scores, hazards_grade, None, None, expert_choices