import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

import math

from models.model_utils import *
from nystrom_attention import NystromAttention
import admin_torch

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        #straight through estimator, y_hard used in feed forward, y_soft used in back propagation
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, y_soft

class MLP_Gate(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        super(MLP_Gate, self).__init__()
        self.bnum = branch_num
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.clsfer = nn.Linear(dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False, top_k = None):
        x1, x2 = self.fc1(x1), self.fc2(x2)
        x = x1.mean(dim=1) + x2.mean(dim=1)
        #logits = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=1) #old version
        logits, y_soft = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=-1)

        if top_k is not None:
            _, top_k_indices = torch.topk(y_soft, k=top_k, dim=-1)
            return top_k_indices

        return logits, y_soft

# CNN1: Concatenate + Convolution + Global Pooling + Fully Connected
class CNN1_Gate(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=512, hidden_dim=32):
        super(CNN1_Gate, self).__init__()
        self.bnum = branch_num
        self.conv = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        x = torch.cat([x1, x2], dim=1)  # batch_size*(m1+m2)*dim
        x = x.transpose(1, 2)  # batch_size*dim*(m1+m2)

        x = self.conv(x)
        x = self.global_pool(x).squeeze(-1)  # batch_size*hidden_dim
        logits = self.fc(x)  # batch_size*branch_num
        logits, y_soft = DiffSoftmax(logits, tau=temp, hard=hard, dim=-1)
        return logits, y_soft

# Independent Modal Role Encoding
class IndependentModalRoleEncoding(nn.Module):
    def __init__(self, proj_dim=256):
        super().__init__()
        # Orthogonally initialized role encoding
        self.rel_pos_bias = nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(2, proj_dim))
        )
        self.rel_pos_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, layer_idx=None):
        return self.rel_pos_bias * self.rel_pos_scale


class SignGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return torch.sign(scores)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MM_CosineGate(nn.Module):
    def __init__(self, branch_num, dim=256, proj_dim=256, norm_layer=RMSNorm, init_gates=0.5, max_experts=2, init_t=0.07, 
    layer_idx=0, role_encoding_type="Independent"):
        super().__init__()
        self.bnum = branch_num
        self.max_experts = max_experts
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        
        self.fc1 = nn.Sequential(
            nn.Linear(dim, proj_dim),
            norm_layer(proj_dim),
            nn.GELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(dim, proj_dim),
            norm_layer(proj_dim),
            nn.GELU(),
        )
        # Select role encoding method based on parameters
        self.layer_idx = layer_idx
        if role_encoding_type == "Independent":
            self.role_encoder = IndependentModalRoleEncoding(proj_dim)
        else:
            self.role_encoder = None

        
        # Adjust similarity matrix shape to match GAMoEGate
        self.register_parameter('sim_matrix', 
            nn.Parameter(torch.nn.init.orthogonal_(
                torch.empty(branch_num, proj_dim * 2)).T.contiguous(), 
            requires_grad=True)
        )
        
        # Threshold value for each expert
        self.gates = nn.Parameter(torch.zeros(branch_num))
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * math.log(1 / init_t))
        
        # Experts mask
        self.register_parameter('experts_mask', 
            nn.Parameter(torch.ones(branch_num), requires_grad=False)
        )
        
    def forward(self, x1, x2):
        # Process two inputs through MLP
        x1_processed, x2_processed = self.fc1(x1), self.fc2(x2)

        # Add role encoding based on whether role_encoder is None
        if self.role_encoder:
            # Calculate feature means and add modal role encoding
            rel_pos = self.role_encoder(self.layer_idx)
            x1_mean = x1_processed.mean(dim=1) + rel_pos[0]  # Add position encoding for features to be encoded
            x2_mean = x2_processed.mean(dim=1) + rel_pos[1]  # Add position encoding for reference features
        else:
            # Calculate means of the two processed features
            x1_mean, x2_mean = x1_processed.mean(dim=1), x2_processed.mean(dim=1)
        
        # Concatenate along feature dimension
        fused_feat = torch.cat([x1_mean, x2_mean], dim=-1)  # [batch_size, proj_dim*2]
        # Normalize concatenated features
        fused_feat = F.normalize(fused_feat, dim=-1)
        # Calculate similarity with experts, add temperature scaling
        sim_matrix_norm = F.normalize(self.sim_matrix, dim=0)
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = torch.sigmoid(torch.matmul(fused_feat, sim_matrix_norm) * logit_scale)
        
        # Apply experts mask
        logits = logits * self.experts_mask
        gates = torch.sigmoid(self.gates * logit_scale)
        
        if self.training:
            # Calculate difference for expert selection
            diff = logits - gates  # [batch_size, num_experts]
            
            logits = F.relu(logits - gates)
            logits = SignGrad.apply(logits)
            top_k = torch.sum(logits > 0, dim=1).to(torch.int)

            # Handle case when no expert is selected
            zero_mask = (top_k == 0)
            if zero_mask.any():
                # Select expert with maximum difference (closest to threshold)
                closest_expert = torch.argmax(diff[zero_mask], dim=1)
                logits[zero_mask, closest_expert] = 1.0
                top_k[zero_mask] = 1
        else:
            # Calculate difference for expert selection
            diff = logits - gates  # [batch_size, num_experts]
            new_logits = F.relu(diff)
            new_logits = SignGrad.apply(new_logits)
            top_k = torch.sum(new_logits > 0, dim=1).to(torch.int)


            # Handle case when no expert is selected
            zero_mask = (top_k == 0)
            if zero_mask.any():
                # Select expert with maximum difference (closest to threshold)
                closest_expert = torch.argmax(diff[zero_mask], dim=1)
                new_logits[zero_mask, closest_expert] = 1.0
                top_k[zero_mask] = 1
            
            # Handle case when selected experts exceed maximum
            over_max_mask = (top_k > self.max_experts)
            if over_max_mask.any():
                for idx in torch.where(over_max_mask)[0]:
                    # Get indices of selected experts for current sample
                    selected_idx = torch.where(new_logits[idx] > 0)[0]
                    # Select top-k experts based on difference
                    expert_diff = diff[idx, selected_idx]
                    _, top_indices = torch.topk(expert_diff, self.max_experts)
                    keep_idx = selected_idx[top_indices]
                    # Reset new_logits
                    new_logits[idx] = 0
                    new_logits[idx, keep_idx] = 1
                    top_k[idx] = self.max_experts
            
            logits = new_logits
        
        
        return logits, top_k
