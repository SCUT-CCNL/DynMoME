import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *
import admin_torch
import sys

# import MultiheadAttention
from models_coattn import *

# import PPEGFusion
from models.model_PPEG import *

# import Gating Network
from models.model_Gating import *

class CoAFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.25):
        super(CoAFusion, self).__init__()
        self.coattn = MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x1, x2):
        # x1: (1, n1, 512), x2: (1, n2, 512)
        # x1 as query，x2 as both key and value
        x1 = x1.transpose(0, 1)  # (n1, 1, 512)
        x2 = x2.transpose(0, 1)  # (n2, 1, 512)
        
        attn_output, _ = self.coattn(x1, x2, x2)
        
        attn_output = attn_output.transpose(0, 1)  # (1, n1, 512)
        
        fused_feature = self.fusion_layer(attn_output)
        
        return fused_feature
        
class SNNFusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class ZeroFusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()

    def forward(self, x1, x2):
        return x1
        
class MCMoE(nn.Module): 
    # Dynamic Multi-modal Cosine Mixture-of-Experts Layer
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256, max_experts=2, 
    ablation_expert_id=None, expert_id = 0, gating_network = "CosMLP", role_encoding_type="Independent", top_k=1):
        super().__init__()
        # Initialize experts
        self.SNNFusion = SNNFusion(norm_layer, dim)
        self.ZeroFusion = ZeroFusion(norm_layer, dim)
        self.CoAFusion = CoAFusion(dim=dim)
        self.PPEGFusion = PPEGFusion(dim=dim)

        self.gating_network = gating_network
        
        self.expert_id = expert_id
        self.top_k = top_k
        experts = [
            self.CoAFusion,
            self.SNNFusion,
            self.PPEGFusion,
            self.ZeroFusion,
        ]

        # If specified, remove expert by ID from the list
        if ablation_expert_id is not None and 0 <= ablation_expert_id < len(experts):
            del experts[ablation_expert_id]

        # Rebuild dictionary to ensure continuous IDs
        self.routing_dict = {i: expert for i, expert in enumerate(experts)}
        
        # Add load counter
        self.register_buffer('expert_counts', torch.zeros(len(self.routing_dict)), persistent=False)
        self.register_buffer('total_samples', torch.zeros(1), persistent=False)
        #self.accumulation_steps = 32  #Accumulation update steps

        # Add expert activation distribution statistics
        self.max_experts = max_experts
        self.register_buffer('expert_k_counts', torch.zeros(max_experts + 1), persistent=False)
        self.register_buffer('total_samples_k', torch.zeros(1), persistent=False)
        # Initialize gating network
        if self.gating_network == "CosMLP":
            self.routing_network = MM_CosineGate(
                branch_num=len(self.routing_dict),
                dim=dim,
                max_experts=max_experts,
                role_encoding_type=role_encoding_type
            )
        elif self.gating_network == "MLP":
            self.routing_network = MLP_Gate(len(self.routing_dict), dim=dim)
        else: 
            self.routing_network = None
        
        # For calculating load balancing loss
        self.num_experts = len(self.routing_dict)
        
    def _update_load_counts(self, logits):
        """Update expert load counts
        Args:
            logits: tensor of shape [batch_size, num_experts]
        """
        # Count how many times each expert is activated in the entire batch
        self.expert_counts += (logits > 0).float().sum(dim=0)  # Sum along expert dimension
        self.total_samples += logits.size(0)  # Increase total batch samples

    def _update_k_distribution(self, top_k):
        """Update expert activation distribution statistics"""
        for k in range(self.max_experts + 1):
            self.expert_k_counts[k] += torch.sum(top_k == k)

        #print(f"top_k: {top_k}")
        self.total_samples_k += len(top_k)        

    def get_gating_params(self):
        """Get similarity matrix and threshold values of the gating network"""
        return {
            'sim_matrix': self.routing_network.sim_matrix.data.clone(),
            'activation_gates': self.routing_network.gates.data.clone(),
            'expert_k_counts': self.expert_k_counts.clone(),
            'expert_counts': self.expert_counts.clone()
        }

    def forward(self, x1, x2):
        # Initialize output
        outputs = torch.zeros_like(x1)
        if self.gating_network == "CosMLP":
            # Get gating network output
            logits, top_k = self.routing_network(x1, x2)  # logits: [1, num_experts]
            
            # Find selected experts (logits > 0)
            selected_experts = torch.where(logits[0] > 0)[0]  # Use logits[0] because batch_size=1
            num_selected = selected_experts.size(0)  # Get number of selected experts
            # Only process selected experts
            for expert_id in selected_experts:
                # Get expert weight
                expert_weight = logits[0, expert_id].unsqueeze(-1)  # [1]
                # Compute expert output and weight
                expert_output = self.routing_dict[expert_id.item()](x1, x2)
                outputs += expert_weight * expert_output
            outputs /= num_selected
            
            # Compute auxiliary loss
            self._update_load_counts(logits)

            # Update expert activation distribution
            self._update_k_distribution(top_k)
        elif self.gating_network == "MLP":
            selected_experts = self.routing_network(x1, x2, temp=1.0, top_k=self.top_k) # selected_experts: [batch_size, k]
            #print(f"selected_experts: {selected_experts}")
            outputs = 0
            for i in range(self.top_k):
                expert_id = selected_experts[0, i]  # Get the index of the i-th expert
                expert_output = self.routing_dict[expert_id.item()](x1, x2)
                outputs += expert_output
            outputs = outputs / self.top_k  # Simple average
        else:
            #print("hello, I'm single expert CoA")
            outputs = self.routing_dict[self.expert_id](x1, x2)
        balance_loss = 0
        
        return outputs, balance_loss, None

    def load_state_dict(self, state_dict, strict=True):
        # Remove all MoME layer statistics related buffers from state_dict
        new_state_dict = {}
        for key, value in state_dict.items():
            # Skip buffers containing statistical information
            if any(x in key for x in ['expert_counts', 'total_samples', 'expert_k_counts', 'total_samples_k']):
                continue
            new_state_dict[key] = value
            
        # Call parent class's load_state_dict
        return super().load_state_dict(new_state_dict, strict=False)

class DynMoME(nn.Module):
    def __init__(self, n_bottlenecks, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,
                 gating_network='MLP', expert_idx=0, ablation_expert_id=None,
                 mof_gating_network='MLP', mof_expert_idx=0, mof_ablation_expert_id=None, 
                 max_experts = 2, route_mode = True, encoding_type='AE', encoding_pairs=2, role_encoding_type="Independent", top_k = 1):
        super(DynMoME, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}

        # Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        # WSI FC Layer
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        # MoME Layers
        self.encoding_type = encoding_type
        self.encoding_pairs = encoding_pairs
        self.mome_patho_layers = nn.ModuleList()
        self.mome_genom_layers = nn.ModuleList()
        self.routing_network = gating_network
        self.route_mode = route_mode
        self.top_k = top_k
        for _ in range(encoding_pairs):
            self.mome_patho_layers.append(
                MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], 
                      ablation_expert_id=ablation_expert_id, max_experts=max_experts, 
                      expert_id=expert_idx, gating_network=gating_network, role_encoding_type=role_encoding_type, top_k=top_k)
            )
            self.mome_genom_layers.append(
                MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], 
                      ablation_expert_id=ablation_expert_id, max_experts=max_experts,
                      expert_id=expert_idx, gating_network=gating_network, role_encoding_type=role_encoding_type, top_k=top_k)
            )

        #SA Layer for final fusion
        self.SA_path = SelfAttention(dim=size[2])
        self.SA_omic = SelfAttention(dim=size[2])

        # survival Classifier
        self.classifier = nn.Linear(size[2] * 2, n_classes) 
        # grade classifier
        self.classifier_grade = nn.Linear(size[2] * 2, 3) 
        self.act_grad = nn.Softmax(dim=1)

    def get_gating_params(self):
        if self.routing_network == 'CosMLP':
            patho1_gp = self.mome_patho_layers[0].get_gating_params()
            genom1_gp = self.mome_genom_layers[0].get_gating_params()
            patho2_gp = self.mome_patho_layers[1].get_gating_params()
            genom2_gp = self.mome_genom_layers[1].get_gating_params()
            return patho1_gp, genom1_gp, patho2_gp, genom2_gp
        return None, None, None, None

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]

        # Feature extraction
        h_path_bag = self.wsi_net(x_path)
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic)
        
        h_path_bag = h_path_bag.unsqueeze(0)
        h_omic_bag = h_omic_bag.unsqueeze(0)

        #forward through moe
        total_time_cost = 0
        h_path_list = [h_path_bag]  # Store intermediate results in list
        h_omic_list = [h_omic_bag]
        for i in range(0, len(self.mome_patho_layers)):
            if self.encoding_type == 'BPE':  # Bidirectional Parallel Encoding
                # Process each encoding layer pair in parallel
                h_path_new, cost_p, _ = self.mome_patho_layers[i](h_path_list[-1], h_omic_list[-1])
                h_omic_new, cost_g, _ = self.mome_genom_layers[i](h_omic_list[-1], h_path_list[-1])
            else:  # AE: Alternating Encoding
                if i % 2 == 0: # Even: Path first, then Genomic
                    h_path_new, cost_p, _ = self.mome_patho_layers[i](h_path_list[-1], h_omic_list[-1])
                    h_omic_new, cost_g, _ = self.mome_genom_layers[i](h_omic_list[-1], h_path_new)
                else: # Odd: Genomic first, then Path
                    h_omic_new, cost_g, _ = self.mome_genom_layers[i](h_omic_list[-1], h_path_list[-1])
                    h_path_new, cost_p, _ = self.mome_patho_layers[i](h_path_list[-1], h_omic_new)
            h_path_list.append(h_path_new)
            h_omic_list.append(h_omic_new)
            total_time_cost += cost_p + cost_g
            
        # Use separate SA for path and omic, then concat
        h_path, _ = self.SA_path(h_path_list[-1])
        h_omic, _ = self.SA_omic(h_omic_list[-1])
        h = torch.cat([h_path, h_omic], dim=-1) 
        jacobian_loss, corresponding_net_id_fuse = None, -1
        # Prediction
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        hazards_grade = self.classifier_grade(h)
        hazards_grade = self.act_grad(hazards_grade)
        
        attention_scores = {}
        expert_choices = {}
        
        return hazards, S, Y_hat, attention_scores, hazards_grade, jacobian_loss, total_time_cost, expert_choices
    
    def load_state_dict(self, state_dict, strict=True):
        # Remove all MoME layer statistics related buffers from state_dict
        new_state_dict = {}
        for key, value in state_dict.items():
            # Skip buffers containing statistical information
            if any(x in key for x in ['expert_counts', 'total_samples', 'expert_k_counts', 'total_samples_k']):
                continue
            new_state_dict[key] = value
            
        # Call parent class's load_state_dict
        return super().load_state_dict(new_state_dict, strict=False)
