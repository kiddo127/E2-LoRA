import copy
import torch
from torch import nn
from .linears import SimpleContinualLinear

import timm
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Attention
from timm.models.layers import Mlp


init_threshold = 0.9999
threshold_step = 1.0 - init_threshold

def get_principal_direction_svd(act_data):
    original_shape = act_data.shape
    feature_dim = original_shape[-1]
    if act_data.dim() > 2:
        act_data = act_data.reshape(-1, feature_dim)
    n_samples = act_data.shape[0]  # n
    feature_dim = act_data.shape[1]  # d
    act_data = act_data - torch.mean(act_data, dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(act_data, full_matrices=False)
    eigenvectors = Vh.T
    eigenvalues = S**2 / (n_samples - 1)
    if eigenvectors.shape[1] < feature_dim:
        full_eig = torch.eye(feature_dim, device=act_data.device, dtype=act_data.dtype)
        full_eig[:, :eigenvectors.shape[1]] = eigenvectors
        eigenvectors = full_eig
        full_eigenvalues = torch.zeros(feature_dim, device=eigenvalues.device, dtype=eigenvalues.dtype)
        full_eigenvalues[:len(eigenvalues)] = eigenvalues
        eigenvalues = full_eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    return eigenvalues.detach().cpu(), eigenvectors


def get_energy_threshold(eigenvalues, threshold=0.999):
    total_energy = eigenvalues.sum()
    cumulative_energy = torch.cumsum(eigenvalues, dim=0)
    mask = cumulative_energy <= (threshold * total_energy)
    num_to_keep = max(1, mask.sum().item() + 1)
    return num_to_keep


class LoRAMlp(Mlp):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., **kwargs):
        super().__init__(in_features, hidden_features=hidden_features, out_features=out_features, drop=drop, **kwargs)
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self._cur_task = 0
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.pca_lora = False
        self.role = 'student'
        
        self.lora_A1 = nn.ModuleList()
        self.lora_B1 = nn.ModuleList()
        self.eigenvalues = []

    def add_task(self):
        self._cur_task += 1
        
        old_rank = 0

        if self._cur_task > 1:
            saved_lora_B1_weight = []
            threshold = init_threshold
            for task in range(self._cur_task-1):
                rank_to_keep = get_energy_threshold(self.eigenvalues[task], threshold)
                old_rank += rank_to_keep
            while old_rank > (self._cur_task - 1) * self.hidden_features // self._cur_task:
                old_rank = 0
                threshold -= threshold_step
                for task in range(self._cur_task-1):
                    rank_to_keep = get_energy_threshold(self.eigenvalues[task], threshold)
                    old_rank += rank_to_keep

            old_rank = 0
            for task in range(self._cur_task - 1):
                lora_A1_weight = self.lora_A1[task].weight.data.clone()
                lora_B1_weight = self.lora_B1[task].weight.data.clone()
                
                rank_to_keep = get_energy_threshold(self.eigenvalues[task], threshold)
                old_rank += rank_to_keep
                
                self.lora_A1[task] = nn.Linear(self.in_features, rank_to_keep, bias=False)
                self.lora_B1[task] = nn.Linear(rank_to_keep, self.hidden_features, bias=False)
                
                self.lora_A1[task].weight.data.copy_(lora_A1_weight[:rank_to_keep, :])
                self.lora_B1[task].weight.data.copy_(lora_B1_weight[:, :rank_to_keep])
                
                saved_lora_B1_weight.append(lora_B1_weight[:, rank_to_keep:])
            
            print(f"old_rank: {old_rank}/{self.hidden_features}")

        r1 = self.hidden_features - old_rank
        lora_A1 = nn.Linear(self.in_features, r1, bias=False)
        lora_B1 = nn.Linear(r1, self.hidden_features, bias=False)

        if self._cur_task == 1:
            nn.init.zeros_(lora_A1.weight)
            lora_B1.weight.data = torch.eye(self.hidden_features)
        else:
            nn.init.zeros_(lora_A1.weight)
            lora_B1.weight.data.copy_(torch.cat(saved_lora_B1_weight, dim=1))

        self.lora_A1.append(lora_A1)
        self.lora_B1.append(lora_B1)
    
    def forward(self, x):
        tasks = self._cur_task
        if self.role == 'teacher':
            tasks = self._cur_task - 1

        weight_lora1 = torch.stack([
            torch.mm(self.lora_B1[t].weight, self.lora_A1[t].weight)
            for t in range(tasks)
        ], dim=0).sum(dim=0)

        lora_x = F.linear(x, weight_lora1)

        if self.pca_lora:
            r1 = self.lora_B1[-1].weight.shape[1]
            cur_weight = torch.mm(self.lora_B1[-1].weight, self.lora_A1[-1].weight)
            eigenvalues, eigenvectors = get_principal_direction_svd(F.linear(x, cur_weight))
            principal_direction = eigenvectors[:, :r1]
            self.eigenvalues.append(eigenvalues)
            self.lora_A1[-1].weight.data = principal_direction.T @ self.lora_B1[-1].weight.data @ self.lora_A1[-1].weight.data
            self.lora_B1[-1].weight.data = principal_direction


        x = self.fc1(x) + lora_x
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x


class LoRAAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=True, **kwargs):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **kwargs)

        self._cur_task = 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pca_lora = False
        self.role = 'student'

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()
        self.eigenvalues = []
        
    def add_task(self):
        self._cur_task += 1

        dim = self.dim
        qkv_dim = self.dim * 3

        old_rank = 0

        if self._cur_task > 1:
            saved_lora_B_weight = []
            threshold = init_threshold
            for task in range(self._cur_task-1):
                rank_to_keep = get_energy_threshold(self.eigenvalues[task], threshold)
                old_rank += rank_to_keep
            while old_rank > (self._cur_task-1) * qkv_dim // self._cur_task:
                old_rank = 0
                threshold -= threshold_step
                for task in range(self._cur_task-1):
                    rank_to_keep = get_energy_threshold(self.eigenvalues[task], threshold)
                    old_rank += rank_to_keep
            old_rank = 0
            for task in range(self._cur_task-1):
                lora_A_weight = self.lora_A[task].weight.data.clone()
                lora_B_weight = self.lora_B[task].weight.data.clone()

                rank_to_keep = get_energy_threshold(self.eigenvalues[task], threshold)
                old_rank += rank_to_keep

                self.lora_A[task] = nn.Linear(dim, rank_to_keep, bias=False)
                self.lora_B[task] = nn.Linear(rank_to_keep, qkv_dim, bias=False)

                self.lora_A[task].weight.data.copy_(lora_A_weight[:rank_to_keep, :])
                self.lora_B[task].weight.data.copy_(lora_B_weight[:, :rank_to_keep])

                saved_lora_B_weight.append(lora_B_weight[:, rank_to_keep:])

            print(f"old_rank: {old_rank}/{qkv_dim}")

        r = qkv_dim - old_rank
        lora_A = nn.Linear(dim, r, bias=False)
        lora_B = nn.Linear(r, qkv_dim, bias=False)

        if self._cur_task == 1:
            nn.init.zeros_(lora_A.weight)
            lora_B.weight.data = torch.eye(qkv_dim)
        else:
            nn.init.zeros_(lora_A.weight)
            lora_B.weight.data.copy_(torch.cat(saved_lora_B_weight, dim=1))

        self.lora_A.append(lora_A)
        self.lora_B.append(lora_B)
    
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        tasks = self._cur_task
        if self.role == 'teacher':
            tasks = self._cur_task - 1

        weight_qkv = torch.stack([
            torch.mm(self.lora_B[t].weight, self.lora_A[t].weight)
            for t in range(tasks)
        ], dim=0).sum(dim=0)

        lora_qkv = F.linear(x, weight_qkv).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = qkv + lora_qkv

        if self.pca_lora:
            r = self.lora_B[-1].weight.shape[1]

            cur_weight_qkv = torch.mm(self.lora_B[-1].weight, self.lora_A[-1].weight)
            cur_task_qkv = F.linear(x, cur_weight_qkv)
            eigenvalues, eigenvectors = get_principal_direction_svd(cur_task_qkv)
            principal_direction = eigenvectors[:, :r]
            self.eigenvalues.append(eigenvalues)

            self.lora_A[-1].weight.data = principal_direction.T @ self.lora_B[-1].weight.data @ self.lora_A[-1].weight.data
            self.lora_B[-1].weight.data = principal_direction


        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


def get_backbone(args, pretrained=True):
    name = args["backbone"].lower()
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.backbone = get_backbone(args, pretrained)

        for block in self.backbone.blocks:
            attn = block.attn
            dim = attn.qkv.in_features
            num_heads = attn.num_heads
            qkv_bias = attn.qkv.bias is not None
            attn_drop_rate = attn.attn_drop.p
            proj_drop_rate = attn.proj_drop.p

            lora_attn = LoRAAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
            )

            lora_attn.qkv.weight.data = attn.qkv.weight.data.clone()
            if qkv_bias:
                lora_attn.qkv.bias.data = attn.qkv.bias.data.clone()

            lora_attn.proj.weight.data = attn.proj.weight.data.clone()
            if attn.proj.bias is not None:
                lora_attn.proj.bias.data = attn.proj.bias.data.clone()

            block.attn = lora_attn

        for block in self.backbone.blocks:
            mlp = block.mlp
            in_features = mlp.fc1.in_features
            hidden_features = mlp.fc1.out_features
            out_features = mlp.fc2.out_features
            drop1_rate = mlp.drop1.p if hasattr(mlp, 'drop1') else 0.0
            drop2_rate = mlp.drop2.p if hasattr(mlp, 'drop2') else 0.0
            drop_rate = max(drop1_rate, drop2_rate)

            lora_mlp = LoRAMlp(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                drop=drop_rate,
            )

            lora_mlp.fc1.weight.data = mlp.fc1.weight.data.clone()
            if mlp.fc1.bias is not None:
                lora_mlp.fc1.bias.data = mlp.fc1.bias.data.clone()

            lora_mlp.fc2.weight.data = mlp.fc2.weight.data.clone()
            if mlp.fc2.bias is not None:
                lora_mlp.fc2.bias.data = mlp.fc2.bias.data.clone()

            block.mlp = lora_mlp
        self.fc = None
        self._device = args["device"][0]

        self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class E2LoRANet(BaseNet):

    def __init__(self, args, pretrained=True, fc_with_ln=False):
        super().__init__(args, pretrained)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln


    def extract_layerwise_vector(self, x, pool=True):
        with torch.no_grad():
            features = self.backbone(x, layer_feat=True)['features']
        for f_i in range(len(features)):
            if pool:
                features[f_i] = features[f_i].mean(1).cpu().numpy() 
            else:
                features[f_i] = features[f_i][:, 0].cpu().numpy() 
        return features


    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleContinualLinear(in_dim, out_dim)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})

        return out

