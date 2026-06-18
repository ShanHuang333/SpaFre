import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from gat_conv import GATConv
from unifilter_utils import load_dataset
from unifilter_utils import Combination
import math


class DualSpaFre(torch.nn.Module):
    def __init__(self, hidden_dims, K=4, #tau=0.5, #gating_weight=0.8,
                 init_tau=0.5, learnable_tau=True,
                 init_gating_weight=0.8, learnable_gating_weight=True,  # 修改
                 init_homo_ratio=0.6, learnable_homo_ratio=True):  # 新增参数
        super(DualSpaFre, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims

        # ---------------------------
        # 新增：可学习的 homo_ratio
        # ---------------------------
        self.learnable_homo_ratio = learnable_homo_ratio
        if learnable_homo_ratio:
            # 使用 sigmoid 的反函数初始化，确保 sigmoid 后得到 init_homo_ratio
            # sigmoid(x) = init_homo_ratio => x = log(init_homo_ratio / (1 - init_homo_ratio))
            init_val = math.log(init_homo_ratio / (1 - init_homo_ratio + 1e-8))
            self._homo_ratio_raw = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        else:
            self.register_buffer('_homo_ratio_fixed', torch.tensor(init_homo_ratio, dtype=torch.float32))

        # ---------------------------
        # 新增：可学习的 gating_weight
        # ---------------------------
        self.learnable_gating_weight = learnable_gating_weight
        if learnable_gating_weight:
            init_val_gw = math.log(init_gating_weight / (1 - init_gating_weight + 1e-8))
            self._gating_weight_raw = nn.Parameter(torch.tensor(init_val_gw, dtype=torch.float32))
        else:
            self.register_buffer('_gating_weight_fixed', torch.tensor(init_gating_weight, dtype=torch.float32))

        # ---------------------------
        # 新增：可学习的 tau
        # ---------------------------
        self.learnable_tau = learnable_tau
        if learnable_tau:
            # tau 也用 sigmoid 约束在 (0, 1)
            init_val_tau = math.log(init_tau / (1 - init_tau + 1e-8))
            self._tau_raw = nn.Parameter(torch.tensor(init_val_tau, dtype=torch.float32))
        else:
            self.register_buffer('_tau_fixed', torch.tensor(init_tau, dtype=torch.float32))

        self.K = K

        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)


        self.uni_comb = Combination(channels=in_dim, level=K + 1, dropout=0.5)
        self.uni_lin1 = nn.Linear(in_dim, num_hidden)
        self.uni_lin2 = nn.Linear(num_hidden, out_dim)

        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    @property
    def homo_ratio(self):
        """返回当前的 homo_ratio 值（经过 sigmoid 约束在 0-1 之间）"""
        if self.learnable_homo_ratio:
            # 先 clamp 原始值，防止 sigmoid 输入过大/过小
            raw_clamped = torch.clamp(self._homo_ratio_raw, min=-5.0, max=5.0)
            return torch.sigmoid(self._homo_ratio_raw)
        else:
            return self._homo_ratio_fixed

    @property
    def gating_weight(self):
        if self.learnable_gating_weight:
            raw_clamped = torch.clamp(self._homo_ratio_raw, min=-5.0, max=5.0)
            return torch.sigmoid(self._gating_weight_raw)
        else:
            return self._gating_weight_fixed

    @property
    def tau(self):
        if self.learnable_tau:
            raw_clamped = torch.clamp(self._homo_ratio_raw, min=-5.0, max=5.0)
            return torch.sigmoid(self._tau_raw)
        else:
            return self._tau_fixed

    def forward(self, features, edge_index, LP):
        current_homo_ratio = self.homo_ratio
        current_tau = self.tau
        spectral_features = self._compute_spectral_features(features, LP, current_homo_ratio, current_tau)

        h1_gat = F.elu(self.conv1(features, edge_index))
        h2_gat = self.conv2(h1_gat, edge_index, attention=False)

        h_uni_combined = self.uni_comb(spectral_features)
        h1_uni = F.elu(self.uni_lin1(h_uni_combined))
        h2_uni = self.uni_lin2(h1_uni)

        current_gating_weight = self.gating_weight
        z = current_gating_weight * h2_gat + (1.0 - current_gating_weight) * h2_uni

        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)

        h3 = F.elu(self.conv3(z, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return z, h4, current_gating_weight #alpha

    def _compute_spectral_features(self, feat, LP, homo_ratio, tau):
        num_nodes, dim = feat.shape
        K = self.K
        device = feat.device
        dtype = feat.dtype

        if LP.is_sparse:
            LP_dense = LP.to_dense()
        else:
            LP_dense = LP
        LP_dense = LP_dense.to(dtype=dtype, device=device)

        homo_ratio_safe = torch.clamp(homo_ratio, min=0.1, max=0.9)
        tau_safe = torch.clamp(tau, min=0.1, max=0.9)

        cosval = torch.cos(math.pi * (1.0 - homo_ratio_safe) / 2.0)
        cosval = torch.clamp(cosval, min=0.15, max=0.99)

        norm = torch.norm(feat, dim=0)
        norm = torch.clamp(norm, min=1e-8)
        last = feat / norm
        second = torch.zeros_like(last)
        basis_sum = torch.zeros_like(last)

        HM = feat.clone()
        basis_sum = basis_sum + last
        features = [feat]

        for k in range(1, K + 1):
            V_k = torch.mm(LP_dense, last)
            HM = torch.mm(LP_dense, HM)

            project_1 = torch.einsum('nd,nd->d', V_k, last)
            project_2 = torch.einsum('nd,nd->d', V_k, second)
            V_k = V_k - (project_1 * last + project_2 * second)

            norm = torch.norm(V_k, dim=0)
            norm = torch.clamp(norm, min=1e-8)
            V_k = V_k / norm

            H_k = basis_sum / k
            dot_product = torch.einsum('nd,nd->d', H_k, features[-1])

            ratio = dot_product / cosval
            ratio = torch.clamp(ratio, min=-5.0, max=5.0)

            term1 = torch.square(ratio)
            term2 = ((k - 1) * cosval + 1) / k
            inside_sqrt = term1 - term2

            inside_sqrt_safe = torch.clamp(inside_sqrt, min=0.0)

            Tf = torch.sqrt(inside_sqrt_safe + 0.01)
            Tf = torch.clamp(Tf, min=0.0, max=3.0)
            H_k = H_k + Tf * V_k
            norm = torch.norm(H_k, dim=0)
            norm = torch.clamp(norm, min=1e-8)
            H_k = H_k / norm

            norm_HM = torch.norm(HM, dim=0)
            norm_HM = torch.clamp(norm_HM, min=1e-8)
            HM_normalized = HM / norm_HM

            fused_feature = HM_normalized * tau_safe + H_k * (1.0 - tau_safe)
            if torch.isnan(fused_feature).any():
                fused_feature = torch.nan_to_num(fused_feature, nan=0.0)

            features.append(fused_feature)

            basis_sum = basis_sum + H_k
            second = last
            last = V_k

        spectral_features = torch.stack(features, dim=1)

        if torch.isnan(spectral_features).any():
            spectral_features = torch.nan_to_num(spectral_features, nan=0.0)

        return spectral_features