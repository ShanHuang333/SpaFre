import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

from SpaFre import DualSpaFre
from utils import Transfer_pytorch_Data
from unifilter_utils import edgeindex_construct, load_dataset


def train_Dual_SpaFre(adata, hidden_dims=[512, 30], n_epochs=2000, lr=0.001,
                       key_added='SpaFre', gradient_clipping=5., weight_decay=0.0001, random_seed=0,
                       # UniFilter 参数
                       K=4, #tau=0.5,
                       init_tau=0.5, learnable_tau=True,
                       # 修改：homo_ratio 现在是初始值，可学习
                       init_homo_ratio=0.6, learnable_homo_ratio=True,
                       # gating_weight=0.8,
                       init_gating_weight=0.8, learnable_gating_weight=True,
                       save_loss=False, save_reconstrction=False,
                       device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    data = Transfer_pytorch_Data(adata_Vars)

    # ==========================================
    # 2. 构建图传播算子 LP
    # ==========================================
    print("--- Constructing Graph Propagation Operator ---")
    num_nodes = data.x.shape[0]
    feat_dim = data.x.shape[1]

    # 构建传播算子 LP (D^{-1/2} A D^{-1/2})
    LP, _, _ = edgeindex_construct(data.edge_index, num_nodes)

    # 转换为 Tensor 并移至 GPU（兼容 Tensor 和 scipy 稀疏矩阵）
    if hasattr(LP, 'toarray'):
        LP_tensor = torch.tensor(LP.toarray(), dtype=torch.float32).to(device)
    else:
        LP_tensor = LP.float().to(device)
    data = data.to(device)

    # ==========================================
    # 3. 初始化双流模型（传入新参数）
    # ==========================================
    model = DualSpaFre(
        hidden_dims=[data.x.shape[1]] + hidden_dims,
        K=K,
        # tau=tau,
        # gating_weight=gating_weight,
        init_tau=init_tau,
        learnable_tau=learnable_tau,
        init_homo_ratio=init_homo_ratio,  # 初始值
        learnable_homo_ratio=learnable_homo_ratio,  # 是否可学习
        init_gating_weight = init_gating_weight,  # 新增
        learnable_gating_weight = learnable_gating_weight  # 新增
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 打印初始 homo_ratio
    print(f"Initial tau: {model.tau.item():.4f}")
    print(f"Initial homo_ratio: {model.homo_ratio.item():.4f}")
    print(f"Initial gating_weight: {model.gating_weight.item():.4f}")

    # ==========================================
    # 4. 训练循环
    # ==========================================
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()

        # forward 现在传入 LP_tensor 而不是 spectral_features
        z, out, alpha = model(data.x, data.edge_index, LP_tensor)

        # 损失计算
        loss_recon = F.mse_loss(data.x, out)

        # 空间一致性损失
        z_src = z[data.edge_index[0]]
        z_dst = z[data.edge_index[1]]
        loss_spatial = F.mse_loss(z_src, z_dst)

        loss = loss_recon + 0.1 * loss_spatial
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        # 可选：每隔一段时间打印 homo_ratio 的变化
        # if epoch % 500 == 0 and learnable_homo_ratio:
        #     print(f"Epoch {epoch}: tau = {model.tau.item():.4f}, "
        #           f"homo_ratio = {model.homo_ratio.item():.4f}, "
        #           f"gating_weight = {model.gating_weight.item():.4f}")

        if learnable_homo_ratio:
            print(f"Epoch {epoch}: tau = {model.tau.item():.6f}, "
            f"homo_ratio = {model.homo_ratio.item():.6f}, "
            f"gating_weight = {model.gating_weight.item():.6f}")

    # ==========================================
    # 5. 推理与保存
    # ==========================================
    model.eval()
    with torch.no_grad():
        z, out, alpha = model(data.x, data.edge_index, LP_tensor)

    SpaFre_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = SpaFre_rep

    if save_loss:
        adata.uns['SpaFre_loss'] = loss.item()
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX < 0] = 0
        adata.layers['SpaFre_ReX'] = ReX

    # 保存门控系数
    if isinstance(alpha, torch.Tensor):
        adata.obs['uni_gate_alpha'] = alpha.to('cpu').detach().numpy()
    else:
        adata.obs['uni_gate_alpha'] = alpha

    # ==========================================
    # 新增：保存学习到的 homo_ratio
    # ==========================================
    adata.uns['learned_tau'] = model.tau.item()
    adata.uns['learned_homo_ratio'] = model.homo_ratio.item()
    adata.uns['learned_gating_weight'] = model.gating_weight.item()

    print(f"Final learned tau: {model.tau.item():.4f}")
    print(f"Final learned homo_ratio: {model.homo_ratio.item():.4f}")
    print(f"Final learned gating_weight: {model.gating_weight.item():.4f}")

    # 保存 gating_weight 到 obs（用于可视化）
    if isinstance(alpha, torch.Tensor):
        adata.obs['uni_gate_alpha'] = alpha.to('cpu').detach().numpy()
    else:
        adata.obs['uni_gate_alpha'] = alpha

    # 提取 UniFilter 频率权重
    found_weights = False
    for name, param in model.named_parameters():
        if 'comb_weight' in name:
            raw_weights = param.detach().cpu().numpy().flatten()
            exp_w = np.exp(raw_weights - np.max(raw_weights))
            comb_weights_val = exp_w / exp_w.sum()

            print(f"SUCCESS: Extracted UniFilter Weights: {comb_weights_val}")
            adata.uns['UniFilter_weights'] = comb_weights_val
            found_weights = True
            break

    if not found_weights:
        print("WARNING: Could not find 'comb_weight' in model parameters.")

    return adata