import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import torch
import sys
import pickle as pkl
from time import perf_counter
import struct
import gc
import scipy.special as ss
from scipy.sparse import csr_matrix, coo_matrix
import time

from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, is_undirected

import torch_geometric.transforms as T
import math

import torch.nn as nn
import torch.nn.functional as F

# --- 来自 UniFilter/models.py 的 Combination 类 ---
#这个类就是一个可学习的加权求和器
class Combination(nn.Module):
    def __init__(self, channels, level, dropout=0.5):
        super(Combination, self).__init__()
        self.dropout = dropout
        self.K = level
        # 定义可学习的基底权重，形状为 (1, K, 1)，利用广播机制对所有节点生效
        self.comb_weight = nn.Parameter(torch.ones((1, level, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / self.K
        # 初始化为均匀分布
        TEMP = np.random.uniform(bound, bound, self.K)
        self.comb_weight = nn.Parameter(torch.FloatTensor(TEMP).view(-1, self.K, 1))  #模型只学习“哪一个频段”更重要

    def forward(self, x):
        # 输入 x 形状: (Batch, K, Dim)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x * self.comb_weight  # 加权
        x = torch.sum(x, dim=1)   # 求和融合
        return x


def random_splits(labels, num_classes, percls_trn=20, val_lb=500, seed=12591):   
    
    num_nodes=labels.shape[0]
    index=[i for i in range(0,num_nodes)]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    train_idx=np.array(train_idx)              
    rest_index = [i for i in index if i not in train_idx]
    val_idx=np.array(rnd_state.choice(rest_index,val_lb,replace=False))
    test_idx=np.array([i for i in rest_index if i not in val_idx])    
    return train_idx, val_idx, test_idx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def PropMatrix(adj):   #计算图的对称归一化邻接矩阵
    # 1. 计算度 (Degree)
    # adj.sum(1) 对稀疏矩阵按行求和。
    # 对于邻接矩阵，第 i 行的和就是节点 i 的度 (Degree)，记为 D_ii
    row_sum = np.array(adj.sum(1))
    # 2. 计算度的负二分之一次方 (D^{-1/2})
    # np.power(x, -0.5) 等价于 1 / sqrt(x)
    # 这是为了后面做对称归一化做准备
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    # 3. 内存回收
    # 删除临时变量 row_sum 并强制回收内存，防止大图占用过多内存
    del row_sum
    gc.collect()
    # 4. 处理孤立节点 (除以0的情况)
    # 如果某个节点度为0 (row_sum=0)，那么 0^{-0.5} 会变成无穷大 (inf)。
    # 这里将无穷大修正为 0，防止数值错误。
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    t=time.time()
    # 5. 构建对角矩阵 D^{-1/2}
    # sp.diags 将 1D 数组转换为稀疏对角矩阵
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # 6. 执行对称归一化
    # 计算公式： D^{-1/2} * A * D^{-1/2}
    # .dot() 是稀疏矩阵乘法
    # 这一步将邻接矩阵进行了缩放：
    # - 原来的 A_ij = 1 (如果有边)
    # - 现在的 A_ij = 1 / sqrt(d_i * d_j)
    adj=d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) 
    print('matrix multiplication time: ', time.time()-t)
    return adj

def edgeindex_construct(edge_index, num_nodes):
    # 将输入的 edge_index 转换为 PyTorch 的 LongTensor (整型张量)
    # 这是为了配合 PyTorch Geometric (PyG) 的工具函数使用
    edge_index=torch.LongTensor(edge_index)
    # 检查图是否是无向图 (is_undirected)
    # UniFilter 基于谱图理论，要求拉普拉斯矩阵是对称的，因此必须是无向图
    if not is_undirected(edge_index):
        # 如果是有向图，调用 to_undirected 自动添加反向边，使其变为无向图
        edge_index = to_undirected(edge_index)
    # 将 Tensor 转回 numpy 数组
    # 因为接下来的稀疏矩阵构建使用的是 scipy.sparse 库，它需要 numpy 格式
    edge_index=edge_index.numpy()

    # 获取边的总数量 (E)
    # edge_index[0] 是源节点列表，其长度就是边数
    num_edges=edge_index[0].shape[0]
    # 创建一个全为 1 的数组，长度等于边数
    # 这表示每条边的初始权重都是 1 (无权图)
    data=np.array([1]*num_edges)
    # 使用 scipy.sparse.coo_matrix 构建稀疏矩阵 (COO格式)
    # 参数结构: (data, (row, col)), shape=(N, N)
    # .tocsr(): 随后立即转换为 CSR (Compressed Sparse Row) 格式
    # CSR 格式进行矩阵乘法 (Matrix Multiplication) 效率更高
    adj=sp.coo_matrix((data, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).tocsr()

    t = time.time()# 计时开始
    # 调用 PropMatrix 函数 (在 utils.py 的其他地方定义)
    # 这一步执行了数学运算: D^{-1/2} * A * D^{-1/2}
    # 1. 计算度矩阵 D (Row Sum)
    # 2. 计算 D 的 -1/2 次方
    # 3. 执行矩阵乘法进行归一化
    adj=PropMatrix(adj)
    Propagate_matrix_time = time.time()-t # 记录归一化计算耗时
    print('propagate matrix time: ', Propagate_matrix_time)
    

    t=time.time()
    # 将 Scipy 的稀疏矩阵转换回 PyTorch 的稀疏张量 (SparseTensor)
    # 并转换为 float 类型，以便在神经网络中进行梯度计算
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    sparse_mx_time = time.time()-t
    print('sparse_mx: ', sparse_mx_time)

    # 返回最终的传播矩阵 LP (即代码中的 adj) 和 计时信息
    return adj, Propagate_matrix_time, sparse_mx_time

def data_split(labels, train_rate=0.6, val_rate=0.2, seed=12591):
    num_classes = np.max(labels)+1
    num_nodes = labels.shape[0]
    percls_trn = int(round(train_rate*num_nodes/num_classes))
    val_lb = int(round(val_rate*num_nodes))
    idx_train, idx_val, idx_test = random_splits(labels, num_classes, percls_trn, val_lb, seed)    
    return  idx_train, idx_val, idx_test

def load_dataset(LP, feat, K=4, tau = 1.0, homo_ratio=0.6, plain=False):
    #将输入的原始特征 X，扩展成包含 K+1个不同频率/尺度信息的超级特征矩阵
    ###它想在聚合邻居信息的同时，强制保留原本独特的差异信息
    num_nodes, dim=feat.shape #获取特征矩阵的形状：num_nodes为节点数，dim为特征维度
    # ------------------------------------------------------------------
    # 【核心数学原理 1：设定目标夹角】
    # 对应论文：θ = (1-h) * π / 2
    # 物理意义：
    #   h (同配率) 越高 -> cosval 越大 (接近 1) -> 夹角 θ 越小 -> 基底倾向于相似 (平滑)
    #   h 越低 (异配)   -> cosval 越小 (接近 0) -> 夹角 θ 越大 -> 基底倾向于正交 (锐化/差异)
    # ------------------------------------------------------------------
    #这就像是给模型设定一个“预期的差异程度”
    cosval = math.cos(math.pi*(1.0-homo_ratio)/2.0) #根据同质性比例计算余弦值。这个值用于后续的Chebyshev多项式相关计算
    print('cosval: ', cosval) 


    if not plain: # 开启 UniFilter 核心逻辑 (Adaptive Basis)
        print('Adaptive Basis')
        t1 = time.time()
        # --- 初始化 ---
        # 对输入特征进行列归一化 (Column Normalization)，对应论文中的 u_0
        norm = torch.norm(feat, dim=0) #计算特征矩阵各列的L2范数（沿第0维，即对每个特征维度求范数）
        norm = torch.clamp(norm, 1e-8)  #将范数限制在最小值1e-8，避免除以零
        last = feat/norm #对特征矩阵进行L2正规化，得到第一个基向量last。列归一化
        second = torch.zeros_like(last) # 创建一个形状相同、全为0的张量。second 代表上上轮的基底 u_{k-2} (用于三项递推)
        basis_sum = torch.zeros_like(last) # 记录基底的累加和 (s_k)，用于辅助计算旋转系数
        # --- 双流初始状态 ---
        HM = torch.zeros_like(last) # HM: Homophily Matrix (平滑流/低频流)
        HM += feat  #将原始特征赋值给 HM
        basis_sum +=  last  #更新 basis_sum
        features = [feat] # 最终的特征列表，先放入原始特征 (第 0 跳)
        # --- 核心循环：生成第 1 到 K 跳的基底 ---
        for k in range(1, K+1): #循环K次，生成K阶特征
            # 1. 图传播 (Graph Propagation)
            #torch.spmm() 函数作用：稀疏矩阵乘法 物理意义V_1 是聚合了每个细胞1跳邻居的特征
            V_k = torch.spmm(LP, last)  ## V_k = P * u_{k-1}。这是最朴素的图卷积一步
            # 2. 更新平滑流 (HM)
            # HM 只是不断地乘 LP，没有任何减法。这意味着 HM 随着 k 增加会越来越平滑 (过平滑)。
            # 它的作用是提供由 tau 控制的"保底"去噪能力。
            HM = torch.spmm(LP, HM)    #更新HM：HM乘以LP，用于后续的特征生成  #这就是最普通的 GCN 传播。它就像给图像做高斯模糊;保留数据的低频信息（比如组织的大致区域、大块的细胞类型）。这能起到很好的去噪效果
            # ------------------------------------------------------------------
            # 【核心数学原理 2：Gram-Schmidt 正交化 (锐化流)】
            # 目的：从传播后的特征 V_k 中，剔除掉之前已经包含的信息。
            # 这就像剥洋葱，强制保留"新的"、"差异化"的信息 (高频/边缘信息)。
            # ------------------------------------------------------------------
            project_1 = torch.einsum('nd,nd->d', V_k, last) ## 投影到 u_{k-1}  计算V_k与last的点积（逐列计算），用于Gram-Schmidt正交化的第一个投影系数
            project_2 = torch.einsum('nd,nd->d', V_k, second) # 投影到 u_{k-2}  计算V_k与second的点积（逐列计算），用于Gram-Schmidt正交化的第二个投影系数
            V_k -= (project_1 * last + project_2 * second) # 减去投影分量 -> 得到纯净的差异向量
            # 归一化正交向量
            norm = torch.norm(V_k,dim = 0)
            norm = torch.clamp(norm, 1e-8)
            V_k /= norm #对V_k进行L2正规化

            # ------------------------------------------------------------------
            # 【核心数学原理 3：强制旋转 (Forced Rotation)】
            # 目的：不仅要正交，还要让新基底 u_k 与旧基底保持特定的角度 cosval。
            # 这是一个几何约束求解过程，对应论文公式 (4)。
            # ------------------------------------------------------------------
            H_k = basis_sum / k# 计算当前的平均方向 (辅助变量)
            Tf = torch.sqrt(torch.square(torch.einsum('nd,nd->d', H_k, features[-1])/cosval) - ((k-1)*cosval+1)/k)
            torch.nan_to_num_(Tf, nan=0.0)# 数值稳定性处理
            # 构造异配基底 u_k (代码变量名为 H_k)
            # 它是"平均方向"和"正交差异方向 V_k"的线性组合
            H_k += torch.mul(Tf, V_k)
            # 再次归一化
            norm = torch.norm(H_k,dim = 0)
            norm=torch.clamp(norm, 1e-8)
            H_k /= norm
            # 归一化平滑流 HM (为了尺度一致)
            norm = torch.norm(HM, dim = 0)
            norm = torch.clamp(norm, 1e-8)
            # ------------------------------------------------------------------
            # 【核心数学原理 4：双流融合】
            # 对应论文公式 (3): z = τ * P^k x + (1-τ) * u_k
            # 融合"平滑流 HM"和"锐化流 H_k"
            # ------------------------------------------------------------------
            features.append(HM * tau + H_k * (1.0 - tau))
            # 更新变量，准备下一轮迭代
            basis_sum += H_k
            second = last 
            last = V_k      #注意：下一轮的输入是正交化后的 V_k，而不是混合后的特征
        # --- 结束循环 ---
        features_time = time.time()-t1
        print('feat diffusion time plus: ', features_time)
        del last, second, LP
        gc.collect()
        # 将所有 K+1 个特征矩阵在特征维度拼接
        # 形状变化: [N, D] -> [N, (K+1)*D]
        features = torch.cat(features, 1)   
        print(features.shape)        
    else:
        # --- 普通模式 (Non-orthogonalization) ---
        # 这就是标准的 GCN/SGC 传播：单纯地乘矩阵，不减去任何东西。
        # 结果会导致严重的过平滑。
        print('Non-orthogonalization') 
        t1 = time.time()            
        features=[feat]
        basis=feat    
        for i in range(1,K+1):
            basis=torch.spmm(LP, basis)   
            features.append(basis)
        features_time = time.time()-t1
        print('feat diffusion time: ', features_time)      
        del basis, LP
        gc.collect()
        features = torch.cat(features,1)
        print(features.shape)     
    return features, dim

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def muticlass_f1(output, labels):
    preds = output.max(1)[1]  
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro

def mutilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")
