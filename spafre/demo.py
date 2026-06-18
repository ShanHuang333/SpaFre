import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import os
from Train_SpaFre import train_Dual_SpaFre
np.random.seed(0)
import random
random.seed(0)

from sklearn.metrics.cluster import adjusted_rand_score
from utils import Cal_Spatial_Net, Transfer_pytorch_Data, mclust_R

os.environ['R_HOME'] = '/root/miniconda3/lib/R'
os.environ['R_USER'] = ' /root/miniconda3/lib/python3.10/site-packages/rpy2'

section_id = '151675'
input_dir = os.path.join('D:\data\ST\DLPFC', section_id)
adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

Ann_df = pd.read_csv(os.path.join('D:\data\ST\DLPFC', section_id, section_id+'_truth.txt'), sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

Cal_Spatial_Net(adata, rad_cutoff=150)
adata = train_Dual_SpaFre(adata,
                           hidden_dims=[512, 30],
                           n_epochs=1500,
                           K=4,
                           init_tau=0.7,
                           init_homo_ratio=0.6,
                           init_gating_weight=0.6)

sc.pp.neighbors(adata, use_rep='SpaFre')
sc.tl.umap(adata)
import rpy2.robjects as robjects
robjects.r('set.seed(0)')

adata = mclust_R(adata, used_obsm='SpaFre', num_cluster=5)
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
print('Adjusted rand index = %.2f' %ARI)