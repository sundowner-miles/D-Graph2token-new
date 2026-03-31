import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax

class HerbMultiInstanceEncoder(nn.Module):
    def __init__(self, gnn_model, hidden_dim):
        super(HerbMultiInstanceEncoder, self).__init__()
        self.gnn = gnn_model # 原有的 GNN
        
        # 动态注意力打分网络
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 启发式匹配融合网络：将 4 种维度的特征压缩回 graph_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, batched_A, batched_B):
        # 1. 提取每个分子的特征 (需要确认 GNN 的返回值，这里假设返回的第一项是整图特征)
        h_A, _ = self.gnn(batched_A)  # shape: [total_mols_A, hidden_dim]
        h_B, _ = self.gnn(batched_B)  # shape: [total_mols_B, hidden_dim]
        
        herb_idx_A = batched_A.herb_idx
        herb_idx_B = batched_B.herb_idx

        # 2. 全局平均特征作为 Query
        H_B_mean = global_mean_pool(h_B, herb_idx_B)
        H_A_mean = global_mean_pool(h_A, herb_idx_A)

        # 3. 交叉动态注意力
        # --- 草药 A 被 B 激发 ---
        expanded_H_B = H_B_mean[herb_idx_A]
        cat_A = torch.cat([h_A, expanded_H_B], dim=-1)
        alpha_A = softmax(self.attn_net(cat_A), herb_idx_A)
        H_A_given_B = global_add_pool(alpha_A * h_A, herb_idx_A) 

        # --- 草药 B 被 A 激发 ---
        expanded_H_A = H_A_mean[herb_idx_B]
        cat_B = torch.cat([h_B, expanded_H_A], dim=-1)
        alpha_B = softmax(self.attn_net(cat_B), herb_idx_B)
        H_B_given_A = global_add_pool(alpha_B * h_B, herb_idx_B) 
        
        # 4. 启发式匹配融合
        # 拼接: [A|B, B|A, A|B*B|A, |A|B - B|A|]
        concat_feat = torch.cat([
            H_A_given_B, 
            H_B_given_A, 
            H_A_given_B * H_B_given_A, 
            torch.abs(H_A_given_B - H_B_given_A)
        ], dim=-1)
        
        H_interaction = self.fusion_mlp(concat_feat) # shape: [batch_size, hidden_dim]
        
        return H_interaction, H_A_given_B, H_B_given_A