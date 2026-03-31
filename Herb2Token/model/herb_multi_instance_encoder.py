import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax

class HerbMultiInstanceEncoder(nn.Module):
    """
    多示例图编码器 (Multi-Instance Graph Encoder)
    包含预训练 GNN 特征提取与交互式动态注意力 (Context-aware Dynamic Attention)
    """
    def __init__(self, gnn_model, hidden_dim):
        super(HerbMultiInstanceEncoder, self).__init__()
        self.gnn = gnn_model  # 预训练的 GNN (例如 GIN_graphpred 的底层)
        
        # 对应理论公式中的 W_attn，用于将分子微观特征映射到可以与 Query 计算点积的空间
        self.W_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, batched_A, batched_B):
        """
        参数:
            batched_A: 包含草药 A 集合中所有分子的 PyG Batched Graph
            batched_B: 包含草药 B 集合中所有分子的 PyG Batched Graph
        """
        # 1. 提取每个分子的特征 h_{A,i} 和 h_{B,j}
        # 假设 gnn(batch) 返回 (node_representation, graph_representation) 或者类似结构
        # 这里取返回值的第一个，代表图的整体或汇聚前特征。后续要在集成时确保 GNN 输出匹配 [total_mols, hidden_dim]
        h_A, _ = self.gnn(batched_A)  
        h_B, _ = self.gnn(batched_B)  
        
        # 统一使用 herb_batch 作为草药包的索引名
        herb_batch_A = batched_A.herb_batch
        herb_batch_B = batched_B.herb_batch

        # 2. 全局平均特征作为配伍环境的 Query: H_{B_mean} 和 H_{A_mean}
        H_B_mean = global_mean_pool(h_B, herb_batch_B)
        H_A_mean = global_mean_pool(h_A, herb_batch_A)

        # 3. 交叉动态注意力计算 ("君臣佐使" 动态激活)
        
        # --- 草药 A 被 B 激发 ---
        # 广播 Query 以对齐分子数量
        expanded_H_B = H_B_mean[herb_batch_A]
        # 公式: e_{A,i} = (h_{A,i})^T W_attn (H_{B_mean})
        h_A_proj = self.W_attn(h_A)
        e_A = torch.sum(h_A_proj * expanded_H_B, dim=-1)
        # 公式: \alpha_{A,i} = Softmax(e_{A,i})
        alpha_A = softmax(e_A, index=herb_batch_A)
        # 公式: H_{A|B} = \sum \alpha_{A,i} h_{A,i}
        H_A_given_B = global_add_pool(alpha_A.unsqueeze(-1) * h_A, herb_batch_A) 

        # --- 草药 B 被 A 激发 ---
        # 广播 Query 以对齐分子数量
        expanded_H_A = H_A_mean[herb_batch_B]
        # 公式: e_{B,j} = (h_{B,j})^T W_attn (H_{A_mean})
        h_B_proj = self.W_attn(h_B)
        e_B = torch.sum(h_B_proj * expanded_H_A, dim=-1)
        # 公式: \alpha_{B,j} = Softmax(e_{B,j})
        alpha_B = softmax(e_B, index=herb_batch_B)
        # 公式: H_{B|A} = \sum \alpha_{B,j} h_{B,j}
        H_B_given_A = global_add_pool(alpha_B.unsqueeze(-1) * h_B, herb_batch_B) 
        
        # 4. 直接返回激活后的宏观特征供 LLM 作为 Token 读取
        return H_A_given_B, H_B_given_A