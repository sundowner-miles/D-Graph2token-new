import logging
import os

import torch
import torch.nn as nn
from math import sqrt

from torch.nn import L1Loss, CrossEntropyLoss
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
from transformers import AutoTokenizer

from pathlib import Path

# from model.gin_model import GNN
# from model.modeling_llama3 import LlamaForSequenceClassification
from transformers import LlamaModel, BitsAndBytesConfig
from model.molecule_gnn.molecule_gnn_model import GNN, GNN_graphpred


local_rank = int(os.environ.get('LOCAL_RANK', '0'))
device_map = {'': local_rank}

current_file = Path(__file__).resolve()  # 绝对路径的 Path 对象
project_root = current_file.parents[2]  # 上两级目录（根据项目结构调整）

def init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio):
    # graph_encoder = GNN(
    #     num_layer=gin_num_layers,
    #     emb_dim=gin_hidden_dim,
    #     gnn_type='gin',
    #     drop_ratio=gin_drop_ratio,
    #     JK='last',
    #     # JK='sum',
    # )

    molecule_node_model = GNN(
        num_layer=gin_num_layers,
        emb_dim=gin_hidden_dim,
        JK='last',
        drop_ratio=gin_drop_ratio,
        gnn_type='gin'
    )
    molecule_model = GNN_graphpred(
        num_layer=gin_num_layers,
        emb_dim=gin_hidden_dim,
        JK='last',
        graph_pooling='mean',
        num_tasks=1,
        molecule_node_model=molecule_node_model
    )
    input_model_path = project_root / 'GNN_pretrained/output_model/molecule_model_final.pth'
    # input_model_path = '/home/mingqi/zebron/LLM/molecule_model_final.pth'
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    molecule_model.load_state_dict(state_dict)

    # ckpt = torch.load('pretrained_model/graphcl_80.pth', map_location=torch.device('cpu'))
    # missing_keys, unexpected_keys = graph_encoder.load_state_dict(ckpt, strict=False)
    # if len(missing_keys) or len(unexpected_keys):
    #     print(missing_keys)
    #     print(unexpected_keys)

    # ln_graph = LayerNorm(graph_encoder.num_features)
    ln_graph = nn.LayerNorm(4096)

    return molecule_model, ln_graph


def init_mol2latent(gin_hidden_dim):
    mol2latent = nn.Linear(gin_hidden_dim, 768)
    input_model_path = project_root / 'GNN_pretrained/output_model/mol2latent_model_final.pth'
    # input_model_path = '/home/mingqi/zebron/LLM/mol2latent_model_final.pth'
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)
    return mol2latent


class HerbCompatibilityAttention(nn.Module):
    """
    交互式动态注意力模块 (Herb Compatibility Module)
    用于计算草药 A 与草药 B 之间的“君臣佐使”配伍激活权重。
    """
    def __init__(self, d_graph):
        super(HerbCompatibilityAttention, self).__init__()
        # 对应公式中的 W_attn，用于计算相关性得分
        self.W_attn = nn.Linear(d_graph, d_graph, bias=False)

    def forward(self, h_mol_A, herb_batch_A, h_mol_B, herb_batch_B):
        """
        参数:
            h_mol_A: 草药 A 集合中所有分子的特征, shape [N_total_mols_A, d_graph]
            herb_batch_A: 分子到草药 Batch 的映射索引, shape [N_total_mols_A]
            h_mol_B: 草药 B 集合中所有分子的特征, shape [N_total_mols_B, d_graph]
            herb_batch_B: 分子到草药 Batch 的映射索引, shape [N_total_mols_B]
        返回:
            H_A_given_B: 感知了 B 之后 A 的宏观特征, shape [Batch_Size, d_graph]
            H_B_given_A: 感知了 A 之后 B 的宏观特征, shape [Batch_Size, d_graph]
        """
        # 1. 计算配伍环境的全局均值 Query: H_{B_mean} 和 H_{A_mean}
        # shape: [Batch_Size, d_graph]
        H_B_mean = global_mean_pool(h_mol_B, herb_batch_B)
        H_A_mean = global_mean_pool(h_mol_A, herb_batch_A)

        # 2. 将全局 Query 广播展开，使其与分子数量对齐
        # H_B_mean_expanded shape: [N_total_mols_A, d_graph]
        H_B_mean_expanded = H_B_mean[herb_batch_A]
        H_A_mean_expanded = H_A_mean[herb_batch_B]

        # ---------------- 草药 A 在 B 环境下的动态激活 ----------------
        # e_{A,i} = (h_{A,i})^T W_{attn} (H_{B_mean})
        h_mol_A_proj = self.W_attn(h_mol_A)  # [N_total_mols_A, d_graph]
        # 逐元素相乘并在特征维度求和，等价于点积
        e_A = torch.sum(h_mol_A_proj * H_B_mean_expanded, dim=-1)  # [N_total_mols_A]

        # 在每个草药包 (Bag) 内部进行 Softmax，计算动态权重 \alpha_{A,i}
        alpha_A = pyg_softmax(e_A, index=herb_batch_A)  # [N_total_mols_A]

        # 根据权重对分子特征进行加权求和，得到宏观表征 H_{A|B}
        # alpha_A.unsqueeze(-1) shape 变为 [N_total_mols_A, 1] 以便广播
        H_A_given_B = global_add_pool(h_mol_A * alpha_A.unsqueeze(-1), herb_batch_A) # [Batch_Size, d_graph]

        # ---------------- 草药 B 在 A 环境下的动态激活 (对称结构) ----------------
        h_mol_B_proj = self.W_attn(h_mol_B)
        e_B = torch.sum(h_mol_B_proj * H_A_mean_expanded, dim=-1)
        alpha_B = pyg_softmax(e_B, index=herb_batch_B)
        H_B_given_A = global_add_pool(h_mol_B * alpha_B.unsqueeze(-1), herb_batch_B)

        return H_A_given_B, H_B_given_A


class ReprogrammingLayer(nn.Module):
    def __init__(self, attention_dim, graph_dim, n_heads, d_keys=None, llm_dim=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (attention_dim // n_heads)

        self.query_projection = nn.Linear(graph_dim, d_keys * n_heads)
        # self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.key_projection = nn.Linear(llm_dim, d_keys * n_heads)
        # self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(llm_dim, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, llm_dim)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Align2llama(nn.Module):

    def __init__(
            self,
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio,
            tune_gnn=False,
            lora_tuning=False,
            llm_model="decapoda-research/llama-7b-hf",
            args=None,
    ):
        super().__init__()
        self.graph_encoder, self.ln_graph = init_graph_encoder(
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio
        )
        self.tune_gnn = tune_gnn

        self.mol2latent = init_mol2latent(gin_hidden_dim)
        
        # [新增] 注册草药配伍动态注意力模块 (输入维度为 768，即 mol2latent 的输出)
        self.herb_compatibility = HerbCompatibilityAttention(d_graph=768)

        if not self.tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.mol2latent.named_parameters():
                param.requires_grad = False

            logging.info("freeze graph encoder")

        # initialize opt model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model,
                                                           use_fast=False,
                                                           padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.llm_tokenizer.pad_token_id 

        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|mol_g|>']})  
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer("<|mol_g|>", add_special_tokens=False).input_ids[0]
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|reg_g|>']}) 
        self.llm_tokenizer.reg_token_id = self.llm_tokenizer("<|reg_g|>", add_special_tokens=False).input_ids[0]

        attn_implementation = "flash_attention_2" if (args and getattr(args, 'enable_flash', False)) else "eager"
        
        print("=== 调试：加载的模型路径 ===")
        print(llm_model)  
        import json
        with open(f"{llm_model}/config.json", "r") as f:
            config_dict = json.load(f)
        print("=== 调试：config.json 中的 rope_theta ===")
        print(config_dict.get("rope_theta", "字段缺失！"))

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # ===================== 🌟 核心修改 1：用局部变量加载模型，并开启 device_map =====================
        _llm_model = LlamaModel.from_pretrained(
            llm_model, 
            torch_dtype=torch.float16, 
            attn_implementation=attn_implementation,
            quantization_config=bnb_config, 
            device_map="auto"               # 开启自动多卡切片
        )
        _llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in _llm_model.named_parameters():
            param.requires_grad = False
            
        _llm_model.gradient_checkpointing_enable()

        # 🌟🌟🌟 瞒天过海大法：将模型装进原生 list，避开 PyTorch Lightning 的强制设备转移！
        self.llm_model_holder = [_llm_model]
        # ==============================================================================================

        self.new_add_token = 3
        self.num_tokens = 100
        self.graph_dim = 768  
        self.attention_dim = 512
        self.n_head = 8
        self.key = self.attention_dim // self.n_head
        self.llm_dim = 4096
        self.output_hidden_dim = 256

        # 这里通过属性调用 self.llm_model，依然能正常获取 word_embeddings
        self.word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.attention_dim, self.graph_dim, self.n_head, self.key,
                                                      self.llm_dim)

        self.score = nn.Linear(self.llm_dim, 2, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        for name, param in self.mapping_layer.named_parameters():
            param.requires_grad = False

    # ===================== 🌟 核心修改 2：增加属性代理 =====================
    @property
    def llm_model(self):
        """
        属性代理：拦截所有的 self.llm_model 调用，使其指向 list 中的模型。
        这样既不影响 forward 和 generate 的代码结构，又成功向 Lightning 隐藏了模型！
        """
        return self.llm_model_holder[0]
    # =====================================================================

    def forward(self, batch):
        batched_A, batched_B, instruction_tokens, label_values = batch
        
        # ===================== 1. 暴力截断防线 =====================
        MAX_SEQ_LEN = 512  
        if instruction_tokens.input_ids.shape[1] > MAX_SEQ_LEN:
            instruction_tokens.input_ids = instruction_tokens.input_ids[:, :MAX_SEQ_LEN]
            instruction_tokens.attention_mask = instruction_tokens.attention_mask[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_mol_token'):
                instruction_tokens.is_mol_token = instruction_tokens.is_mol_token[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_reg_token'):
                instruction_tokens.is_reg_token = instruction_tokens.is_reg_token[:, :MAX_SEQ_LEN]
        # ==============================================================================

        # [修改] 分别编码 batched_A 和 batched_B 中的所有分子，获取微观特征
        h_graph_feature_A, _ = self.graph_encoder(batched_A)
        h_mol_A = self.mol2latent(h_graph_feature_A)

        h_graph_feature_B, _ = self.graph_encoder(batched_B)
        h_mol_B = self.mol2latent(h_graph_feature_B)

        # [新增] 君臣佐使：计算交互式动态配伍注意力
        # 防御性将 herb_batch 转移到对应特征设备的内存上
        herb_batch_A = batched_A.herb_batch.to(h_mol_A.device)
        herb_batch_B = batched_B.herb_batch.to(h_mol_B.device)
        
        H_A_given_B, H_B_given_A = self.herb_compatibility(
            h_mol_A=h_mol_A, 
            herb_batch_A=herb_batch_A, 
            h_mol_B=h_mol_B, 
            herb_batch_B=herb_batch_B
        )

        # [修改] 使用配伍后的宏观特征在序列维度(dim=1)拼接，满足 [B, 2, 768] 的双图输入
        h_graph = torch.cat([H_A_given_B.unsqueeze(1), H_B_given_A.unsqueeze(1)], dim=1)

        word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)

        # 因为 h_graph 已经是三维 [B, 2, 768]，直接传入，无需 unsqueeze(1)
        graph_inputs_llm = self.reprogramming_layer(h_graph, source_embeddings, source_embeddings)
        graph_inputs_llm = self.ln_graph(graph_inputs_llm)
        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)

        word_embeddings_avg = word_embeddings.mean(dim=0, keepdim=True)

        # 这里 graph_inputs_llm.flatten(0, 1) 的形状变为 [B*2, llm_dim]，正好对应每条序列中的 2 个 mol_token
        instruction_embeds[instruction_tokens.is_mol_token] = graph_inputs_llm.flatten(0, 1).to(instruction_embeds.dtype)
        instruction_embeds[instruction_tokens.is_reg_token] = word_embeddings_avg.repeat(graph_inputs_llm.size()[0], 1).to(instruction_embeds.dtype)

        outputs = self.llm_model(
            inputs_embeds=instruction_embeds,
            attention_mask=instruction_tokens.attention_mask,
            return_dict=True,
        )

        # ===================== 2. 基础模型特征获取 =====================
        output = outputs.last_hidden_state  

        sequence_lengths = torch.eq(instruction_tokens.input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % instruction_tokens.input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(output.device)

        output_token_embedding = output[torch.arange(output.shape[0], device=output.device), sequence_lengths]

        # ===================== 3. 跨卡张量对齐 =====================
        output_token_embedding = output_token_embedding.to(self.score.weight.device)
        label_values = label_values.to(self.score.weight.device)  
        # ==============================================================================

        logits = self.score(output_token_embedding)
        loss_fct = CrossEntropyLoss()
        
        loss_cls = loss_fct(logits.view(-1, 2), label_values.view(-1))
        return {"loss": loss_cls}


    @torch.no_grad()
    def generate(
            self,
            samples,
            num_beams=None,
    ):
        batched_A = samples['batched_A']
        batched_B = samples['batched_B']
        instruction_tokens = samples['instruction_tokens']
        
        # ===================== 1. 暴力截断防线 =====================
        MAX_SEQ_LEN = 512
        if instruction_tokens.input_ids.shape[1] > MAX_SEQ_LEN:
            instruction_tokens.input_ids = instruction_tokens.input_ids[:, :MAX_SEQ_LEN]
            instruction_tokens.attention_mask = instruction_tokens.attention_mask[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_mol_token'):
                instruction_tokens.is_mol_token = instruction_tokens.is_mol_token[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_reg_token'):
                instruction_tokens.is_reg_token = instruction_tokens.is_reg_token[:, :MAX_SEQ_LEN]
        # ==============================================================================

        # [修改] 双路图微观特征编码
        h_graph_feature_A, _ = self.graph_encoder(batched_A)
        h_mol_A = self.mol2latent(h_graph_feature_A)

        h_graph_feature_B, _ = self.graph_encoder(batched_B)
        h_mol_B = self.mol2latent(h_graph_feature_B)

        # [新增] 君臣佐使：计算交互式动态配伍注意力
        herb_batch_A = batched_A.herb_batch.to(h_mol_A.device)
        herb_batch_B = batched_B.herb_batch.to(h_mol_B.device)

        H_A_given_B, H_B_given_A = self.herb_compatibility(
            h_mol_A=h_mol_A, 
            herb_batch_A=herb_batch_A, 
            h_mol_B=h_mol_B, 
            herb_batch_B=herb_batch_B
        )

        # [修改] 拼接宏观特征
        h_graph = torch.cat([H_A_given_B.unsqueeze(1), H_B_given_A.unsqueeze(1)], dim=1)

        word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)

        graph_inputs_llm = self.reprogramming_layer(h_graph, source_embeddings, source_embeddings)
        graph_inputs_llm = self.ln_graph(graph_inputs_llm)
        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)

        word_embeddings_avg = word_embeddings.mean(dim=0, keepdim=True)

        instruction_embeds[instruction_tokens.is_mol_token] = graph_inputs_llm.flatten(0, 1).to(instruction_embeds.dtype)
        instruction_embeds[instruction_tokens.is_reg_token] = word_embeddings_avg.repeat(graph_inputs_llm.size()[0], 1).to(instruction_embeds.dtype)

        outputs = self.llm_model(
            inputs_embeds=instruction_embeds,
            attention_mask=instruction_tokens.attention_mask,
            return_dict=True,
        )

        # ===================== 2. 基础模型特征获取 =====================
        output = outputs.last_hidden_state  

        sequence_lengths = torch.eq(instruction_tokens.input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % instruction_tokens.input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(output.device)

        output_token_embedding = output[torch.arange(output.shape[0], device=output.device), sequence_lengths]

        # ===================== 3. 跨卡张量对齐 =====================
        output_token_embedding = output_token_embedding.to(self.score.weight.device)
        # ==============================================================================

        output_token_embedding = self.score(output_token_embedding)
        logits = self.softmax(output_token_embedding)

        return logits.view(-1, 2)