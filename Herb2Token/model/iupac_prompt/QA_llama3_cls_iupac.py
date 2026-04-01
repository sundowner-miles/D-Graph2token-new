import logging
import os
import math  # [阶段 3 新增]: 用于计算余弦退火

import torch
import torch.nn as nn
from math import sqrt

from torch.nn import L1Loss, CrossEntropyLoss
from transformers import AutoTokenizer, LlamaModel, BitsAndBytesConfig
from pathlib import Path

# 导入 GNN 模型与我们刚才重构好的多示例编码器
from model.molecule_gnn.molecule_gnn_model import GNN, GNN_graphpred
from model.herb_multi_instance_encoder import HerbMultiInstanceEncoder

local_rank = int(os.environ.get('LOCAL_RANK', '0'))
device_map = {'': local_rank}

current_file = Path(__file__).resolve()  
project_root = current_file.parents[2]  

def init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio):
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
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    molecule_model.load_state_dict(state_dict)
    ln_graph = nn.LayerNorm(4096)
    return molecule_model, ln_graph

# ==========================================================
# [阶段 3 新增]: 隐空间物理重构多层感知机 (Reconstruction MLP)
# ==========================================================
class ReconstructionMLP(nn.Module):
    """
    将 LLM 的均值池化隐向量重构回图物理特征空间
    """
    def __init__(self, llm_dim, graph_dim):
        super(ReconstructionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, graph_dim)
        )

    def forward(self, z):
        return self.net(z)

# 阶段 2: 启发式匹配融合模块
class HeuristicMatchingFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HeuristicMatchingFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, h_A, h_B):
        h_mul = h_A * h_B
        h_diff = torch.abs(h_A - h_B)
        h_concat = torch.cat([h_A, h_B, h_mul, h_diff], dim=-1)
        return self.mlp(h_concat)

class ReprogrammingLayer(nn.Module):
    def __init__(self, attention_dim, graph_dim, n_heads, d_keys=None, llm_dim=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (attention_dim // n_heads)
        self.query_projection = nn.Linear(graph_dim, d_keys * n_heads)
        self.key_projection = nn.Linear(llm_dim, d_keys * n_heads)
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
        self.graph_encoder, self.ln_graph = init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        self.herb_encoder = HerbMultiInstanceEncoder(self.graph_encoder, hidden_dim=gin_hidden_dim)
        self.graph_dim = 768  
        self.fusion_layer = HeuristicMatchingFusion(input_dim=gin_hidden_dim, output_dim=self.graph_dim)

        if not self.tune_gnn:
            self.graph_encoder.freeze_bottom_layers(tune_layers=1)
            logging.info("GNN bottom layers are frozen.")

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.llm_tokenizer.pad_token_id 
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|herb_interaction|>']})  
        self.llm_tokenizer.interaction_token_id = self.llm_tokenizer("<|herb_interaction|>", add_special_tokens=False).input_ids[0]
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|reg_g|>']}) 
        self.llm_tokenizer.reg_token_id = self.llm_tokenizer("<|reg_g|>", add_special_tokens=False).input_ids[0]

        attn_implementation = "flash_attention_2" if (args and getattr(args, 'enable_flash', False)) else "eager"
        
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        _llm_model = LlamaModel.from_pretrained(
            llm_model, 
            torch_dtype=torch.float16, 
            attn_implementation=attn_implementation,
            quantization_config=bnb_config, 
            device_map="auto"
        )
        _llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        for param in _llm_model.parameters():
            param.requires_grad = False
        _llm_model.gradient_checkpointing_enable()

        self.llm_model_holder = [_llm_model]

        self.new_add_token = 3 
        self.num_tokens = 100
        self.attention_dim = 512
        self.n_head = 8
        self.key = self.attention_dim // self.n_head
        self.llm_dim = 4096
        self.output_hidden_dim = 256

        self.word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.attention_dim, self.graph_dim, self.n_head, self.key, self.llm_dim)
        self.score = nn.Linear(self.llm_dim, 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # ==========================================================
        # [阶段 3 新增]: 初始化重构网络与退火超参数
        # ==========================================================
        self.recon_mlp = ReconstructionMLP(llm_dim=self.llm_dim, graph_dim=self.graph_dim)
        self.lambda_min = 0.01
        self.lambda_max = 1.0

        for param in self.mapping_layer.parameters():
            param.requires_grad = False

    @property
    def llm_model(self):
        return self.llm_model_holder[0]

    # [阶段 3 新增]: forward 增加 current_epoch 和 max_epochs 参数
    def forward(self, batch, current_epoch=0, max_epochs=50):
        batched_A, batched_B, instruction_tokens, label_values = batch
        
        # (已注释掉暴力阶段，防止截断 <|herb_interaction|>)
        # MAX_SEQ_LEN = 512  
        # if instruction_tokens.input_ids.shape[1] > MAX_SEQ_LEN:
        #     instruction_tokens.input_ids = instruction_tokens.input_ids[:, :MAX_SEQ_LEN]
        #     ...

        # [阶段 3 修改]: 解包步骤 1 返回的 6 个参数
        H_A_given_B, H_B_given_A, h_A, h_B, alpha_A, alpha_B = self.herb_encoder(batched_A, batched_B)

        H_interaction = self.fusion_layer(H_A_given_B, H_B_given_A)
        h_graph = H_interaction.unsqueeze(1)

        word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)

        graph_inputs_llm = self.reprogramming_layer(h_graph, source_embeddings, source_embeddings)
        graph_inputs_llm = self.ln_graph(graph_inputs_llm)
        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
        word_embeddings_avg = word_embeddings.mean(dim=0, keepdim=True)

        instruction_embeds[instruction_tokens.is_interaction_token] = graph_inputs_llm.flatten(0, 1).to(instruction_embeds.dtype)
        
        if hasattr(instruction_tokens, 'is_reg_token'):
            instruction_embeds[instruction_tokens.is_reg_token] = word_embeddings_avg.repeat(graph_inputs_llm.size()[0], 1).to(instruction_embeds.dtype)

        outputs = self.llm_model(
            inputs_embeds=instruction_embeds,
            attention_mask=instruction_tokens.attention_mask,
            return_dict=True,
        )

        output = outputs.last_hidden_state  
        
        # ==========================================================
        # [阶段 3 核心]: 隐式思维链 (CoT) 提取与物理约束退火
        # ==========================================================
        # 1. 均值池化提取代表大模型推理全过程的浓缩隐向量 Z_final
        mask = (instruction_tokens.attention_mask).float().unsqueeze(-1)
        z_final = (output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8) # [B, 4096]

        # 2. 将隐向量重构回图特征空间
        H_recon = self.recon_mlp(z_final) # [B, 768]
        
        # 3. 宏观消融基准 (Macro-Reconstruction: 重构整体特征)
        loss_recon_macro = nn.MSELoss()(H_recon, H_interaction.detach())
        
        # 4. 微观消融基准 (Micro-Reconstruction: 重构 Top-1 核心成分)
        batch_size = H_interaction.size(0)
        h_A_top1_list, h_B_top1_list = [], []
        for i in range(batch_size):
            mask_A = (batched_A.herb_batch == i)
            top1_idx_A = torch.argmax(alpha_A[mask_A])
            h_A_top1_list.append(h_A[mask_A][top1_idx_A])
            
            mask_B = (batched_B.herb_batch == i)
            top1_idx_B = torch.argmax(alpha_B[mask_B])
            h_B_top1_list.append(h_B[mask_B][top1_idx_B])
            
        H_interaction_micro = self.fusion_layer(torch.stack(h_A_top1_list), torch.stack(h_B_top1_list))
        loss_recon_micro = nn.MSELoss()(H_recon, H_interaction_micro.detach())
        
        # 此处使用宏观重构作为默认主约束，您可以根据消融实验需要将其切换为 loss_recon_micro
        loss_recon = loss_recon_macro 

        # 5. 硬到软物理锚点退火 (Cosine Annealing)
        e_ratio = min(current_epoch / max(max_epochs, 1), 1.0)
        lambda_e = self.lambda_min + 0.5 * (self.lambda_max - self.lambda_min) * (1 + math.cos(e_ratio * math.pi))
        # ==========================================================

        sequence_lengths = torch.eq(instruction_tokens.input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % instruction_tokens.input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(output.device)

        output_token_embedding = output[torch.arange(output.shape[0], device=output.device), sequence_lengths]
        output_token_embedding = output_token_embedding.to(self.score.weight.device)
        label_values = label_values.to(self.score.weight.device)  

        logits = self.score(output_token_embedding)
        loss_cls = CrossEntropyLoss()(logits.view(-1, 2), label_values.view(-1))
        
        # 6. 多任务联合优化
        loss_total = loss_cls + lambda_e * loss_recon
        
        # 将子损失分离返回，方便在 Trainer 中进行日志记录
        return {"loss": loss_total, "loss_cls": loss_cls, "loss_recon": loss_recon}

    @torch.no_grad()
    def generate(self, samples, num_beams=None):
        batched_A = samples['batched_A']
        batched_B = samples['batched_B']
        instruction_tokens = samples['instruction_tokens']
        
        # [阶段 3 修改]: 同样在推理时解包 6 个参数，防止报错
        H_A_given_B, H_B_given_A, h_A, h_B, alpha_A, alpha_B = self.herb_encoder(batched_A, batched_B)
        
        H_interaction = self.fusion_layer(H_A_given_B, H_B_given_A)
        h_graph = H_interaction.unsqueeze(1)

        word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)

        graph_inputs_llm = self.reprogramming_layer(h_graph, source_embeddings, source_embeddings)
        graph_inputs_llm = self.ln_graph(graph_inputs_llm)
        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
        word_embeddings_avg = word_embeddings.mean(dim=0, keepdim=True)

        instruction_embeds[instruction_tokens.is_interaction_token] = graph_inputs_llm.flatten(0, 1).to(instruction_embeds.dtype)
        
        if hasattr(instruction_tokens, 'is_reg_token'):
            instruction_embeds[instruction_tokens.is_reg_token] = word_embeddings_avg.repeat(graph_inputs_llm.size()[0], 1).to(instruction_embeds.dtype)

        outputs = self.llm_model(
            inputs_embeds=instruction_embeds,
            attention_mask=instruction_tokens.attention_mask,
            return_dict=True,
        )

        output = outputs.last_hidden_state  
        sequence_lengths = torch.eq(instruction_tokens.input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % instruction_tokens.input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(output.device)

        output_token_embedding = output[torch.arange(output.shape[0], device=output.device), sequence_lengths]
        output_token_embedding = output_token_embedding.to(self.score.weight.device)
        output_token_embedding = self.score(output_token_embedding)
        logits = self.softmax(output_token_embedding)

        return logits.view(-1, 2)