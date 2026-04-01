import logging
import os

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

# ===== 新增：启发式匹配融合模块 (Heuristic Matching Fusion) =====
class HeuristicMatchingFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HeuristicMatchingFusion, self).__init__()
        # 对应公式: MLP([H_{A|B} || H_{B|A} || (H_{A|B} * H_{B|A}) || |H_{A|B} - H_{B|A}|])
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
        # 1. 初始化基础 GNN
        self.graph_encoder, self.ln_graph = init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn

        # 2. 多示例编码器，提取交互环境下的宏观表征
        self.herb_encoder = HerbMultiInstanceEncoder(self.graph_encoder, hidden_dim=gin_hidden_dim)
        
        # 3. 启发式匹配融合网络 (Heuristic Matching Fusion)
        self.graph_dim = 768  
        self.fusion_layer = HeuristicMatchingFusion(input_dim=gin_hidden_dim, output_dim=self.graph_dim)

        if not self.tune_gnn:
            self.graph_encoder.freeze_bottom_layers(tune_layers=1)
            logging.info("GNN bottom layers are frozen.")
            # 注意: self.fusion_layer 是全新初始化的，必须参与训练，无需 freeze

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.llm_tokenizer.pad_token_id 
        
        # 修改点：将原先的 <|mol_g|> 替换为单 token <|herb_interaction|>
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

        self.new_add_token = 3 # pad, herb_interaction, reg_g
        self.num_tokens = 100
        self.attention_dim = 512
        self.n_head = 8
        self.key = self.attention_dim // self.n_head
        self.llm_dim = 4096
        self.output_hidden_dim = 256

        self.word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # ReprogrammingLayer 等价于计算：Q=HW_Q, K=M'W_K, V=M'W_V, Softmax(QK^T/sqrt(d))V
        self.reprogramming_layer = ReprogrammingLayer(self.attention_dim, self.graph_dim, self.n_head, self.key, self.llm_dim)
        self.score = nn.Linear(self.llm_dim, 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        for param in self.mapping_layer.parameters():
            param.requires_grad = False

    @property
    def llm_model(self):
        return self.llm_model_holder[0]

    def forward(self, batch):
        batched_A, batched_B, instruction_tokens, label_values = batch
        
        MAX_SEQ_LEN = 512  
        if instruction_tokens.input_ids.shape[1] > MAX_SEQ_LEN:
            instruction_tokens.input_ids = instruction_tokens.input_ids[:, :MAX_SEQ_LEN]
            instruction_tokens.attention_mask = instruction_tokens.attention_mask[:, :MAX_SEQ_LEN]
            # 注意：此处需跟随 dataloader 中的新变量名，假设我们已将其修改为 is_interaction_token
            if hasattr(instruction_tokens, 'is_interaction_token'):
                instruction_tokens.is_interaction_token = instruction_tokens.is_interaction_token[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_reg_token'):
                instruction_tokens.is_reg_token = instruction_tokens.is_reg_token[:, :MAX_SEQ_LEN]

        # 核心 1：多示例图编码器提取相互激发的宏观表征
        H_A_given_B, H_B_given_A = self.herb_encoder(batched_A, batched_B)

        # 核心 2：启发式匹配融合，捕捉共性与差异
        H_interaction = self.fusion_layer(H_A_given_B, H_B_given_A)

        # 扩充维度以适配 ReprogrammingLayer [Batch, Length=1, 768]
        h_graph = H_interaction.unsqueeze(1)

        word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)

        # 核心 3：词表降维检索与生成 Token (由 ReprogrammingLayer 完成交叉注意力)
        graph_inputs_llm = self.reprogramming_layer(h_graph, source_embeddings, source_embeddings)
        graph_inputs_llm = self.ln_graph(graph_inputs_llm)
        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
        word_embeddings_avg = word_embeddings.mean(dim=0, keepdim=True)

        # 注入生成的 <herb_interaction> 特征
        instruction_embeds[instruction_tokens.is_interaction_token] = graph_inputs_llm.flatten(0, 1).to(instruction_embeds.dtype)
        
        # 可选的 reg 占位操作 (原逻辑保留)
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
        label_values = label_values.to(self.score.weight.device)  

        logits = self.score(output_token_embedding)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, 2), label_values.view(-1))
        
        return {"loss": loss_cls}

    @torch.no_grad()
    def generate(self, samples, num_beams=None):
        batched_A = samples['batched_A']
        batched_B = samples['batched_B']
        instruction_tokens = samples['instruction_tokens']
        
        MAX_SEQ_LEN = 512
        if instruction_tokens.input_ids.shape[1] > MAX_SEQ_LEN:
            instruction_tokens.input_ids = instruction_tokens.input_ids[:, :MAX_SEQ_LEN]
            instruction_tokens.attention_mask = instruction_tokens.attention_mask[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_interaction_token'):
                instruction_tokens.is_interaction_token = instruction_tokens.is_interaction_token[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_reg_token'):
                instruction_tokens.is_reg_token = instruction_tokens.is_reg_token[:, :MAX_SEQ_LEN]

        H_A_given_B, H_B_given_A = self.herb_encoder(batched_A, batched_B)
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