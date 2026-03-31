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

def init_mol2latent(gin_hidden_dim):
    mol2latent = nn.Linear(gin_hidden_dim, 768)
    input_model_path = project_root / 'GNN_pretrained/output_model/mol2latent_model_final.pth'
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)
    return mol2latent

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

        # 2. 引入重构后的多示例编码器，代理图特征提取
        self.herb_encoder = HerbMultiInstanceEncoder(self.graph_encoder, hidden_dim=gin_hidden_dim)
        
        # 3. 初始化对齐潜空间映射
        self.mol2latent = init_mol2latent(gin_hidden_dim)

        # 4. 执行底层冻结逻辑
        if not self.tune_gnn:
            # 仅微调 GNN 的最顶层
            self.graph_encoder.freeze_bottom_layers(tune_layers=1)
            # 冻结 mol2latent
            for param in self.mol2latent.parameters():
                param.requires_grad = False
            logging.info("GNN bottom layers and mol2latent are frozen.")

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.llm_tokenizer.pad_token_id 
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|mol_g|>']})  
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer("<|mol_g|>", add_special_tokens=False).input_ids[0]
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
        self.graph_dim = 768  
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
            if hasattr(instruction_tokens, 'is_mol_token'):
                instruction_tokens.is_mol_token = instruction_tokens.is_mol_token[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_reg_token'):
                instruction_tokens.is_reg_token = instruction_tokens.is_reg_token[:, :MAX_SEQ_LEN]

        # 核心：使用多示例图编码器提取相互激发的宏观表征
        H_A_given_B, H_B_given_A = self.herb_encoder(batched_A, batched_B)

        # 映射到 768 潜空间以适配 ReprogrammingLayer
        H_A_latent = self.mol2latent(H_A_given_B)
        H_B_latent = self.mol2latent(H_B_given_A)

        # 拼接以满足 [Batch, 2, 768] 的双图输入
        h_graph = torch.cat([H_A_latent.unsqueeze(1), H_B_latent.unsqueeze(1)], dim=1)

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
            if hasattr(instruction_tokens, 'is_mol_token'):
                instruction_tokens.is_mol_token = instruction_tokens.is_mol_token[:, :MAX_SEQ_LEN]
            if hasattr(instruction_tokens, 'is_reg_token'):
                instruction_tokens.is_reg_token = instruction_tokens.is_reg_token[:, :MAX_SEQ_LEN]

        H_A_given_B, H_B_given_A = self.herb_encoder(batched_A, batched_B)

        H_A_latent = self.mol2latent(H_A_given_B)
        H_B_latent = self.mol2latent(H_B_given_A)

        h_graph = torch.cat([H_A_latent.unsqueeze(1), H_B_latent.unsqueeze(1)], dim=1)

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

        output = outputs.last_hidden_state  
        sequence_lengths = torch.eq(instruction_tokens.input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % instruction_tokens.input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(output.device)

        output_token_embedding = output[torch.arange(output.shape[0], device=output.device), sequence_lengths]
        output_token_embedding = output_token_embedding.to(self.score.weight.device)
        output_token_embedding = self.score(output_token_embedding)
        logits = self.softmax(output_token_embedding)

        return logits.view(-1, 2)