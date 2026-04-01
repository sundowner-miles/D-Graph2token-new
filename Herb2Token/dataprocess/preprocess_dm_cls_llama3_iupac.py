import torch
import pandas as pd
import os
import hashlib
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
from dataprocess.smiles2graph_regression import smiles2graph

# ===================== 全局添加安全白名单 =====================
try:
    from torch_geometric.data.data import Data as TGData
    torch.serialization.add_safe_globals([TGData])
    print("已将 torch_geometric.data.Data 加入 PyTorch 序列化安全白名单")
except Exception as e:
    print(f"添加安全白名单失败（低版本PyTorch无需处理）：{e}")

class TrainCollater:
    def __init__(self, tokenizer, text_max_len, interaction_token_id, reg_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.interaction_token_id = interaction_token_id
        self.reg_token_id = reg_token_id

    def __call__(self, batch):
        graphs_A_batch, graphs_B_batch, instruction, text_values = zip(*batch)
        
        flat_graphs_A, herb_batch_A = [], []
        flat_graphs_B, herb_batch_B = [], []
        
        for sample_idx, (g_A_list, g_B_list) in enumerate(zip(graphs_A_batch, graphs_B_batch)):
            flat_graphs_A.extend(g_A_list)
            herb_batch_A.extend([sample_idx] * len(g_A_list))
            flat_graphs_B.extend(g_B_list)
            herb_batch_B.extend([sample_idx] * len(g_B_list))

        flat_graphs_A = [g for g in flat_graphs_A if isinstance(g, Data)]
        flat_graphs_B = [g for g in flat_graphs_B if isinstance(g, Data)]
            
        batched_A = Batch.from_data_list(flat_graphs_A)
        batched_A.herb_batch = torch.tensor(herb_batch_A, dtype=torch.long)
        
        batched_B = Batch.from_data_list(flat_graphs_B)
        batched_B.herb_batch = torch.tensor(herb_batch_B, dtype=torch.long)

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(text=instruction,
                                            truncation=True,
                                            max_length=self.text_max_len,
                                            padding='longest',
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            return_attention_mask=True)

        instruction_tokens['is_interaction_token'] = instruction_tokens.input_ids == self.interaction_token_id
        instruction_tokens['is_reg_token'] = instruction_tokens.input_ids == self.reg_token_id

        text_values = torch.tensor(text_values).to(torch.int64)
        return batched_A, batched_B, instruction_tokens, text_values

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, interaction_token_id, reg_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.interaction_token_id = interaction_token_id
        self.reg_token_id = reg_token_id

    def __call__(self, batch):
        graphs_A_batch, graphs_B_batch, instruction, text_values = zip(*batch)
        
        flat_graphs_A, herb_batch_A = [], []
        flat_graphs_B, herb_batch_B = [], []
        
        for sample_idx, (g_A_list, g_B_list) in enumerate(zip(graphs_A_batch, graphs_B_batch)):
            flat_graphs_A.extend(g_A_list)
            herb_batch_A.extend([sample_idx] * len(g_A_list))
            flat_graphs_B.extend(g_B_list)
            herb_batch_B.extend([sample_idx] * len(g_B_list))
            
        batched_A = Batch.from_data_list(flat_graphs_A)
        batched_A.herb_batch = torch.tensor(herb_batch_A, dtype=torch.long)
        batched_B = Batch.from_data_list(flat_graphs_B)
        batched_B.herb_batch = torch.tensor(herb_batch_B, dtype=torch.long)
        
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(instruction,
                                            return_tensors='pt',
                                            max_length=self.text_max_len,
                                            add_special_tokens=True,
                                            padding='longest',
                                            truncation=True,
                                            return_attention_mask=True)

        instruction_tokens['is_interaction_token'] = instruction_tokens.input_ids == self.interaction_token_id
        instruction_tokens['is_reg_token'] = instruction_tokens.input_ids == self.reg_token_id

        text_values = torch.tensor(text_values).to(torch.int64) 
        return batched_A, batched_B, instruction_tokens, text_values

def smiles2data(smiles):
    if pd.isna(smiles) or smiles.strip() == "":
        return None
    try:
        graph = smiles2graph(smiles)
        x = torch.from_numpy(graph['node_feat']).long()
        edge_index = torch.from_numpy(graph['edge_index']).long()
        edge_attr = torch.from_numpy(graph['edge_feat']).long()
        
        if x.dtype != torch.long:
            x = x.long()
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        return None

def herb_smiles_list2graphs(smiles_list):
    graph_list = []
    for smi in smiles_list:
        graph = smiles2data(smi)
        if graph is not None:
            graph_list.append(graph)
            
    if len(graph_list) == 0:
        node_feat_dim = 9  
        edge_feat_dim = 3  
        empty_data = Data(
            x=torch.zeros((1, node_feat_dim), dtype=torch.long), 
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_feat_dim), dtype=torch.long)
        )
        graph_list.append(empty_data)
    return graph_list

class HerbHerbDataset(Dataset):
    def __init__(self, root_path, text_max_len, split="train", split_by_txt=False, use_cache=True, cache_dir=None):
        self.text_max_len = text_max_len
        self.interaction_g = '<|herb_interaction|>'
        self.reg_g = '<|reg_g|>'
        self.split_by_txt = split_by_txt
        self.split = split
        self.use_cache = use_cache

        self.cache_dir = Path(cache_dir) if cache_dir else Path(root_path) / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_filename = self._get_cache_filename()

        if self.split_by_txt:
            self.herb_herb_txt_path = os.path.join(root_path, f"{split}.txt")
            self.herb_ing_path = os.path.join(root_path, "Herb-Ingredient.csv")
        else:
            self.herb_herb_csv_path = os.path.join(root_path, "Herb-Herb.csv")
            self.herb_ing_path = os.path.join(root_path, "Herb-Ingredient.csv")

        if self.split_by_txt:
            self.herb_herb_df = self._parse_herb_herb_txt(self.herb_herb_txt_path)
        else:
            self.herb_herb_df = pd.read_csv(self.herb_herb_csv_path, encoding="utf-8")
        
        self.herb_ing_df = pd.read_csv(self.herb_ing_path, encoding="utf-8")
        self.herb_herb_df = self.herb_herb_df.dropna(subset=['Herb1_ID', 'Herb2_ID', 'tag'])
        self.herb_ing_df = self.herb_ing_df.dropna(subset=['Herb_ID', 'Herb_Name', 'Ingredient_Name', 'SMILES'])
        self.herb_ing_df = self.herb_ing_df[self.herb_ing_df['SMILES'] != '']
        
        if not self.split_by_txt:
            self.herb_herb_df = self.herb_herb_df[self.herb_herb_df['split'] == split]
        
        self.herb_comp_map = self.herb_ing_df.groupby('Herb_ID').agg({
            'Herb_Name': lambda x: x.iloc[0],
            'Ingredient_Name': list,
            'SMILES': list
        }).reset_index()
        
        # [废弃旧的 prompt_template 字典]
        
        self.samples = self._load_or_generate_samples()

    def _get_cache_filename(self):
        param_str = f"v3_split={self.split}_txtmaxlen={self.text_max_len}_splitbytxt={self.split_by_txt}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return self.cache_dir / f"herb_herb_samples_{self.split}_{param_hash}.pt"

    def _load_samples_cache(self):
        if not self.cache_filename.exists():
            return None
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        try:
            if torch_version >= (2, 6):
                with torch.serialization.safe_globals([TGData]):
                    samples = torch.load(self.cache_filename, map_location="cpu", weights_only=True)
            else:
                samples = torch.load(self.cache_filename, map_location="cpu")
            return samples
        except Exception:
            try:
                samples = torch.load(self.cache_filename, map_location="cpu", weights_only=False)
                return samples
            except Exception:
                return None

    def _save_samples_cache(self, samples):
        try:
            torch.save(samples, self.cache_filename)
        except Exception:
            pass

    def _load_or_generate_samples(self):
        if self.use_cache:
            cached_samples = self._load_samples_cache()
            if cached_samples is not None:
                return cached_samples
        samples = self._generate_samples()
        if self.use_cache:
            self._save_samples_cache(samples)
        return samples

    def _parse_herb_herb_txt(self, txt_path):
        txt_columns = ['Herb1_ID', 'Herb2_ID', 'tag', 'instruction']
        data = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        first_line_parts = lines[0].split('\t')
        is_header = False
        if len(first_line_parts) >= 3:
            header_candidate = first_line_parts[2].strip()
            if header_candidate.lower() == 'tag' or not header_candidate.isdigit():
                is_header = True
        
        start_line = 1 if is_header else 0
        valid_lines = lines[start_line:]
        
        for line_num, line in enumerate(valid_lines, start=start_line+1):
            try:
                parts = line.split('\t')
                parts += [''] * (len(txt_columns) - len(parts))
                row = {
                    'Herb1_ID': str(parts[0]).strip(),
                    'Herb2_ID': str(parts[1]).strip(),
                    'tag': int(parts[2].strip()),
                    'instruction': str(parts[3]).strip()
                }
                data.append(row)
            except Exception:
                continue
        df = pd.DataFrame(data)
        return df

    def _generate_samples(self):
        samples = []
        for _, herb_pair in self.herb_herb_df.iterrows():
            h1_id = herb_pair['Herb1_ID']
            h2_id = herb_pair['Herb2_ID']
            tag = int(herb_pair['tag'])

            h1_comp = self.herb_comp_map[self.herb_comp_map['Herb_ID'] == h1_id]
            h1_name = h1_comp['Herb_Name'].iloc[0] if not h1_comp.empty else "Herb A"
            h1_comp_names = h1_comp['Ingredient_Name'].iloc[0] if not h1_comp.empty else []
            h1_smiles_list = h1_comp['SMILES'].iloc[0] if not h1_comp.empty else []

            h2_comp = self.herb_comp_map[self.herb_comp_map['Herb_ID'] == h2_id]
            h2_name = h2_comp['Herb_Name'].iloc[0] if not h2_comp.empty else "Herb B"
            h2_comp_names = h2_comp['Ingredient_Name'].iloc[0] if not h2_comp.empty else []
            h2_smiles_list = h2_comp['SMILES'].iloc[0] if not h2_comp.empty else []

            if not h1_comp_names and not h2_comp_names:
                continue

            h1_comp_names = h1_comp_names if h1_comp_names else ["Unknown Component"]
            h2_comp_names = h2_comp_names if h2_comp_names else ["Unknown Component"]
            h1_smiles_list = h1_smiles_list if h1_smiles_list else [""]
            h2_smiles_list = h2_smiles_list if h2_smiles_list else [""]

            # ==========================================================
            # [阶段 3 核心] 知识唤醒与 Prompt 注入 (Top-5 核心成分)
            # ==========================================================
            top_k = 5
            h1_comp_str = ", ".join([f"[{name}]" for name in h1_comp_names[:top_k]])
            h2_comp_str = ", ".join([f"[{name}]" for name in h2_comp_names[:top_k]])

            full_prompt = (
                f"Analyze the association between {h1_name} (main active ingredients: {h1_comp_str}) "
                f"and {h2_name} (main active ingredients: {h2_comp_str}). "
                f"Based on their interaction representation <|herb_interaction|>, "
                f"explain the chemical rationale first, and then output the prediction <|reg_g|>."
            )
            # ==========================================================

            graphs_A = herb_smiles_list2graphs(h1_smiles_list)
            graphs_B = herb_smiles_list2graphs(h2_smiles_list)
            
            samples.append((graphs_A, graphs_B, full_prompt, tag))
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ProcessDatasets(LightningDataModule):
    def __init__(self, mode='pretrain', num_workers=0, batch_size=256, root='data/', text_max_len=128, tokenizer=None, args=None, split_by_txt=True, use_cache=True, cache_dir=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size if args else 8
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.root = root
        self.split_by_txt = split_by_txt
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        self.train_dataset = HerbHerbDataset(root_path=self.root, text_max_len=text_max_len, split="train", split_by_txt=self.split_by_txt, use_cache=self.use_cache, cache_dir=self.cache_dir)
        self.val_dataset = HerbHerbDataset(root_path=self.root, text_max_len=text_max_len, split="valid", split_by_txt=self.split_by_txt, use_cache=self.use_cache, cache_dir=self.cache_dir)
        self.test_dataset = HerbHerbDataset(root_path=self.root, text_max_len=text_max_len, split="test", split_by_txt=self.split_by_txt, use_cache=self.use_cache, cache_dir=self.cache_dir)
        
        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer未传入！")
        self.interaction_token_id = self.tokenizer.interaction_token_id
        self.reg_token_id = self.tokenizer.reg_token_id

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.interaction_token_id, self.reg_token_id))

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.interaction_token_id, self.reg_token_id))
        test_loader = DataLoader(self.test_dataset, batch_size=self.inference_batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.interaction_token_id, self.reg_token_id))
        return [val_loader, test_loader]

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.inference_batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.interaction_token_id, self.reg_token_id))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module (Herb-Herb Classification)")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=5)
        parser.add_argument('--inference_batch_size', type=int, default=1)
        parser.add_argument('--root', type=str, default='data/Herb-Herb_c5000/')
        parser.add_argument('--text_max_len', type=int, default=512)
        parser.add_argument('--split_by_txt', action='store_false')
        parser.add_argument('--use_cache', action='store_false')
        parser.add_argument('--cache_dir', type=str, default=None)
        return parent_parser