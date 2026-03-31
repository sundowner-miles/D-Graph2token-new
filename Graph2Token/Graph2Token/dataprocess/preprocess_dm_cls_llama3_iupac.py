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

# ===================== 新增：全局添加安全白名单 =====================
# 信任 torch_geometric 的 Data 类，解决 PyTorch 2.6+ 加载限制
try:
    from torch_geometric.data.data import Data as TGData
    torch.serialization.add_safe_globals([TGData])
    print("已将 torch_geometric.data.Data 加入 PyTorch 序列化安全白名单")
except Exception as e:
    print(f"添加安全白名单失败（低版本PyTorch无需处理）：{e}")

# 保持原有Collater逻辑（仅输入数据类型变化，拼接逻辑不变）
class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, reg_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_token_id = mol_token_id
        self.reg_token_id = reg_token_id

    def __call__(self, batch):
        # 适配草药场景：batch元素为 (graph_batch, instruction, tag)
        graph_batch, instruction, text_values = zip(*batch)
        # 拼接多个成分的graph为Batch
        graphs = self.collater(graph_batch)

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(text=instruction,
                                            truncation=True,
                                            max_length=self.text_max_len,
                                            padding='longest',
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            return_attention_mask=True)

        is_mol_token = instruction_tokens.input_ids == self.mol_token_id
        instruction_tokens['is_mol_token'] = is_mol_token
        is_reg_token = instruction_tokens.input_ids == self.reg_token_id
        instruction_tokens['is_reg_token'] = is_reg_token

        text_values = torch.tensor(text_values).to(torch.int64)
        return graphs, instruction_tokens, text_values


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, reg_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_token_id = mol_token_id
        self.reg_token_id = reg_token_id

    def __call__(self, batch):
        graph_batch, instruction, text_values = zip(*batch)
        graphs = self.collater(graph_batch)
        
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(instruction,
                                            return_tensors='pt',
                                            max_length=self.text_max_len,
                                            add_special_tokens=True,
                                            padding='longest',
                                            truncation=True,
                                            return_attention_mask=True)

        is_mol_token = instruction_tokens.input_ids == self.mol_token_id
        instruction_tokens['is_mol_token'] = is_mol_token
        is_reg_token = instruction_tokens.input_ids == self.reg_token_id
        instruction_tokens['is_reg_token'] = is_reg_token

        text_values = torch.tensor(text_values).to(torch.int64)
        return graphs, instruction_tokens, text_values


# 适配草药成分：支持单个SMILES转图，空值/无效SMILES返回空Data
# 核心修复：节点特征x转为LongTensor（符合Embedding层要求）
def smiles2data(smiles):
    if pd.isna(smiles) or smiles.strip() == "":
        return None
    try:
        graph = smiles2graph(smiles)
        # ------------- 核心修改1：将x从.float()改为.long() -------------
        # 分子图节点特征（如原子序数、元素类型）都是离散整数，适合用LongTensor做Embedding索引
        x = torch.from_numpy(graph['node_feat']).long()
        # edge_index 保持long类型（原有逻辑正确，无需修改）
        edge_index = torch.from_numpy(graph['edge_index']).long()
        # edge_attr 若为连续特征（如键长、键类型概率），保持float类型（原有逻辑正确）
        edge_attr = torch.from_numpy(graph['edge_feat']).long()
        
        # ------------- 新增：数据类型合理性校验（可选，方便排查问题） -------------
        if x.dtype != torch.long:
            print(f"警告：节点特征x类型异常，已强制转为long，原始类型：{x.dtype}")
            x = x.long()
        if edge_index.dtype != torch.long:
            print(f"警告：边索引edge_index类型异常，已强制转为long，原始类型：{edge_index.dtype}")
            edge_index = edge_index.long()
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        print(f"SMILES转图失败，smiles：{smiles}，错误信息：{e}")
        return None


def merge_multiple_mol_graphs(graph_list: list[Data]) -> Data:
    """
    核心函数：将多个成分的分子图节点叠加，合并为单张大图
    适配场景：所有图的节点特征维度、边特征维度完全一致
    Args:
        graph_list: 分子图列表，元素为Data对象或None
    Returns:
        Data: 合并后的单张图，无有效图时返回空图
    """
    # 过滤无效图
    valid_graphs = [g for g in graph_list if g is not None]
    if not valid_graphs:
        # 返回空图，保持类型一致
        return Data(
            x=torch.empty((0, 0), dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 0), dtype=torch.long)
        )

    # 初始化存储变量
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    node_offset = 0  # 节点索引偏移量

    for graph in valid_graphs:
        # 拼接节点特征
        x_list.append(graph.x)
        # 修正边索引：叠加当前总偏移量
        edge_index = graph.edge_index + node_offset
        edge_index_list.append(edge_index)
        # 拼接边特征
        edge_attr_list.append(graph.edge_attr)
        # 更新偏移量：累加当前图的节点数
        node_offset += graph.x.size(0)

    # 维度0拼接所有特征
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def herb_smiles_list2graph_batch(smiles_list):
    """
    改造后：将草药所有成分的SMILES转为分子图，并【节点叠加】为单张合并图
    """
    graph_list = []
    for smi in smiles_list:
        graph = smiles2data(smi)
        if graph is not None:
            graph_list.append(graph)
    
    # 核心修改：调用合并函数，返回叠加后的单张图
    merged_graph = merge_multiple_mol_graphs(graph_list)
    return merged_graph


class HerbHerbDataset(Dataset):
    def __init__(
        self, 
        root_path, 
        text_max_len, 
        split="train", 
        split_by_txt=False,
        use_cache=True,  # 新增：是否使用缓存
        cache_dir=None   # 新增：缓存目录（默认root_path/cache）
    ):
        """
        草药-草药关联二分类数据集（适配txt文件目录结构）
        :param root_path: 数据根路径（data/herb/）
        :param text_max_len: 文本最大长度
        :param split: 数据集划分 (train/valid/test)
        :param split_by_txt: 是否按独立txt文件划分（True=读取train/valid/test.txt，False=按csv的split列）
        :param use_cache: 是否使用预处理缓存（True=使用，False=重新生成并覆盖缓存）
        :param cache_dir: 缓存文件保存目录（默认root_path/cache）
        """
        self.text_max_len = text_max_len
        self.mol_g = '<|mol_g|>'
        self.reg_g = '<|reg_g|>'
        self.split_by_txt = split_by_txt
        self.split = split
        self.use_cache = use_cache

        # ------------- 缓存配置 -------------
        self.cache_dir = Path(cache_dir) if cache_dir else Path(root_path) / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)  # 自动创建缓存目录
        # 生成缓存文件名（包含关键参数，避免冲突）
        self.cache_filename = self._get_cache_filename()

        # -------------------------- 1. 路径适配：支持「txt独立文件」/「csv split列」两种方式 --------------------------
        if self.split_by_txt:
            # 方式1：读取独立txt文件（草药对数据）+ 共享的Herb-Ingredient.csv（成分数据）
            self.herb_herb_txt_path = os.path.join(root_path, f"{split}.txt")
            self.herb_ing_path = os.path.join(root_path, "Herb-Ingredient.csv")
            
            # 校验文件是否存在
            if not os.path.exists(self.herb_herb_txt_path):
                raise FileNotFoundError(f"未找到{split}集草药对txt文件：{self.herb_herb_txt_path}")
            if not os.path.exists(self.herb_ing_path):
                raise FileNotFoundError(f"未找到共享成分csv文件：{self.herb_ing_path}")
        else:
            # 方式2：原有逻辑（读取csv的split列）
            self.herb_herb_csv_path = os.path.join(root_path, "Herb-Herb.csv")
            self.herb_ing_path = os.path.join(root_path, "Herb-Ingredient.csv")
            
            # 校验文件是否存在
            if not os.path.exists(self.herb_herb_csv_path):
                raise FileNotFoundError(f"按split列划分时，未找到Herb-Herb.csv：{self.herb_herb_csv_path}")
            if not os.path.exists(self.herb_ing_path):
                raise FileNotFoundError(f"按split列划分时，未找到Herb-Ingredient.csv：{self.herb_ing_path}")

        # -------------------------- 2. 数据读取与清洗 --------------------------
        # 读取草药对数据（适配txt/csv两种格式）
        if self.split_by_txt:
            # 解析txt文件（默认制表符分隔，列顺序：Herb1_ID\tHerb2_ID\ttag\t[instruction]）
            self.herb_herb_df = self._parse_herb_herb_txt(self.herb_herb_txt_path)
        else:
            # 原有逻辑：读取csv文件
            self.herb_herb_df = pd.read_csv(self.herb_herb_csv_path, encoding="utf-8")
        
        # 读取共享的草药-成分数据（两种方式共用同一个csv）
        self.herb_ing_df = pd.read_csv(self.herb_ing_path, encoding="utf-8")
        
        # 数据清洗（保持原有逻辑）
        self.herb_herb_df = self.herb_herb_df.dropna(subset=['Herb1_ID', 'Herb2_ID', 'tag'])
        self.herb_ing_df = self.herb_ing_df.dropna(subset=['Herb_ID', 'Herb_Name', 'Ingredient_Name', 'SMILES'])
        self.herb_ing_df = self.herb_ing_df[self.herb_ing_df['SMILES'] != '']
        
        # 按split划分逻辑（仅对csv方式生效）
        if not self.split_by_txt:
            if 'split' in self.herb_herb_df.columns:
                self.herb_herb_df = self.herb_herb_df[self.herb_herb_df['split'] == split]
            else:
                raise ValueError("split_by_txt=False时，Herb-Herb.csv必须包含split列！")
        
        # -------------------------- 3. 构建草药-成分映射表（原有逻辑不变） --------------------------
        self.herb_comp_map = self.herb_ing_df.groupby('Herb_ID').agg({
            'Herb_Name': lambda x: x.iloc[0],  # 取第一个草药名
            'Ingredient_Name': list,           # 成分名称列表
            'SMILES': list                     # 成分SMILES列表
        }).reset_index()
        
        # -------------------------- 4. 定义Prompt模板（原有逻辑不变） --------------------------
        self.prompt_template = {
            "herb1_name": "The name of Herb1 is {herb1_name}.",
            "herb2_name": "The name of Herb2 is {herb2_name}.",
            "herb1_comp": "Herb 1 contains the following components:\n{herb1_comp_str}\n",
            "herb2_comp": "Herb 2 contains the following components:\n{herb2_comp_str}",
            "graph_info": "\nThe graph representations of the above molecular components are <|mol_g|> (array format, in the same order as the above components).",
            "instruction": "\nIn traditional Chinese medicine, herbal compatibility is a core principle, and the association between different herbs directly affects the efficacy and safety of formulations. Based on the IUPAC names of the components of Herb 1 and Herb 2 and their molecular graph representations (array format), is there an effective association between Herb 1 and Herb 2?",
            "predict_req": "\nPlease take into account the names of the herbal components and their graph representations (array), and generate the predictions <|reg_g|>."
        }
        
        # -------------------------- 5. 加载/生成样本（核心优化：缓存逻辑） --------------------------
        self.samples = self._load_or_generate_samples()

    def _get_cache_filename(self):
        """生成唯一的缓存文件名，包含关键参数哈希，避免冲突"""
        # 拼接关键参数生成哈希
        param_str = f"split={self.split}_txtmaxlen={self.text_max_len}_splitbytxt={self.split_by_txt}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return self.cache_dir / f"herb_herb_samples_{self.split}_{param_hash}.pt"

    def _load_samples_cache(self):
        """从缓存文件加载预处理后的样本（适配PyTorch 2.6+安全机制）"""
        if not self.cache_filename.exists():
            return None
        
        # 适配不同PyTorch版本的加载策略
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        load_kwargs = {}
        
        try:
            print(f"[缓存加载] 从 {self.cache_filename} 加载{self.split}集预处理样本...")
            
            # 策略1：PyTorch 2.6+，添加白名单后用 weights_only=True 加载（安全）
            if torch_version >= (2, 6):
                with torch.serialization.safe_globals([TGData]):  # 临时白名单
                    samples = torch.load(
                        self.cache_filename, 
                        map_location="cpu",
                        weights_only=True  # 安全模式
                    )
            # 策略2：低版本PyTorch，直接加载
            else:
                samples = torch.load(self.cache_filename, map_location="cpu")
            
            print(f"[缓存加载] 成功加载{self.split}集 {len(samples)} 个样本")
            return samples
        
        except Exception as e1:
            print(f"[缓存加载策略1失败] {e1}")
            # 降级策略：信任缓存来源，关闭 weights_only（仅在确认缓存可信时使用）
            try:
                print("[缓存加载] 降级为 weights_only=False 加载（请确保缓存文件来源可信！）")
                samples = torch.load(
                    self.cache_filename, 
                    map_location="cpu",
                    weights_only=False  # 关闭安全限制，允许加载自定义对象
                )
                print(f"[缓存加载] 降级策略成功加载{self.split}集 {len(samples)} 个样本")
                return samples
            except Exception as e2:
                print(f"[缓存加载失败] 所有策略均失败：{e2}，将重新生成样本并覆盖缓存")
                return None

    def _save_samples_cache(self, samples):
        """将预处理后的样本保存到缓存文件"""
        try:
            torch.save(samples, self.cache_filename)
            print(f"[缓存保存] 成功将{self.split}集 {len(samples)} 个样本保存到 {self.cache_filename}")
        except Exception as e:
            print(f"[缓存保存失败] {e}，跳过缓存保存")

    def _load_or_generate_samples(self):
        """优先加载缓存，无缓存/禁用缓存时生成并保存"""
        # 1. 尝试加载缓存
        if self.use_cache:
            cached_samples = self._load_samples_cache()
            if cached_samples is not None:
                return cached_samples
        
        # 2. 无缓存/禁用缓存，生成样本
        print(f"[样本生成] 开始生成{self.split}集预处理样本（无缓存/禁用缓存）...")
        samples = self._generate_samples()
        
        # 3. 保存到缓存
        if self.use_cache:
            self._save_samples_cache(samples)
        
        return samples

    def _parse_herb_herb_txt(self, txt_path):
        """
        解析草药对txt文件，转为DataFrame（保持与csv格式一致的列名）
        自动跳过表头行，容错tag列转换失败问题
        假设txt列顺序（制表符分隔）：Herb1_ID\tHerb2_ID\ttag\t[instruction]（instruction可选）
        """
        # 定义txt列对应的列名（根据你的实际txt格式调整！）
        txt_columns = ['Herb1_ID', 'Herb2_ID', 'tag', 'instruction']
        data = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            raise ValueError(f"txt文件{txt_path}为空，无有效数据")
        
        # ------------- 自动跳过表头行 -------------
        first_line_parts = lines[0].split('\t')
        is_header = False
        if len(first_line_parts) >= 3:
            header_candidate = first_line_parts[2].strip()
            if header_candidate.lower() == 'tag' or not header_candidate.isdigit():
                is_header = True
        
        # 跳过表头（如果有）
        start_line = 1 if is_header else 0
        valid_lines = lines[start_line:]
        if is_header:
            print(f"检测到txt文件表头，已自动跳过：{lines[0]}")
        
        # ------------- 解析有效数据行 -------------
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
            except Exception as e:
                print(f"警告：第{line_num}行数据解析失败，跳过该行使。错误信息：{e}，行内容：{line}")
                continue
        
        # 转为DataFrame
        df = pd.DataFrame(data)
        print(f"解析{txt_path}成功，共{len(df)}行有效数据（跳过{len(lines)-len(valid_lines)}行无效/表头数据）")
        return df

    def _generate_samples(self):
        """生成样本列表：[(graph_batch, full_prompt, tag), ...]"""
        samples = []
        for _, herb_pair in self.herb_herb_df.iterrows():
            # 提取草药对基础信息
            h1_id = herb_pair['Herb1_ID']
            h2_id = herb_pair['Herb2_ID']
            tag = int(herb_pair['tag'])  # 二分类标签（1=关联，0=无关联）

            # 匹配草药1的成分信息
            h1_comp = self.herb_comp_map[self.herb_comp_map['Herb_ID'] == h1_id]
            h1_name = h1_comp['Herb_Name'].iloc[0] if not h1_comp.empty else "Herb1_Unknown"
            h1_comp_names = h1_comp['Ingredient_Name'].iloc[0] if not h1_comp.empty else []
            h1_smiles_list = h1_comp['SMILES'].iloc[0] if not h1_comp.empty else []

            # 匹配草药2的成分信息
            h2_comp = self.herb_comp_map[self.herb_comp_map['Herb_ID'] == h2_id]
            h2_name = h2_comp['Herb_Name'].iloc[0] if not h2_comp.empty else "Herb2_Unknown"
            h2_comp_names = h2_comp['Ingredient_Name'].iloc[0] if not h2_comp.empty else []
            h2_smiles_list = h2_comp['SMILES'].iloc[0] if not h2_comp.empty else []

            # 过滤无成分的无效样本
            if not h1_comp_names and not h2_comp_names:
                continue

            # 处理空成分列表
            h1_comp_names = h1_comp_names if h1_comp_names else ["Unknown Component"]
            h2_comp_names = h2_comp_names if h2_comp_names else ["Unknown Component"]
            h1_smiles_list = h1_smiles_list if h1_smiles_list else [""]
            h2_smiles_list = h2_smiles_list if h2_smiles_list else [""]

            # 生成成分名称字符串
            h1_comp_str = "\n".join([f"- {name}" for name in h1_comp_names])
            h2_comp_str = "\n".join([f"- {name}" for name in h2_comp_names])

            # 拼接完整Prompt
            full_prompt = (
                self.prompt_template["herb1_name"].format(herb1_name=h1_name) +
                self.prompt_template["herb2_name"].format(herb2_name=h2_name) +
                self.prompt_template["herb1_comp"].format(herb1_comp_str=h1_comp_str) +
                self.prompt_template["herb2_comp"].format(herb2_comp_str=h2_comp_str) +
                self.prompt_template["graph_info"] +
                self.prompt_template["instruction"] +
                self.prompt_template["predict_req"]
            )

            # 生成草药成分的Graph Batch（合并草药1+草药2的所有成分图）
            all_smiles = h1_smiles_list + h2_smiles_list
            graph_batch = herb_smiles_list2graph_batch(all_smiles)

            # 添加样本（确保张量在CPU上，避免缓存设备冲突）
            graph_batch = graph_batch.to("cpu")
            samples.append((graph_batch, full_prompt, tag))
        
        print(f"[{self.__class__.__name__}-{self.split}] 生成 {len(samples)} 个有效草药-草药关联样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ProcessDatasets(LightningDataModule):
    def __init__(
            self,
            mode: str = 'pretrain',
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data/',
            text_max_len: int = 128,
            tokenizer=None,
            args=None,
            split_by_txt: bool = True,  # 默认为True，适配txt文件目录结构
            use_cache: bool = True,     # 新增：是否使用缓存
            cache_dir: str = None       # 新增：缓存目录
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size if args else 8
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.root = root
        self.split_by_txt = split_by_txt  # 保存划分方式（改为控制txt文件）
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # 初始化数据集（根据划分方式选择路径逻辑，传入缓存参数）
        self.train_dataset = HerbHerbDataset(
            root_path=self.root,
            text_max_len=text_max_len,
            split="train",
            split_by_txt=self.split_by_txt,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir
        )
        self.val_dataset = HerbHerbDataset(
            root_path=self.root,
            text_max_len=text_max_len,
            split="valid",
            split_by_txt=self.split_by_txt,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir
        )
        self.test_dataset = HerbHerbDataset(
            root_path=self.root,
            text_max_len=text_max_len,
            split="test",
            split_by_txt=self.split_by_txt,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir
        )
        
        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer未传入！")
        self.mol_token_id = self.tokenizer.mol_token_id
        self.reg_token_id = self.tokenizer.reg_token_id

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )
        return [val_loader, test_loader]

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module (Herb-Herb Classification)")
        parser.add_argument('--num_workers', type=int, default=2, help="数据加载的worker数")
        parser.add_argument('--batch_size', type=int, default=5, help="训练批次大小")
        parser.add_argument('--inference_batch_size', type=int, default=1, help="推理批次大小")
        parser.add_argument('--root', type=str, default='data/Herb-Herb_allin/', help="草药数据根路径")
        parser.add_argument('--text_max_len', type=int, default=512, help="文本最大长度（适配长Prompt）")
        parser.add_argument('--split_by_txt', action='store_false', help="是否按独立txt文件划分（默认True，关闭则用csv的split列）")
        # 新增缓存相关参数
        parser.add_argument('--use_cache', action='store_false', help="是否使用预处理缓存（默认True，关闭则重新生成）")
        parser.add_argument('--cache_dir', type=str, default=None, help="缓存文件保存目录（默认root/cache）")
        return parent_parser