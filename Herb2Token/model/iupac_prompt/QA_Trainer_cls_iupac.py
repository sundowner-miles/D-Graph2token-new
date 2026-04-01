import json
import os
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,  # 用于计算特异度
    average_precision_score
)
from torch import optim

from model.iupac_prompt.QA_llama3_cls_iupac import Align2llama
from utils import AttrDict

import math
from torch.optim.lr_scheduler import _LRScheduler

# 替代 LinearWarmupCosineLRScheduler
class LinearWarmupCosineLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 热身阶段：线性递增学习率
            lr = [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2 for base_lr in self.base_lrs]
        return lr

# 替代 LinearWarmupStepLRScheduler
class LinearWarmupStepLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size  # 每多少步衰减一次
        self.gamma = gamma          # 衰减率
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 热身阶段：线性递增学习率
            lr = [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 步长衰减阶段（修正：从warmup后开始计数）
            decay_steps = self.last_epoch - self.warmup_steps
            lr = [base_lr * (self.gamma ** (decay_steps // self.step_size)) for base_lr in self.base_lrs]
        return lr

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    # try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict

class QA_Trainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        # ========== 新增：关闭严格加载模式，允许 checkpoint 缺少被冻结的参数 ==========
        self.strict_loading = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.num_beams = args.num_beams
        self.llm_tune = args.llm_tune
        self.blip2opt = Align2llama(args.gin_num_layers,
                                    args.gin_hidden_dim,
                                    args.drop_ratio,
                                    args.tune_gnn,
                                    args.llm_tune,
                                    args.opt_model,
                                    args
                                    )
        

        self.save_hyperparameters(args)
        self.test_step_outputs = []
        # ========== 修改1：初始化列表，避免首次访问报错 ==========
        self.list_targets = []
        self.list_predictions = []

    def configure_optimizers(self):
        # 1. 获取训练加载器，计算核心步数
        if self.trainer.train_dataloader is not None:
            train_loader = self.trainer.train_dataloader
        else:
            train_loader = self.trainer.datamodule.train_dataloader()
        steps_per_epoch = len(train_loader)  # 每epoch的步数
        # 总训练步数 = 总epoch × 每epoch步数（调度器需要的是步数而非epoch数）
        total_steps = self.args.max_epochs * steps_per_epoch
        # 热身步数 = 取配置值和总步数的较小值，避免热身步数超过总步数
        warmup_steps = min(self.args.warmup_steps, total_steps)

        # 2. 初始化优化器
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.args.init_lr,
                                weight_decay=self.args.weight_decay
                                )

        # 3. 初始化调度器（修正参数错误）
        self.scheduler = None
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,  # 修改2：传总步数而非epoch数
                eta_min=self.args.min_lr
            )
        elif self.args.scheduler == 'linear_warmup_step_lr':
            # 修改3：区分step_size（衰减间隔步数）和gamma（衰减率）
            # step_size默认设为1个epoch的步数，也可新增参数配置
            step_size = steps_per_epoch * 1  # 每1个epoch衰减一次
            self.scheduler = LinearWarmupStepLRScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                step_size=step_size,        # 衰减间隔步数
                gamma=self.args.lr_decay_rate  # 衰减率
            )
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError(f"不支持的调度器类型：{self.args.scheduler}")

        # 4. 修改4：返回PyTorch Lightning要求的格式（自动管理scheduler.step）
        if self.scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step",  # 按batch步数更新（匹配调度器逻辑）
                    "frequency": 1
                }
            }
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                pass
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    # ========== 修改5：修正max_epochs属性获取方式 ==========
    @property
    def max_epochs(self):
        return self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else self.args.max_epochs

    # ========== 新增：统一保存所有指标的函数（删除了旧的三个保存函数） ==========
    def save_all_metrics(self, acc, precision, recall, f1, specificity, auc_roc, aupr, stage="val"):
        if not hasattr(self.logger, 'log_dir'):
            os.makedirs("./predictions", exist_ok=True)
            log_dir = "./predictions"
        else:
            log_dir = self.logger.log_dir
        # 保存到对应阶段的指标文件（验证/测试）
        file_name = f"{stage}_metrics.txt"
        with open(os.path.join(log_dir, file_name), 'a+', encoding='utf8') as f:
            line = {
                'epoch': self.current_epoch,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'auc_roc': auc_roc,
                'aupr': aupr
            }
            f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def training_step(self, batch, batch_idx):
            # ========== 修改6：删除手动scheduler.step，由PyTorch Lightning自动管理 ==========
            batch_size = batch[-1].size(0)
            
            # ==========================================================
            # [阶段 3 核心修改]：传入 current_epoch 和 max_epochs 以触发物理锚点退火
            # ==========================================================
            loss_dict = self.blip2opt(
                batch, 
                current_epoch=self.current_epoch, 
                max_epochs=self.max_epochs
            )
            
            # 1. 记录联合优化总损失
            self.log("molecule loss", float(loss_dict['loss']), batch_size=batch_size, sync_dist=True)
            
            # 2. [阶段 3 新增] 记录拆解损失，以便在可视化工具中监控重构约束和退火过程
            if 'loss_cls' in loss_dict:
                self.log("loss_cls", float(loss_dict['loss_cls']), batch_size=batch_size, sync_dist=True)
            if 'loss_recon' in loss_dict:
                self.log("loss_recon", float(loss_dict['loss_recon']), batch_size=batch_size, sync_dist=True)
                
            # 3. 记录学习率
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            
            return loss_dict['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            pass
        elif dataloader_idx == 1:
            if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
                return
            
            # ====== Herb2Token 核心修改：解包双边图数据 ======
            batched_A, batched_B, instruction_tokens, label_values = batch
            # ============== Captioning Results =================== #
            samples = {
                'batched_A': batched_A, 
                'batched_B': batched_B, 
                'instruction_tokens': instruction_tokens
            }
            # =====================================================
            
            predictions = self.blip2opt.generate(
                samples,
                num_beams=self.num_beams,
            )
            self.list_predictions.append(predictions)
            self.list_targets.append(label_values)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # ====== Herb2Token 核心修改：解包双边图数据 ======
        batched_A, batched_B, instruction_tokens, label_values = batch
        # ============== Captioning Results =================== #
        samples = {
            'batched_A': batched_A, 
            'batched_B': batched_B, 
            'instruction_tokens': instruction_tokens
        }
        # =====================================================
        
        predictions = self.blip2opt.generate(
            samples,
            num_beams=self.num_beams,
        )
        self.test_step_outputs.append((predictions, label_values))
        return predictions, label_values

    def on_validation_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []

    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
            return

        list_predictions = self.list_predictions
        list_targets = self.list_targets

        if not list_predictions or not list_targets:
            # 空结果时，所有指标设为0
            self.log("val_acc", 0.0, sync_dist=True, prog_bar=True)
            self.log("val_precision", 0.0, sync_dist=True, prog_bar=True)
            self.log("val_recall", 0.0, sync_dist=True, prog_bar=True)
            self.log("val_f1", 0.0, sync_dist=True, prog_bar=True)
            self.log("val_specificity", 0.0, sync_dist=True, prog_bar=True)
            self.log("val_auc_roc", 0.0, sync_dist=True, prog_bar=True)
            return

        predictions = [tensor for tensor in list_predictions]
        targets = [tensor for tensor in list_targets]

        predictions = torch.cat(predictions, dim=0).cpu().to(torch.float32).numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()

        # ========== 分布式兼容：world_size 和 global_rank ==========
        if dist.is_available() and dist.is_initialized() and hasattr(self.trainer, "world_size"):
            world_size = self.trainer.world_size
            global_rank = self.global_rank if hasattr(self, "global_rank") else dist.get_rank()
        else:
            world_size = 1
            global_rank = 0

        all_predictions = [None for _ in range(world_size)]
        all_targets = [None for _ in range(world_size)]

        try:
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(all_predictions, predictions)
                dist.all_gather_object(all_targets, targets)
            else:
                all_predictions = [predictions]
                all_targets = [targets]
        except RuntimeError:
            all_predictions = [predictions]
            all_targets = [targets]

        if global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]

            if len(all_predictions) > 0 and len(all_targets) > 0:
                y_true = np.array(all_targets)
                y_prob = np.array(all_predictions)[:, 1]  # 正类概率
                y_pred = (y_prob > 0.5).astype(int)  # 阈值0.5转成预测类别

                # 计算5种核心指标
                auc_roc = roc_auc_score(y_true, y_prob)
                aupr = average_precision_score(y_true, y_prob)
                acc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # 计算特异度（二分类专属：TN / (TN + FP)）
                if len(np.unique(y_true)) == 2:  # 确保是二分类场景
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    specificity = 0.0  # 多分类/单类时特异度无意义

                # 日志打印所有指标（验证集前缀加val_）
                self.log("val_acc", acc, sync_dist=True, prog_bar=True)
                self.log("val_precision", precision, sync_dist=True, prog_bar=True)
                self.log("val_recall", recall, sync_dist=True, prog_bar=True)
                self.log("val_f1", f1, sync_dist=True, prog_bar=True)
                self.log("val_specificity", specificity, sync_dist=True, prog_bar=True)
                self.log("val_auc_roc", auc_roc, sync_dist=True, prog_bar=True)
                self.log("val_aupr", aupr, sync_dist=True, prog_bar=True)

                # 保存所有指标到文件
                self.save_all_metrics(acc, precision, recall, f1, specificity, auc_roc, aupr, stage="val")
            else:
                # 兜底：空结果时指标设为0
                self.log("val_acc", 0.0, sync_dist=True, prog_bar=True)
                self.log("val_precision", 0.0, sync_dist=True, prog_bar=True)
                self.log("val_recall", 0.0, sync_dist=True, prog_bar=True)
                self.log("val_f1", 0.0, sync_dist=True, prog_bar=True)
                self.log("val_specificity", 0.0, sync_dist=True, prog_bar=True)
                self.log("val_auc_roc", 0.0, sync_dist=True, prog_bar=True)
                self.log("val_aupr", 0.0, sync_dist=True, prog_bar=True)

    def on_test_epoch_end(self):
        # ========== 修改7：修复on_test_epoch_end的分布式兼容 ==========
        outputs = self.test_step_outputs
        if not outputs:
            self.log("test_acc", 0.0, sync_dist=False, prog_bar=True)
            self.log("test_precision", 0.0, sync_dist=False, prog_bar=True)
            self.log("test_recall", 0.0, sync_dist=False, prog_bar=True)
            self.log("test_f1", 0.0, sync_dist=False, prog_bar=True)
            self.log("test_specificity", 0.0, sync_dist=False, prog_bar=True)
            self.log("test_auc_roc", 0.0, sync_dist=False, prog_bar=True)
            self.log("test_aupr", 0.0, sync_dist=False, prog_bar=True)
            return

        list_predictions, list_targets = zip(*outputs)

        # 针对 predictions 的防御性转换
        predictions = [p.float() if isinstance(p, torch.Tensor) else torch.tensor(p).float() for p in list_predictions]
        
        # ====== 针对 targets 的防御性转换（修复 AttributeError） ======
        targets = [t.float() if isinstance(t, torch.Tensor) else torch.tensor(t).float() for t in list_targets]

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()

        # 分布式兼容逻辑
        if dist.is_available() and dist.is_initialized() and hasattr(self.trainer, "world_size"):
            world_size = self.trainer.world_size
            global_rank = self.global_rank if hasattr(self, "global_rank") else dist.get_rank()
        else:
            world_size = 1
            global_rank = 0

        all_predictions = [None for _ in range(world_size)]
        all_targets = [None for _ in range(world_size)]

        try:
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(all_predictions, predictions)
                dist.all_gather_object(all_targets, targets)
            else:
                all_predictions = [predictions]
                all_targets = [targets]
        except RuntimeError:
            all_predictions = [predictions]
            all_targets = [targets]

        if global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]

            if len(all_predictions) > 0 and len(all_targets) > 0:
                y_true = np.array(all_targets)
                y_prob = np.array(all_predictions)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)

                # 计算5种核心指标
                auc_roc = roc_auc_score(y_true, y_prob)
                aupr = average_precision_score(y_true, y_prob)
                acc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # 计算特异度
                if len(np.unique(y_true)) == 2:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    specificity = 0.0

                # 日志打印所有指标（测试集前缀加test_）
                self.log("test_acc", acc, sync_dist=False, prog_bar=True)
                self.log("test_precision", precision, sync_dist=False, prog_bar=True)
                self.log("test_recall", recall, sync_dist=False, prog_bar=True)
                self.log("test_f1", f1, sync_dist=False, prog_bar=True)
                self.log("test_specificity", specificity, sync_dist=False, prog_bar=True)
                self.log("test_auc_roc", auc_roc, sync_dist=False, prog_bar=True)
                self.log("test_aupr", aupr, sync_dist=False, prog_bar=True)

                # 保存测试集所有指标
                self.save_all_metrics(acc, precision, recall, f1, specificity, auc_roc, aupr, stage="test")
            else:
                self.log("test_acc", 0.0, sync_dist=False, prog_bar=True)
                self.log("test_precision", 0.0, sync_dist=False, prog_bar=True)
                self.log("test_recall", 0.0, sync_dist=False, prog_bar=True)
                self.log("test_f1", 0.0, sync_dist=False, prog_bar=True)
                self.log("test_specificity", 0.0, sync_dist=False, prog_bar=True)
                self.log("test_auc_roc", 0.0, sync_dist=False, prog_bar=True)
                self.log("test_aupr", 0.0, sync_dist=False, prog_bar=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.5)
        parser.add_argument('--tune_gnn', type=bool, default=True)

        # OPT
        parser.add_argument('--opt_model', type=str, default="/home/cheung-in-Yonsei/D:/Graph2Token/Graph2Token/LLM/Meta-Llama-3-8B")
        parser.add_argument('--num_beams', type=int, default=1)
        parser.add_argument('--llm_tune', action='store_true', default=False)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-6, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-7, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        # ========== 修改8：新增step_size参数，区分衰减间隔和衰减率 ==========
        parser.add_argument('--lr_decay_step', type=int, default=1, help='每多少个epoch衰减一次学习率')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr',
                            help='type of scheduler')  # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of optimizer')
        parser.add_argument('--finetune_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        # === 修改部分：新增断点续训参数 ===
        parser.add_argument('--resume_ckpt_path', type=str, default=None, help='断点续训的完整ckpt路径，包含优化器和epoch状态')
        parser.add_argument('--caption_eval_epoch', type=int, default=1)
        parser.add_argument('--save_every_n_epochs', type=int, default=5)
        return parent_parser