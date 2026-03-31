import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks import EarlyStopping  # 新增
from pytorch_lightning.loggers import CSVLogger

from dataprocess.preprocess_dm_cls_llama3_iupac import ProcessDatasets
from model.iupac_prompt.QA_Trainer_cls_iupac import QA_Trainer

from utils import MyDeepSpeedStrategy

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
# for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = QA_Trainer.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = QA_Trainer(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    else:
        model = QA_Trainer(args)

    print('total_params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    tokenizer = model.blip2opt.llm_tokenizer

    dm = ProcessDatasets(args.mode,
                         args.num_workers,
                         args.batch_size,
                         args.root,
                         args.text_max_len,
                         tokenizer,
                         args)

    # 1. 配置早停回调（分类任务）
    early_stop_callback = EarlyStopping(
        monitor="val_auc_roc",          # 监控验证集AUC-ROC
        mode="max",                 # AUC越大越好
        patience=10,                # 连续10个epoch无提升则停止
        min_delta=1e-4,             # 最小提升阈值
        verbose=True,               # 打印早停日志
        check_finite=True,          # 检测指标是否为NaN/Inf
        stopping_threshold=0.99,    # 可选：指标达到0.99直接停止（按需设置）
    )

    # 2. 原有ModelCheckpoint回调（已修正）
    checkpoint_callback = plc.ModelCheckpoint(
        dirpath="all_checkpoints/" + args.filename + "/",
        # 修正：将 {auc_roc} 改为 {val_auc_roc}，与 File 2 中 log 的名称严格一致
        filename='{epoch:02d}-{val_auc_roc:.4f}', 
        
        # === 核心防丢策略：改为按步数保存 ===
        # 注释掉按 epoch 保存，避免与按步数保存冲突
        # every_n_epochs=args.save_every_n_epochs, 
        every_n_train_steps=1000,  # 新增：每 1000 个 batch 步数强制保存一次（数值根据你的数据集大小自行调整）
        
        save_last=True,                 # 每次触发保存时，都会同步覆写 last.ckpt
        save_top_k=3,                   # 只保存top3最优模型
        monitor="val_auc_roc",          # 按AUC排序保存最优模型
        mode="max",                     
        save_on_train_epoch_end=False   # 既然用了按步数保存，这个可以设为 False 避免重复触发
    )

    # 3. 合并回调列表
    callbacks = [early_stop_callback, checkpoint_callback]
    
    # fixme save only used parameters
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=1.0,  # 梯度裁剪，避免显存波动
        sync_batchnorm=True,    # 双卡时必须开，单卡也建议开
        num_sanity_val_steps=0, # 跳过验证集预检查（临时减少显存占用）
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,  # 包含早停+模型保存
        strategy=strategy,
        logger=logger,
        # num_sanity_val_steps=-1,
    )
    if args.mode in {'pretrain', 'ft'}:
        ckpt_path = args.resume_ckpt_path if args.resume_ckpt_path else None
        if ckpt_path:
            print(f"Resuming training from checkpoint: {ckpt_path}")
            
        # === 修改部分：加入强制异常捕获 ===
        try:
            trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        except KeyboardInterrupt:
            # 捕获手动 Ctrl+C 或系统的 Kill 信号
            print("Training manually interrupted! Saving current state...")
        except Exception as e:
            # 捕获其他异常（如 OOM）
            print(f"Training crashed due to Exception: {e}")
        finally:
            # 无论什么原因中断，强制在退出前触发一次保存
            # 注意：DeepSpeed 策略下会自动保存为目录形式
            if trainer.global_rank == 0:
                print("Force saving interrupted checkpoint...")
            interrupted_ckpt_path = os.path.join(f"./all_checkpoints/{args.filename}/", "interrupted_last.ckpt")
            trainer.save_checkpoint(interrupted_ckpt_path)
            print(f"Saved interrupted state to {interrupted_ckpt_path}")
        # ==================================
        
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        # trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="debug")
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--mode', type=str, default='ft')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser = QA_Trainer.add_model_specific_args(parser)  # add model args
    parser = ProcessDatasets.add_model_specific_args(parser)  # add data args
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--max_epochs', type=int, default=80)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args


if __name__ == '__main__':
    main(get_args())
