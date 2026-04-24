import torch

ckpt_path = "all_checkpoints1/debug/last.ckpt"
# ckpt_path = "all_checkpoints/debug/interrupted_last.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# 打印核心字段是否存在
print("=== 检查点核心状态检查 ===")
# print(f"是否有 optimizer_states: {'optimizer_states' in ckpt} → {ckpt.get('optimizer_states', '缺失')}")
print(f"是否有 epoch: {'epoch' in ckpt} → {ckpt.get('epoch', '缺失')}")
print(f"是否有 global_step: {'global_step' in ckpt} → {ckpt.get('global_step', '缺失')}")