#!/usr/bin/env bash
# M2 两基线全量训练 + eval 全链（顺序执行，单卡 4060 Laptop 8GB）
# 预计：MSHNet 15 轮 ≈ 7.5h；T-MSD3D 15 轮 ≈ 5.5h；eval 各 ≈ 10min
set -e
cd /d/DBLIRST
PY=D:/Anaconda_envs/envs/Alirst/python.exe

echo "=== [1/4] MSHNet 单帧基线训练（官方口径：Adagrad 0.05 / batch 4 / warm_epoch 5 / fp32）==="
$PY main.py --config-name mshnet 2>&1 | tee experiments/mshnet_B_train.log

echo "=== [2/4] MSHNet eval 全链（val-int 指标 + val-official 官方分 + 效率）==="
$PY scripts/eval_baseline.py --ckpt experiments/mshnet_B/ckpt/B_0_15.pt 2>&1 | tee experiments/mshnet_B_eval.log

echo "=== [3/4] T-MSD3D 时序基线训练（AdamW 3e-4 + warmup/cosine / bf16 / T=32 裁剪 240×320）==="
$PY main.py --config-name msd3d 2>&1 | tee experiments/msd3d_B_train.log

echo "=== [4/4] T-MSD3D eval 全链 ==="
$PY scripts/eval_baseline.py --ckpt experiments/msd3d_B/ckpt/B_0_15.pt 2>&1 | tee experiments/msd3d_B_eval.log

echo "=== M2 两基线训练+评测全部完成 ==="
