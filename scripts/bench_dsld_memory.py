"""DSLD 核心显存/速度基准（M3 报告 §2 表格的可复算出处；评审 2.2c 修复）。

训练步口径：前向（bf16 autocast）+ 三项损失 + 反向 + 梯度裁剪 + optimizer.step，
与 trainer.run_training 的 dsld_core 分支逐步同构；峰值显存 =
torch.cuda.max_memory_allocated（含所有驻留与瞬态），速度为 synchronize 后的
每优化步墙钟。

用法（与 M3 报告两行数字对应）：
  python scripts/bench_dsld_memory.py --batch 1 --T 32 --H 480 --W 640   # 全幅
  python scripts/bench_dsld_memory.py --batch 2 --T 32 --H 320 --W 240   # crop 档
可选：--width 1.0 --no-checkpoint --steps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dsld.models.dsld_core import DsldCore  # noqa: E402
from dsld.train.losses import decouple_loss, focal_dice_loss, recon_loss  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--H", type=int, default=480)
    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--no-checkpoint", action="store_true", help="关闭逐帧梯度检查点")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "本基准按 CUDA 训练步口径设计"
    model = DsldCore(width=args.width, use_checkpoint=not args.no_checkpoint).cuda()
    model.train()  # 检查点仅在 training 模式生效（评审 2.2c 指出的失效路径）
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.rand(args.batch, args.T, 1, args.H, args.W, device="cuda")
    tgt = (torch.rand(args.batch, args.T, 1, args.H, args.W, device="cuda") > 0.999).float()
    q = torch.ones(args.batch, device="cuda")

    def step() -> float:
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, quality=q)
        loss = (focal_dice_loss(out["logits"], tgt)
                + 0.5 * recon_loss(out["y_b"], out["x_main"], out["m_tgt"], tgt)
                + 0.1 * decouple_loss(out["h_t"], out["h_b"]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss)

    step()  # 预热（cudnn benchmark / 分配器）
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps
    print(json.dumps({
        "config": {"batch": args.batch, "T": args.T, "H": args.H, "W": args.W,
                   "width": args.width, "checkpoint": not args.no_checkpoint,
                   "steps": args.steps},
        "loss_last": round(loss, 4),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "step_s": round(dt, 3),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
