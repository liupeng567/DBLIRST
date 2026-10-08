"""核心链吞吐/显存实测（P1 交付件；为 P2 的日历时间与 lite 档裁决提供数字）。

用法（conda 环境 Alirst）：
    python scripts/bench_core.py --device cuda --batch 2 --iters 3      # 训练档口径
    python scripts/bench_core.py --device cpu --batch 1 --iters 1       # CPU 冒烟口径
默认按 configs/dsld_core.yaml 的窗口口径（T=32、crop[240,320]）从真实缓存取窗，
测 forward 与 forward+backward 两段耗时；CUDA 上另报峰值显存。

为什么单独一个脚本而不是塞进训练循环：P2 之前的所有裁决（stride-4 lite 档是否启用、
accum 多少、快评多久一次）都只需要**一次**干净计时，而训练循环还缺优化器/哨兵/EMA，
用它计时会把未落地代码的开销算进预算。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from dsld.train.build import build_loss_args, build_model, build_windows  # noqa: E402
from dsld.train.losses import total_loss  # noqa: E402

CFG = OmegaConf.load(REPO / "configs" / "dsld_core.yaml")


def collate(items: list[dict], device: str) -> dict:
    """四个监督/输入张量成批 + quality/seq 元数据（与 DataLoader 默认口径一致）。"""
    b = {k: torch.from_numpy(np.stack([it[k] for it in items])).to(device)
         for k in ("windows", "target", "teacher_mask", "dt")}
    b["quality"] = torch.tensor([float(it["quality"]) for it in items], device=device)
    b["seq_id"] = [int(it["seq_id"]) for it in items]
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description="DSLD 核心链吞吐实测")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--t", type=int, default=int(CFG.train.window.T))
    ap.add_argument("--crop", default="240,320")
    ap.add_argument("--c-h", dest="c_h", type=int, default=None,
                    help="覆盖 model.liquid.c_h（隐通道容量档；裁决容量时测成本用）")
    ap.add_argument("--enc-out", dest="enc_out", type=int, default=None,
                    help="覆盖 model.encoder.out_ch（= 核心 c_in）")
    ap.add_argument("--backward", action="store_true",
                    help="训练档：forward+四分量损失+backward（否则 no_grad 推理档）")
    ap.add_argument("--checkpoint", action="store_true", help="梯度检查点（显存回退档）")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("本机无 CUDA；--device cpu 或换 4060 环境跑")
    ch, cw = [int(v) for v in args.crop.split(",")]
    cfg = OmegaConf.create(OmegaConf.to_container(CFG, resolve=True))
    cfg.train.window.T = args.t
    cfg.train.window.crop = [ch, cw]
    cfg.model.liquid.use_checkpoint = bool(args.checkpoint)
    if args.c_h:
        cfg.model.liquid.c_h = int(args.c_h)
    if args.enc_out:
        cfg.model.encoder.out_ch = int(args.enc_out)
    ds = build_windows(cfg, split="val-int", augment=False)
    batch = collate([ds[i] for i in range(args.batch)], args.device)
    model = build_model(cfg).to(args.device)
    n = sum(p.numel() for p in model.parameters())
    print(f"device={args.device} B={args.batch} T={args.t} crop={[ch, cw]} "
          f"ckpt={args.checkpoint} enc_out={cfg.model.encoder.out_ch} "
          f"c_h={cfg.model.liquid.c_h} params={n:,} 输入={tuple(batch['windows'].shape)}")

    def step():
        out = model(batch["windows"], dt=batch["dt"], quality=batch["quality"],
                    teacher_mask=batch["teacher_mask"])
        if not args.backward:
            return out["logits"]
        return total_loss(out, batch, **build_loss_args(cfg))["loss"]

    model.train(args.backward)
    for _ in range(args.warmup):
        if args.backward:
            model.zero_grad(set_to_none=True)
            step().backward()
        else:
            with torch.no_grad():
                step()
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        if args.backward:
            model.zero_grad(set_to_none=True)
            loss = step()
            loss.backward()
        else:
            with torch.no_grad():
                loss = step().abs().mean()
        if args.device == "cuda":
            torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else 0.0
    print(f"{'fwd+bwd 训练档' if args.backward else 'no_grad 推理档'}: "
          f"{dt * 1000:.0f} ms/次 (batch={args.batch} → "
          f"{dt / max(args.batch, 1) * 1000:.0f} ms/窗)"
          + (f"  峰值显存 {peak:.2f} GiB" if args.device == "cuda" else "")
          + f"  标量={float(loss):.4f}")
    if not torch.isfinite(loss):
        raise SystemExit("损失非有限——冒烟不通过")


if __name__ == "__main__":
    main()
