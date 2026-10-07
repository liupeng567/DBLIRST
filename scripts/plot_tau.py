"""τ 曲线图（M3 交付物）：读取训练 metrics.jsonl，绘制 τ_B/τ_T 逐轮分位数曲线。

用法：python scripts/plot_tau.py experiments/dsld_core_m3/metrics.jsonl \
          --out experiments/dsld_core_m3/tau_curves.png
返工后 Gate 口径（阶段 A）：tau_scale_ratio ≥ 3 为构造性质（重点看 lam_at_bound
未把自由度吃死）；θ_λ 基准分位（tau_b_*/tau_t_*）降为必要非充分，判据带即通道
硬区间 [24,192] / [2,8]——曲线顶到带边 = 调制/保持被终 clamp 吃掉（配合
lam_at_bound_* 判读）。实现值 tau_eff_*（动力学空间）以虚线叠加。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="metrics.jsonl 路径")
    ap.add_argument("--out", default=None, help="输出 PNG 路径（默认 metrics 同目录 tau_curves.png）")
    args = ap.parse_args()
    rows = load_metrics(args.metrics)
    out = Path(args.out or Path(args.metrics).parent / "tau_curves.png")

    epochs = [r["epoch"] for r in rows if "tau_b_median" in r or "tau_single_median" in r]
    if not epochs:
        raise SystemExit("metrics.jsonl 中无 τ 记录（确认 model.type=dsld_core 的训练轮次）")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2 if any("tau_b_median" in r for r in rows) else 1,
                             figsize=(11, 4.2))
    axes = np.atleast_1d(axes).ravel().tolist()  # 单/双子图统一为一维 Axes 列表

    def band(ax, rows, prefix, color, label):
        ep = [r["epoch"] for r in rows if f"{prefix}_median" in r]
        if not ep:
            return
        med = [r[f"{prefix}_median"] for r in rows if f"{prefix}_median" in r]
        lo = [r.get(f"{prefix}_p10", m) for r, m in zip(rows, med) if f"{prefix}_median" in r]
        hi = [r.get(f"{prefix}_p90", m) for r, m in zip(rows, med) if f"{prefix}_median" in r]
        ax.fill_between(ep, lo, hi, alpha=0.25, color=color, label=f"{label} p10–p90")
        ax.plot(ep, med, color=color, marker=".", ms=3, label=f"{label} 中位")
        print(f"{label}: init {med[0]} → final {med[-1]}（p10–p90 {lo[-1]}–{hi[-1]}）")

    ax = axes[0]
    band(ax, rows, "tau_b", "tab:blue", "τ_B（背景慢通道）")
    band(ax, rows, "tau_t", "tab:red", "τ_T（目标快通道）")
    # 实现值（动力学空间）叠加：虚线阶梯
    for prefix, color, label in (("tau_eff_b", "tab:blue", "τ_eff_B 实现值"),
                                 ("tau_eff_t", "tab:red", "τ_eff_T 实现值")):
        ep = [r["epoch"] for r in rows if prefix in r]
        if ep:
            val = [r[prefix] for r in rows if prefix in r]
            ax.plot(ep, val, color=color, ls="--", lw=1.0, label=label)
    ax.axhspan(24, 192, color="tab:blue", alpha=0.06)  # 返工后通道硬区间
    ax.axhspan(2, 8, color="tab:red", alpha=0.06)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("τ（帧）")
    ax.set_title("双状态液态核心 τ 迁移（返工后：区间构造分立，关注 lam_at_bound 顶边）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if len(axes) > 1:  # 单状态对照（消融 a）同图右侧
        ax = axes[1]
        band(ax, rows, "tau_single", "tab:green", "τ_single（单状态并集）")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("τ（帧）")
        ax.set_title("单状态对照 τ 迁移")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"τ 曲线 → {out}")


if __name__ == "__main__":
    main()
