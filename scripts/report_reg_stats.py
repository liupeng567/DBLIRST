"""配准指标修正口径统计（评审 2.2a）：直接读 87 段缓存 reg.npz 重算。

把"像素域 KLT 内点 RMSE"与"FM 帧的 1−corr 无量纲代理"拆开：
  - klt 帧的 rmse 受 0.5px 接受门限截断（≤0.5 才被接受）——其 P95 是"门限满足性"，
    不是独立精度证据；独立精度证据 = M1 扩展审计（背景残差中位 1.42 灰度级、
    热图峰距 GT 中心 99.9%≤2px）+ 合成仿射反解单测。
  - fm 帧单独报告相关峰质量（corr = 1 − rmse 字段）分布。

用法：python scripts/report_reg_stats.py [--cache data/cache/ittd]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/ittd")
    args = ap.parse_args()
    root = Path(args.cache)
    seqs = sorted(root.glob("seq_*"))
    rows = []
    for d in seqs:
        reg = np.load(d / "reg.npz")
        method = reg["method"]
        rmse = reg["rmse"]
        failed = reg["failed"]
        flat = method == "flat"
        ok = ~(failed | flat)
        klt = ok & np.isin(method, ["klt", "klt2"])
        fm = ok & np.isin(method, ["fm", "fm2"])
        rows.append({
            "seq_id": int(d.name[-4:]),
            "n": int(len(method)),
            "n_klt": int(klt.sum()), "n_fm": int(fm.sum()),
            "n_failed": int(failed.sum()), "n_flat": int(flat.sum()),
            "klt_rmse_p95_px": float(np.percentile(rmse[klt], 95)) if klt.any() else None,
            "klt_rmse_med_px": float(np.median(rmse[klt])) if klt.any() else None,
            "fm_corr_med": float(np.median(1.0 - rmse[fm])) if fm.any() else None,
            "fm_corr_min": float((1.0 - rmse[fm]).min()) if fm.any() else None,
        })
    klt_p95 = np.array([r["klt_rmse_p95_px"] for r in rows if r["klt_rmse_p95_px"] is not None])
    fm_corr = np.array([r["fm_corr_med"] for r in rows if r["fm_corr_med"] is not None])
    n_frames = sum(r["n"] for r in rows)
    n_klt = sum(r["n_klt"] for r in rows)
    n_fm = sum(r["n_fm"] for r in rows)
    summary = {
        "n_seqs": len(rows), "n_frames": n_frames,
        "frames": {"klt": n_klt, "fm": n_fm,
                   "failed": sum(r["n_failed"] for r in rows),
                   "flat": sum(r["n_flat"] for r in rows)},
        "klt_px_rmse": {
            "p95_max_seq": float(klt_p95.max()) if len(klt_p95) else None,
            "p95_med_seq": float(np.median(klt_p95)) if len(klt_p95) else None,
            "note": "受 0.5px 接受门限截断（门限满足性）；独立精度证据见 M1 扩展审计",
        },
        "fm_corr": {
            "med_of_seq_med": float(np.median(fm_corr)) if len(fm_corr) else None,
            "min_of_seq_min": (float(min(r["fm_corr_min"] for r in rows
                                         if r["fm_corr_min"] is not None))
                               if any(r["fm_corr_min"] is not None for r in rows) else None),
            "note": "FM 帧相关峰质量（1−rmse 字段还原），门限 0.5",
        },
    }
    out = Path("reports/audit/reg_stats_corrected.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_seq": rows}, ensure_ascii=False,
                              indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"明细 → {out}")


if __name__ == "__main__":
    main()
