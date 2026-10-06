"""M1 QC 附加定量面板：背景稳定度 + 目标捕捉精度（覆盖全部 87 段）。

背景稳定度（对齐质量）：
  对抽样帧对 (t, t+1)，各自 warpAffine 到参考帧（WARP_INVERSE_MAP 口径）后取差分，
  在"背景区"（排除 GT 框外扩 6px）统计平均残差灰度；同时给出未对齐原始差分作对照。
  指标：bg_residual（对齐后背景残差，灰度级，越低越稳）、improvement（原始/对齐比值）。

目标捕捉精度（热图落点）：
  对抽样实例，取 GT 框中心 7×7 邻域内热图峰值位置，量测峰与中心距离。
  指标：≤1px / ≤2px / >3px 占比（σ=1.5 量化热图理论上 ≤0.71px，除非多峰互扰）。

产出：reports/m1/qc_stability_targets.{json,png}，并打印按指标排序的"最值得目检"榜单。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.utils.vis import setup_cjk  # noqa: E402

from dsld.data.preprocess.register import align_to_anchor  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "cache" / "ittd"


def bg_stability(seq_id: int, n_pairs: int = 24) -> dict:
    """背景稳定度（M2 采样器口径：每对帧用 align_to_anchor 复合对齐到前一帧）。"""
    cache = CACHE / f"seq_{seq_id:04d}"
    frames = np.load(cache / "frames.u8.npy", mmap_mode="r")
    reg = np.load(cache / "reg.npz")
    lab = np.load(cache / "labels.npz")
    boxes = lab["boxes"]
    M = reg["M"].astype(np.float64)
    ref_idx = reg["ref_idx"]
    bridge = reg["bridge_M"] if "bridge_M" in reg else None

    def bg_mask(f: int) -> np.ndarray:
        m = np.ones((480, 640), bool)
        for x1, y1, x2, y2 in boxes[boxes[:, 0] == f][:, 1:5]:
            m[max(0, y1 - 6):y2 + 7, max(0, x1 - 6):x2 + 7] = False
        return m

    pair_idx = np.linspace(0, 248, n_pairs).astype(int)
    res_aligned, res_raw = [], []
    for t in pair_idx:
        f1, f2 = int(t) + 1, int(t) + 2  # 1-based 帧号
        if bridge is not None:
            W = align_to_anchor(M[f2 - 1], int(ref_idx[f2 - 1]), M[f1 - 1],
                                int(ref_idx[f1 - 1]), bridge)
        else:
            W = M[f2 - 1]
        w1 = np.asarray(frames[f1 - 1])
        w2 = cv2.warpAffine(np.asarray(frames[f2 - 1]), W, (640, 480),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        bm = bg_mask(f1) & bg_mask(f2)
        if bm.sum() < 10_000:
            continue
        d_al = np.abs(w1.astype(np.int16) - w2.astype(np.int16))[bm].mean()
        d_raw = np.abs(w1.astype(np.int16)
                       - np.asarray(frames[f2 - 1]).astype(np.int16))[bm].mean()
        res_aligned.append(float(d_al))
        res_raw.append(float(d_raw))
    ra, rr = float(np.median(res_aligned)), float(np.median(res_raw))
    return {"bg_residual": ra, "raw_residual": rr,
            "improvement": rr / max(ra, 1e-6)}


def capture_accuracy(seq_id: int, step: int = 10) -> dict:
    cache = CACHE / f"seq_{seq_id:04d}"
    lab = np.load(cache / "labels.npz")
    hm = np.load(cache / "heatmaps.u8.npy", mmap_mode="r")
    boxes = lab["boxes"]
    frames_f = np.unique(boxes[:, 0])
    frames_f = frames_f[(frames_f - 1) % step == 0]
    d_all = []
    for f in frames_f:
        h = np.asarray(hm[int(f) - 1])
        for x1, y1, x2, y2 in boxes[boxes[:, 0] == f][:, 1:5]:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            x0, y0 = int(cx) - 3, int(cy) - 3
            patch = h[max(0, y0):y0 + 7, max(0, x0):x0 + 7]
            if patch.max() == 0:
                d_all.append(99.0)
                continue
            py, px = np.unravel_index(patch.argmax(), patch.shape)
            d = float(np.hypot(px + max(0, x0) - cx, py + max(0, y0) - cy))
            d_all.append(d)
    d_arr = np.array(d_all)
    if not len(d_arr):
        return {"n": 0}
    return {"n": int(len(d_arr)),
            "p_le1": float((d_arr <= 1.01).mean()),
            "p_le2": float((d_arr <= 2.01).mean()),
            "p_gt3": float((d_arr > 3.01).mean()),
            "median_d": float(np.median(d_arr))}


def main() -> int:
    setup_cjk()
    import matplotlib.pyplot as plt

    rows = {}
    for sid in range(1, 88):
        s = bg_stability(sid)
        s.update(capture_accuracy(sid))
        rows[sid] = s
        if sid % 20 == 0:
            print(f"  ... {sid}/87")

    # 榜单：最不稳定 / 捕捉最差
    worst_stab = sorted(rows.items(), key=lambda kv: -kv[1]["bg_residual"])[:10]
    worst_cap = sorted(
        ((sid, r) for sid, r in rows.items() if r.get("n", 0) > 0),
        key=lambda kv: -kv[1]["p_gt3"])[:10]
    print("\n背景残差最高 10 段（灰度级，越低越稳）:")
    for sid, r in worst_stab:
        print(f"  seq{sid:3d}  bg={r['bg_residual']:.2f}  raw={r['raw_residual']:.2f}  "
              f"improve={r['improvement']:.1f}x")
    print("\n热图中心距 >3px 占比最高 10 段:")
    for sid, r in worst_cap:
        print(f"  seq{sid:3d}  >3px={r['p_gt3']:.2%}  median_d={r['median_d']:.2f}")

    agg = {
        "bg_residual_median": float(np.median([r["bg_residual"] for r in rows.values()])),
        "bg_residual_p95": float(np.percentile([r["bg_residual"] for r in rows.values()], 95)),
        "capture_le2_overall": float(np.sum(r.get("n", 0) * r.get("p_le2", 0) for r in rows.values())
                                     / max(1, sum(r.get("n", 0) for r in rows.values()))),
        "capture_gt3_overall": float(np.sum(r.get("n", 0) * r.get("p_gt3", 0) for r in rows.values())
                                     / max(1, sum(r.get("n", 0) for r in rows.values()))),
    }
    out = REPO / "reports" / "m1"
    (out / "qc_stability_targets.json").write_text(
        json.dumps({"aggregate": agg, "per_seq": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sids = list(rows)
    ax = axes[0]
    ax.bar(sids, [rows[s]["bg_residual"] for s in sids], color="#3778ae")
    ax.set_title("对齐后背景残差（灰度级）")
    ax = axes[1]
    ax.bar(sids, [rows[s]["improvement"] for s in sids], color="#55a868")
    ax.set_title("稳定化增益（原始差分/对齐差分）")
    ax = axes[2]
    ax.bar(sids, [100 * rows[s].get("p_le2", 0) for s in sids], color="#c44e52",
           label="峰距≤2px")
    ax.bar(sids, [100 * rows[s].get("p_gt3", 0) for s in sids],
           bottom=[100 * rows[s].get("p_le2", 0) for s in sids], color="#dddddd",
           label=">2px")
    ax.set_ylim(95, 100.3)
    ax.set_title("热图峰-框中心距离分布（%）")
    ax.legend(fontsize=8)
    fig.suptitle("背景稳定度 × 目标捕捉精度（全部 87 段）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "qc_stability_targets.png", dpi=150)
    print("\naggregate:", json.dumps(agg, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
