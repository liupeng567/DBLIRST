"""P0 目检与量化前置核验（M3 v2.0 方案 §8.2 P0 Gate：3 窗目检 + 判据 ① 前置）。

产出三件事：
  1. 目检图 reports/m3/p0_windows.png：每个抽样窗给出 对齐后 t=0/15/31（GT 轮廓叠加）、
     窗内逐像素时序标准差（灰度级，背景区）、以及**合成注入位置**（与关注入同 seed
     重放取差得到，红圈标出）——增强表是否真的生效，只能用像素证明；
  2. 背景稳定度（与 M1 `qc_stability_targets.bg_stability` **同式**，只是把抽样帧对
     限制在窗口内）：判据 ① 要求 ≤ M1 全局 P95 = 3.39 灰度级；
  3. 裁剪分档统计（all_tracks/one_track/one_frame/negative/fallback）与 GT 贴边率、
     轨迹覆盖率——§4.7 约束在真实数据上的兑现程度（方案只保证 ≥1 条完整）。

用法：
    python scripts/qc_windows.py --seqs 21 22 23 38 67 68 --n-windows 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dsld.data.ittd_window_dataset import IttdWindows  # noqa: E402
from dsld.data.preprocess.register import (  # noqa: E402
    dilate_mask, gt_masks_anchor, is_identity_warp, warp_frame, window_anchor_warps,
)
from dsld.utils.vis import setup_cjk  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CACHE = REPO / "data" / "cache" / "ittd"
MANIFEST = str(REPO / "data" / "manifests" / "ittd_split_v3.json")
M1_BG_P95 = 3.392885479840494  # reports/m1/qc_stability_targets.json aggregate
M1_BG_MED = 1.4199368353846413


def to_u8(x: np.ndarray) -> np.ndarray:
    """归一化域 [0,1] → 可视 8bit（2σ 线性拉伸，弱目标可见性优先）。"""
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99.9))
    hi = max(hi, lo + 1e-3)
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def window_bg_residual(seq: dict, start: int, T: int, Ws) -> dict:
    """窗内背景稳定度（灰度级）+ 滞后曲线。口径说明（两条都是 M1 面板的同式变体）：

    · M1 面板 `bg_stability` 量的是**相邻帧对**（t,t+1 互相对齐）的背景残差，全局
      中位 1.42 / P95 3.39；判据 ① 那句"≤ M1 P95"是在这个口径上说的。
    · 本函数量的是**对齐到窗锚点**后的残差（M3 慢通道真正看到的量），随滞后增长：
      lag1 与 M1 口径可比（同一相对变换），lag16/lag(T−1) 反映窗内积分代价。
      排除区必须用**锚点系** GT 并集（原帧框坐标会漏掉对齐后目标落点——正是 M3 审核
      L4 那条"平台位移污染残差统计"的同一陷阱）。
    """
    frames = seq["frames"]
    H, W = frames.shape[1:3]
    base = np.asarray(frames[start]).astype(np.int16)
    anchor_gt = gt_masks_anchor(seq["boxes"], start, T, Ws, (H, W)).max(axis=0)
    bm_bg = dilate_mask(anchor_gt, 6) < 0.5
    if bm_bg.sum() < 10_000:
        return {"lag1": float("nan"), "lag_mid": float("nan"), "lag_end": float("nan"),
                "raw_lag_end": float("nan"), "bg_px": int(bm_bg.sum())}
    curve = []
    for t in range(1, T):
        cur = np.asarray(frames[start + t])
        # 准恒等 → 免重采样用原帧（与数据集同治）；此处若误用 base 会把残差算成恒零
        w_t = (cur if is_identity_warp(Ws[t]) else warp_frame(cur, Ws[t])).astype(np.int16)
        curve.append(float(np.abs(base - w_t)[bm_bg].mean()))
    raw_end = float(np.abs(base - np.asarray(frames[start + T - 1]).astype(np.int16))[bm_bg].mean())
    return {"lag1": curve[0], "lag_mid": curve[len(curve) // 2], "lag_end": curve[-1],
            "curve_median": float(np.median(curve)), "raw_lag_end": raw_end,
            "bg_px": int(bm_bg.sum())}


def render(seqs, n_windows, T, stride, crop, seed, out_dir: Path) -> list:
    """逐窗渲染目检图。注入 footprint = 同 seed 下"关注入"重放与默认重放的正差。

    两次重放共用逐样本确定性 RNG（default_rng([seed, idx])），且 _inject 之前的所有
    随机数消耗与开关无关 → 差值即纯注入信号；start0 不一致会立即断言失败（可失败）。
    """
    rows = []
    common = dict(manifest_path=MANIFEST, cache_root=str(CACHE), split="val-int", T=T,
                  stride=stride, crop=crop, augment=True, align=True, seed=seed)
    ds = IttdWindows(**common)
    ds_off = IttdWindows(highlight=False, hotspot=False, **common)
    seqs = [s for s in seqs if s in ds.manifest["splits"]["val-int"]["seqs"]]
    if not seqs:
        raise SystemExit(f"--seqs {seqs} 不在 val-int 划分内（快评子集须来自 val-int）")
    positions = {}
    for idx, (sid, start) in enumerate(ds.index):
        positions.setdefault(sid, []).append(idx)
    for sid in seqs:
        picks = positions.get(sid, [])[:n_windows]
        if not picks:
            print(f"[跳过] seq {sid}：无可用窗（配准失败密度门限或段太短）")
            continue
        for pos in picks:
            it = ds[pos]
            start = it["start0"]
            seq = ds._load_seq(sid)
            Ws = window_anchor_warps(seq["reg"], start, T)
            stab = window_bg_residual(seq, start, T, Ws)  # dict：lag1/lag_mid/lag_end/raw
            it_off = ds_off[pos]
            assert it_off["start0"] == start, "两次重放窗口起点不一致，diff 不可用"
            d = np.clip(it["windows"][:, 0] - it_off["windows"][:, 0], 0, None)
            inj_max = d.max(axis=0)
            inj_px = int((d > 1e-6).any(axis=0).sum())
            win, msk = it["windows"][:, 0], it["target"][:, 0]
            # 归一化域 1 单位 = 16·σ_raw 灰度级（σ_raw 取窗内逐帧鲁棒 σ 中位）；
            # 只乘 16 得到的是"σ 单位"，不是灰度级——单位错一格，判据 ① 就不可比。
            sigma_raw = float(np.median(seq["stats"][start:start + T, 1]))
            std_map = np.std(win.astype(np.float32), axis=0) * 16.0 * sigma_raw
            panels = [(f"t=0  GT绿框", to_u8(win[0]), (msk[0] > 0)),
                      (f"t={T // 2}", to_u8(win[T // 2]), (msk[T // 2] > 0)),
                      (f"t={T - 1}", to_u8(win[T - 1]), (msk[T - 1] > 0)),
                      (f"时序σ(灰度级)\n中位 {np.median(std_map):.2f}",
                       np.clip(std_map / 8 * 255, 0, 255).astype(np.uint8), None),
                      ("合成注入footprint\n(×32帧 max)",
                       np.clip(inj_max * 16 * 32, 0, 255).astype(np.uint8), None)]
            fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
            for ax, (title, img, contour) in zip(axes, panels):
                ax.imshow(img, cmap="gray")
                if contour is not None:
                    # GT 在 stride-2 网格上，画回 2× 尺度轮廓
                    cs = cv2.findContours((contour.astype(np.uint8)), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)[0]
                    for c in cs:
                        ax.plot(2 * c[:, 0, 0], 2 * c[:, 0, 1], color=(0, 1, 0), lw=1.2)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            tier = it["crop_tier"]
            n_ok, n_tot = it["crop_cover"]
            axes[0].set_title(
                f"seq {sid} start {start}  {crop} 档 {tier}\n"
                f"背景残差 lag1={stab['lag1']:.2f} lag{T // 2}={stab['lag_mid']:.2f} "
                f"lag{T - 1}={stab['lag_end']:.2f} 灰度级 (M1 相邻帧 P95={M1_BG_P95:.2f})\n"
                f"轨迹完整 {n_ok}/{n_tot}  注入像素 {inj_px}", fontsize=8.0)
            png = out_dir / f"p0_seq{sid:04d}_s{start:03d}.png"
            fig.tight_layout()
            fig.savefig(png, dpi=120)
            plt.close(fig)
            rows.append({"seq_id": sid, "start": start, "tier": tier,
                         "crop_cover": [n_ok, n_tot], "gt_clipped": it["gt_clipped"],
                         "bg": stab, "inject_px": inj_px,
                         "quality": it["quality"], "png": png.name})
            print(f"seq {sid:2d} start {start:3d} 档 {tier:10s} 轨迹完整 {n_ok}/{n_tot}  "
                  f"背景 lag1 {stab['lag1']:.2f} / lag{T // 2} {stab['lag_mid']:.2f} / "
                  f"lag{T - 1} {stab['lag_end']:.2f}（未对齐 {stab['raw_lag_end']:.2f}）  "
                  f"注入像素 {inj_px}  → {png.name}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[21, 22, 23, 38, 67, 68])
    ap.add_argument("--n-windows", type=int, default=2)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8, help="训练档 8；推理档 24（P4）")
    ap.add_argument("--crop", type=int, nargs=2, default=[240, 320])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports/m3")
    args = ap.parse_args()

    setup_cjk()
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    crop = tuple(args.crop)
    rows = render(args.seqs, args.n_windows, args.T, args.stride, crop, args.seed, out_dir)
    if not rows:
        print("无可视化产出（检查 --seqs 是否属于 val-int）")
        return 1
    lag1 = np.array([r["bg"]["lag1"] for r in rows], float)
    lag_end = np.array([r["bg"][f"lag_end"] for r in rows], float)
    raw = np.array([r["bg"]["raw_lag_end"] for r in rows], float)
    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    over = [r for r in rows if r["bg"]["lag1"] > M1_BG_P95]
    summary = {
        "n_windows": len(rows), "crop": list(crop), "T": args.T, "stride": args.stride,
        "seed": args.seed,
        "tiers": tiers,
        "bg_lag1": {"median": float(np.median(lag1)), "max": float(lag1.max()),
                    "note": "与 M1 面板可比口径（相邻帧对齐残差）"},
        "bg_lag_end": {"median": float(np.median(lag_end)), "max": float(lag_end.max()),
                       "note": f"锚点系 lag{args.T - 1} 残差：慢窗积分代价，非 M1 同口径"},
        "raw_lag_end": {"median": float(np.median(raw)), "note": "不配准的对照（越小说明该段本身越稳）"},
        "m1_reference": {"bg_residual_p95": M1_BG_P95, "bg_residual_median": M1_BG_MED},
        "gate1_bg_over_m1_p95": [{"seq_id": r["seq_id"], "start": r["start"],
                                  "bg_lag1": r["bg"]["lag1"]} for r in over],
        "gt_clipped_rate": float(np.mean([r["gt_clipped"] for r in rows])),
        "track_complete_frac": [int(np.sum([r["crop_cover"][0] for r in rows])),
                                int(np.sum([r["crop_cover"][1] for r in rows]))],
        "windows": rows,
    }
    (out_dir / "p0_windows.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== P0 量化前置 ==")
    print(f"抽样窗 {len(rows)}  档位分布 {tiers}")
    print(f"背景残差 lag1（可比口径）中位 {summary['bg_lag1']['median']:.2f} "
          f"最大 {summary['bg_lag1']['max']:.2f}；"
          f"锚点系 lag{args.T - 1} 中位 {summary['bg_lag_end']['median']:.2f}；"
          f"不配准对照中位 {summary['raw_lag_end']['median']:.2f}")
    print(f"M1 参照（相邻帧口径）：中位 {M1_BG_MED:.2f}、P95 {M1_BG_P95:.2f} = 判据 ① 线")
    print(f"lag1 超 M1 P95 的窗：{len(over)}  {[(r['seq_id'], r['start']) for r in over]}")
    print(f"GT 贴边率 {summary['gt_clipped_rate']:.2f}；"
          f"轨迹整窗完整比例 {summary['track_complete_frac'][0]}/{summary['track_complete_frac'][1]}")
    print(f"产出：{out_dir}/p0_windows.json 与逐窗 PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
