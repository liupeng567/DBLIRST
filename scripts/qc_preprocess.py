"""M1: 预处理质检（方案 2.9）。

检查项（M1 验收 Gate）：
  ① 配准成功率（1 − reg_failed 占比）≥ 98%
  ② 配准 RMSE 分布 P95 ≤ 0.5 px
  ③ 标准化后每帧非零像素占比 ∈ [5%, 95%]（x̂>0 占比口径，防全黑/全白帧）
  ④ 热图峰值数与标注框数一致（3×3 局部极大；交汇目标可能合并，统计一致率）
  ⑤ 每段随机抽 3 窗生成可视化拼图（原图 + 配准差分 + 热图叠加）人工过目
  ⑥ 吞吐 ≥ 200 帧/s（取自 preprocess_all 的 wall 统计）

产出：reports/m1/qc_report.{json,md}、reports/m1/qc/*.png（逐段拼图）、汇总图表
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.utils.vis import setup_cjk  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "cache" / "ittd"
QC_DIR = REPO / "reports" / "m1"
WINDOW = 32

# 展示用的代表性序列（外场白天 / 内场白天 / 内场傍晚 / 外场傍晚）
SHOW_SEQS = [2, 30, 45, 70]


def imread_u8(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def count_peaks(hm: np.ndarray, thr: int = 48) -> int:
    """峰数 = [3×3 局部极大且 > thr] 的连通域数（uint8 平台会成片相等，必须取连通域）。"""
    from scipy.ndimage import label, maximum_filter

    mx = maximum_filter(hm, size=3, mode="constant")
    peaks = (hm == mx) & (hm > thr)
    _, n = label(peaks)
    return int(n)


def make_montage(seq_id: int, n_windows: int = 3, seed: int = 0) -> np.ndarray | None:
    """单段拼图：n_windows × [原图 | 配准差分 | 热图叠加]。

    差分用 M2 采样器口径：相邻两帧经 align_to_anchor 复合对齐到前一帧
    （可跨参考块边界过桥），未对齐差分会因滑动参考边界出现假性跳变。
    """
    from dsld.data.preprocess.register import align_to_anchor

    cache = CACHE / f"seq_{seq_id:04d}"
    if not (cache / "seq_meta.json").exists():
        return None
    frames = np.load(cache / "frames.u8.npy", mmap_mode="r")
    reg = np.load(cache / "reg.npz")
    hm = np.load(cache / "heatmaps.u8.npy", mmap_mode="r")
    rng = random.Random(seed + seq_id)
    starts = sorted(rng.sample(range(0, 250 - WINDOW, 16), n_windows))

    rows = []
    M = reg["M"]
    ref_idx = reg["ref_idx"]
    bridge = reg["bridge_M"] if "bridge_M" in reg else None
    for s in starts:
        t_mid = s + WINDOW // 2
        t2 = min(s + WINDOW - 1, t_mid + 1)
        img_mid = np.asarray(frames[t_mid])
        # 配准差分：t2 复合对齐到 t_mid（M2 口径），与 t_mid 原图作差
        if bridge is not None:
            W = align_to_anchor(M[t2].astype(np.float64), int(ref_idx[t2]),
                                M[t_mid].astype(np.float64), int(ref_idx[t_mid]),
                                bridge)
        else:
            W = M[t2]
        w2 = cv2.warpAffine(np.asarray(frames[t2]), W, (640, 480),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        diff = np.abs(img_mid.astype(np.int16) - w2.astype(np.int16)).astype(np.uint8)
        hm_mid = np.asarray(hm[t_mid])
        overlay = cv2.cvtColor(img_mid, cv2.COLOR_GRAY2RGB)
        m = hm_mid > 48
        overlay[m] = (255, 60, 60)  # 热图叠加（RGB 红）
        row = np.concatenate([cv2.cvtColor(img_mid, cv2.COLOR_GRAY2RGB),
                              cv2.cvtColor(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX),
                                           cv2.COLOR_GRAY2RGB),
                              overlay], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    # 列标题条
    return grid


def main() -> int:
    setup_cjk()
    import matplotlib.pyplot as plt

    metas = []
    for d in sorted(CACHE.glob("seq_*")):
        p = d / "seq_meta.json"
        if p.exists():
            metas.append(json.loads(p.read_text(encoding="utf-8")))
    metas.sort(key=lambda m: m["seq_id"])
    print(f"[qc] 载入 {len(metas)} 段 seq_meta")

    # ① ② ③ 聚合
    ok_rate = np.array([m["reg"]["success_rate"] for m in metas])
    n_failed = sum(m["reg"]["n_failed"] for m in metas)
    n_flat = sum(m["reg"]["n_flat"] for m in metas)
    total_frames = sum(m["n_frames"] for m in metas)
    rmse_p95 = np.array([m["reg"]["rmse_p95"] for m in metas if m["reg"]["rmse_p95"] is not None])
    nz_mean = np.array([m["nonzero_frac"]["mean"] for m in metas])
    nz_min = np.array([m["nonzero_frac"]["min"] for m in metas])
    nz_max = np.array([m["nonzero_frac"]["max"] for m in metas])
    method_counts = {
        k: sum(m["reg"][k] for m in metas)
        for k in ("n_klt", "n_klt2", "n_fm", "n_fm2", "n_flat", "n_failed")
    }

    gate_success = (1 - n_failed / max(1, total_frames - n_flat)) >= 0.98
    # Gate② 诚实语义（评审 2.2a）：rmse_p95 现为 KLT 帧像素域分位，但 KLT 仅在
    # RMSE≤0.5 时被接受——本 Gate 验证的是"接受门限被满足"（循环口径），独立精度
    # 证据见 M1 扩展审计（背景残差中位 1.42 灰度级、热图峰距 GT 中心 99.9%≤2px）
    gate_rmse = bool(rmse_p95.max() <= 0.5) if len(rmse_p95) else False
    gate_nonzero = bool(((nz_mean >= 0.05) & (nz_mean <= 0.95)).all())

    # ④ 热图峰值一致性（抽 12 段 × 每 10 帧检查，全量太慢）
    rng = random.Random(42)
    check_seqs = rng.sample([m["seq_id"] for m in metas], min(12, len(metas)))
    match, total_chk, merge_examples = 0, 0, []
    for sid in check_seqs:
        cache = CACHE / f"seq_{sid:04d}"
        lab = np.load(cache / "labels.npz")
        hm = np.load(cache / "heatmaps.u8.npy", mmap_mode="r")
        boxes = lab["boxes"]
        for f in range(10, 251, 10):
            n_box = int((boxes[:, 0] == f).sum())
            n_peak = count_peaks(np.asarray(hm[f - 1]))
            total_chk += 1
            if n_peak == n_box:
                match += 1
            elif len(merge_examples) < 10:
                merge_examples.append((sid, f, n_box, n_peak))
    peak_match_rate = match / max(1, total_chk)
    gate_peaks = peak_match_rate >= 0.99

    # ⑥ 吞吐（preprocess_full.json）
    thr = None
    full_json = QC_DIR / "preprocess_full.json"
    if full_json.exists():
        thr = json.loads(full_json.read_text(encoding="utf-8"))["summary"]
    gate_thr = bool(thr and thr["throughput_fps_wall"] >= 200)

    # ⑤ 逐段拼图
    qc_img_dir = QC_DIR / "qc"
    qc_img_dir.mkdir(parents=True, exist_ok=True)
    for m in metas:
        g = make_montage(m["seq_id"])
        if g is not None:
            cv2.imwrite(str(qc_img_dir / f"seq_{m['seq_id']:04d}.png"),
                        cv2.cvtColor(g, cv2.COLOR_RGB2BGR))

    # 汇总图表
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ids = [m["seq_id"] for m in metas]
    ax = axes[0, 0]
    ax.bar(ids, ok_rate * 100, color="#3778ae")
    ax.axhline(98, color="red", ls="--", lw=1, label="98% 门限")
    ax.set_ylim(90, 100.5)
    ax.set_title("① 配准成功率（%）")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.bar(ids, rmse_p95, color="#c44e52")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="0.5px 门限")
    ax.set_title("② 逐段 RMSE P95（px）")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.hist(nz_mean, bins=30, color="#55a868")
    ax.axvline(0.05, color="red", ls="--", lw=1)
    ax.axvline(0.95, color="red", ls="--", lw=1)
    ax.set_title("③ 非零像素占比均值分布（应落 [5%,95%]）")
    ax = axes[1, 1]
    clut = sorted([(m["clutter_density_mean"], m["seq_id"]) for m in metas])
    vals = [c for c, _ in clut]
    ax.plot(range(len(vals)), vals, ".-", ms=4, color="#8172b2")
    k = int(len(vals) * 0.75)
    thr_line = sorted(vals)[k]
    ax.axhline(thr_line, color="red", ls="--", lw=1,
               label=f"strong_clutter 阈(前25%)={thr_line:.2f}")
    for c, sid in clut[int(len(vals) * 0.75):]:
        ax.annotate(str(sid), (len(vals) - 1 - list(reversed(vals)).index(c) if False else list(vals).index(c), c),
                    fontsize=6)
    ax.set_title("clutter_density 排序（个/千像素）与 strong_clutter 阈值")
    ax.legend(fontsize=8)
    fig.suptitle("M1 预处理质检汇总", fontsize=13)
    fig.tight_layout()
    fig.savefig(QC_DIR / "qc_summary.png", dpi=150)
    plt.close(fig)

    report = {
        "n_seqs": len(metas),
        "total_frames": total_frames,
        "reg": {
            "n_failed": n_failed, "n_flat": n_flat, "methods": method_counts,
            "success_rate_overall": 1 - n_failed / max(1, total_frames - n_flat),
            "rmse_p95_worst_seq": float(rmse_p95.max()),
        },
        "nonzero": {"min_of_mean": float(nz_mean.min()), "max_of_mean": float(nz_mean.max())},
        "heatmap_peak_match_rate": peak_match_rate,
        "heatmap_merge_examples": merge_examples,
        "throughput": thr,
        "gates": {
            "1_配准成功率>=98%": gate_success,
            "2_RMSE_P95<=0.5px": gate_rmse,
            "3_非零占比∈[5%,95%]": gate_nonzero,
            "4_热图峰值一致>=99%": gate_peaks,
            "5_拼图人工过目": "manual",
            "6_吞吐>=200fps": gate_thr,
        },
    }
    (QC_DIR / "qc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    md = [
        "# M1 预处理质检报告（方案 2.9）", "",
        f"- 覆盖 {len(metas)} 段 / {total_frames} 帧", "",
        "| Gate | 结果 | 数值 |", "| --- | --- | --- |",
        f"| ① 配准成功率 ≥98% | {'PASS' if gate_success else 'FAIL'} | "
        f"{report['reg']['success_rate_overall']:.4%}（失败 {n_failed} 帧 / 平坦 {n_flat} 帧另计） |",
        f"| ② KLT 帧像素 RMSE P95 ≤0.5px（接受门限满足性，非独立精度证据） | "
        f"{'PASS' if gate_rmse else 'FAIL'} | 逐段 P95 最大 {rmse_p95.max():.3f}px |",
        f"| ③ 非零占比 ∈[5%,95%] | {'PASS' if gate_nonzero else 'FAIL'} | "
        f"段均值范围 [{nz_mean.min():.3f}, {nz_mean.max():.3f}] |",
        f"| ④ 热图峰值=框数 ≥99% | {'PASS' if gate_peaks else 'FAIL'} | "
        f"{peak_match_rate:.3%}（抽 12 段；合并例 {len(merge_examples)}） |",
        f"| ⑤ 拼图人工过目 | 见 reports/m1/qc/seq_*.png | 每段 3 窗 × [原图\\|配准差分\\|热图叠加] |",
        f"| ⑥ 吞吐 ≥200 帧/s | {'PASS' if gate_thr else 'FAIL'} | "
        f"{thr['throughput_fps_wall'] if thr else '-'} fps（{thr['workers'] if thr else '-'} 进程） |",
        "", f"配准方法分布：{method_counts}", "",
    ]
    if merge_examples:
        md += ["热图合并示例（交汇/近距目标，合法）：",
               "| seq | frame | boxes | peaks |", "| --- | --- | --- | --- |"]
        md += [f"| {s} | {f} | {b} | {p} |" for s, f, b, p in merge_examples]
    (QC_DIR / "qc_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report["gates"], ensure_ascii=False, indent=1))
    print(json.dumps(report["reg"], ensure_ascii=False, indent=1))
    return 0 if all(v is True or v == "manual" for v in report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
