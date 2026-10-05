"""M1-6: manifest 统计字段回填（方案 1.4/1.6 第三类字段，v2 → v3 版本升级）。

回填字段（provenance=auto）：
  - mean_scr / min_scr：逐轨迹 SCR = |μ_T − μ_B|/σ_B（掩码对齐，背景环 8px）
  - clutter_density：背景局部极大值密度（个/千像素），strong_clutter = 前 25%
  - long_occlusion：轨迹连续消失 ≥ 20 帧（含 max_track_gap 原始值）
  - small_target：框对角线 < 8px 实例占比 > 50%（含 frac 原始值）
  - motion_blur：官方晃动列 ∪ 自动模糊帧占比 > 10%（拉普拉斯方差 < 0.3×序列中值口径）
  - reg_difficulty：序列配准 RMSE 均值（从 reg.npz 重算，仅成功帧）

规程（1.6）：版本升级 + CHANGELOG（变更字段、理由、口径）；v2 保持不动可回溯。
产出：data/manifests/ittd_split_v3.json、reports/m1/manifest_v3_backfill.{md,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.manifest import build_manifest_object  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "cache" / "ittd"


def main() -> int:
    v2 = json.loads((REPO / "data/manifests/ittd_split_v2.json").read_text(encoding="utf-8"))
    metas = {}
    for d in sorted(CACHE.glob("seq_*")):
        p = d / "seq_meta.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            metas[m["seq_id"]] = m
    assert len(metas) == 87, f"缺 seq_meta: {87 - len(metas)} 段"
    assert sum(m["n_instances"] for m in metas.values()) == 89_174, "实例数与 ITTD 断言不符"

    # strong_clutter 阈值：clutter_density 前 25%（87 段的第 66 位，从高往低）
    clut_sorted = sorted((m["clutter_density_mean"] for m in metas.values()), reverse=True)
    sc_threshold = clut_sorted[int(87 * 0.25) - 1]

    sequences = []
    for s in v2["sequences"]:
        sid = s["seq_id"]
        m = metas[sid]
        # reg_difficulty：从 reg.npz 重算成功帧 RMSE 均值
        reg = np.load(REPO / "data/cache/ittd" / f"seq_{sid:04d}" / "reg.npz")
        ok = ~(reg["failed"] | (reg["method"] == "flat"))
        reg_mean = float(reg["rmse"][ok].mean()) if ok.any() else None

        dt = s["difficulty_tags"]
        dt["strong_clutter"] = {
            "value": bool(m["clutter_density_mean"] >= sc_threshold),
            "source": "auto", "confidence": "high",
            "raw": m["clutter_density_mean"], "threshold": sc_threshold,
        }
        dt["long_occlusion"] = {
            "value": bool(m["long_occlusion"]), "source": "auto", "confidence": "high",
            "raw": m["max_track_gap"], "threshold": 20,
        }
        dt["small_target"] = {
            "value": bool(m["small_target"]), "source": "auto", "confidence": "high",
            "raw": m["small_target_frac"], "threshold": 0.5,
        }
        official_blur = dt["motion_blur"]["value"]  # 官方"平台大幅晃动"列
        dt["motion_blur"] = {
            "value": bool(official_blur or m["blur_tag"]),
            "source": "official+auto", "confidence": "medium",
            "official_shake": official_blur, "auto_blur_ratio": m["blur_ratio"],
            "note": "官方晃动列 ∪ 拉普拉斯方差<0.3×序列中值的帧占比>10%",
        }
        s = dict(s)
        s["difficulty_tags"] = dt
        s["mean_scr"] = m["mean_scr"]
        s["min_scr"] = m["min_scr"]
        s["clutter_density"] = m["clutter_density_mean"]
        s["reg_difficulty"] = reg_mean
        sequences.append(s)

    v3 = build_manifest_object(
        "v3", {sp: info["seqs"] for sp, info in v2["splits"].items()}, sequences,
        extra={
            "created": "2026-10-05",
            "changelog": [{
                "version": "v3",
                "date": "2026-10-05",
                "change": "M1 统计字段回填（1.4/1.6 第三类）：mean_scr/min_scr/clutter_density/"
                          "reg_difficulty + strong_clutter/long_occlusion/small_target 标签 + "
                          "motion_blur 并集；splits 与 v2 完全一致",
                "caliber": "SCR=|μT−μB|/σB(背景环8px, 掩码对齐); clutter=背景局部极大/千像素; "
                           "blur=拉普拉斯方差<0.3×中值帧占比>10%",
                "source": "data/cache/ittd/seq_*/seq_meta.json (scripts/preprocess_all.py)",
            }],
        },
    )
    (REPO / "data/manifests/ittd_split_v3.json").write_text(
        json.dumps(v3, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # 分布图表 + md
    from dsld.utils.vis import setup_cjk

    setup_cjk()
    import matplotlib.pyplot as plt

    seqs = v3["sequences"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    cl = sorted([(s["clutter_density"], s["seq_id"]) for s in seqs])
    colors = ["#c44e52" if c >= sc_threshold else "#3778ae" for c, _ in cl]
    ax.bar([sid for _, sid in cl], [c for c, _ in cl], color=colors)
    ax.axhline(sc_threshold, color="k", ls="--", lw=1)
    ax.set_title(f"clutter_density（红=strong_clutter 前25%, 阈值 {sc_threshold:.2f}）")
    ax = axes[0, 1]
    scrs = [s["mean_scr"] for s in seqs if s["mean_scr"] is not None]
    ax.hist(scrs, bins=25, color="#55a868")
    ax.set_title(f"逐段 mean SCR 分布（中位数 {np.median(scrs):.1f}）")
    ax = axes[1, 0]
    regd = [s["reg_difficulty"] for s in seqs if s["reg_difficulty"] is not None]
    ax.hist(regd, bins=25, color="#8172b2")
    ax.axvline(0.5, color="red", ls="--", lw=1)
    ax.set_title("逐段配准 RMSE 均值（红虚线=0.5px 门限）")
    ax = axes[1, 1]
    tags = ["strong_clutter", "long_occlusion", "small_target", "motion_blur"]
    counts = [sum(1 for s in seqs if s["difficulty_tags"][t]["value"]) for t in tags]
    ax.bar(tags, counts, color=["#c44e52", "#dd8452", "#55a868", "#3778ae"])
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(c), ha="center")
    ax.set_title("自动标签覆盖段数（87 段）")
    fig.suptitle("manifest v3 统计字段回填分布", fontsize=13)
    fig.tight_layout()
    fig.savefig(REPO / "reports/m1/manifest_v3_backfill.png", dpi=150)
    plt.close(fig)

    md = [
        "# manifest v2 → v3 统计字段回填（M1-6）", "",
        f"- v3 checksum: `{v3['checksum']}`（splits 与 v2 完全一致，仅回填统计字段）",
        f"- strong_clutter 阈值（前 25%）：clutter_density ≥ {sc_threshold:.2f} 个/千像素",
        f"- mean SCR 中位数：{np.median(scrs):.1f}；reg RMSE 均值最大：{max(regd):.3f}px", "",
        "| 标签 | 覆盖段数 | 口径 |", "| --- | --- | --- |",
        f"| strong_clutter | {counts[0]} | clutter_density 前 25% |",
        f"| long_occlusion | {counts[1]} | 轨迹连续消失 ≥20 帧 |",
        f"| small_target | {counts[2]} | 对角线<8px 实例占比 >50% |",
        f"| motion_blur(并集) | {counts[3]} | 官方晃动 ∪ 自动模糊占比>10% |",
        "", "图表：reports/m1/manifest_v3_backfill.png",
    ]
    (REPO / "reports/m1/manifest_v3_backfill.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print(f"manifest v3 已生成；strong_clutter 阈值 = {sc_threshold:.3f}")
    print(f"标签覆盖: {dict(zip(tags, counts))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
