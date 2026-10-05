"""M0-3: 生成 ITTD 划分 manifest v1 / v2（方案 1.3 v1.6 重平衡条款 + 1.5 工程实现）。

v1 = 官方口径切分（1-64 / 65-76 / 77-87）——记录失衡事实，仅作对照。
v2 = 重平衡切分（1.3 执行动作）：
  - 67–76 中的 6 段外场傍晚（71–76）调入 train-int；
  - 从 1–64 调出 6 段白天（外场 3: 21–23，内场 3: 38–40）补入 val-int；
  - 目标构成：val-int = 白天 6（内3外3）/ 傍晚 6（内2外4），train-int 含外场傍晚 ≥ 6。

产出：
  - data/manifests/ittd_split_v1.json / ittd_split_v2.json
  - reports/m0/split_rebalance_table.md（前后分布对照表，入库）
  - reports/m0/split_rebalance_comparison.png（堆叠柱状对照图）

用法：python scripts/build_manifest.py [--stats data/cache/ittd_parse_stats.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.manifest import (  # noqa: E402
    ManifestError,
    build_manifest_object,
    load_official_attributes,
    make_sequence_entry,
)

REPO = Path(__file__).resolve().parents[1]

V1_SPLITS = {
    "train-int": list(range(1, 65)),
    "val-int": list(range(65, 77)),
    "val-official": list(range(77, 88)),
}
# v2 重平衡（1.3 执行动作；具体段号为脚本确定性选择并记录进 manifest）
TO_TRAIN_FROM_VAL = list(range(71, 77))        # 外场傍晚 ×6 → train-int
TO_VAL_FROM_TRAIN = list(range(21, 24)) + list(range(38, 41))  # 白天 外3+内3 → val-int

def make_v2_splits() -> dict[str, list[int]]:
    v2 = {sp: set(v) for sp, v in V1_SPLITS.items()}
    for s in TO_TRAIN_FROM_VAL:
        v2["val-int"].discard(s)
        v2["train-int"].add(s)
    for s in TO_VAL_FROM_TRAIN:
        v2["train-int"].discard(s)
        v2["val-int"].add(s)
    return {sp: sorted(v) for sp, v in v2.items()}


def composition(manifest: dict, label: str) -> Counter:
    """统计某 manifest 各 split 的 scene×daytime 构成与难点标签覆盖。"""
    seqs = {s["seq_id"]: s for s in manifest["sequences"]}
    out: Counter = Counter()
    for sp, info in manifest["splits"].items():
        combo = Counter(
            f"{seqs[s]['scene']['value'][:3]}-{seqs[s]['daytime']['value']}"
            for s in info["seqs"]
        )
        for k, n in combo.items():
            out[f"{sp}|combo:{k}"] = n
        for tag in ("crossing", "occlusion", "static_target", "distractor_present",
                    "motion_blur"):
            n = sum(
                1 for s in info["seqs"] if seqs[s]["difficulty_tags"][tag]["value"]
            )
            out[f"{sp}|tag:{tag}"] = n
        out[f"{sp}|n_seqs"] = len(info["seqs"])
        out[f"{sp}|n_instances"] = sum(
            seqs[s]["n_instances"] for s in info["seqs"]
        )
    out["_label"] = label
    return out


def write_comparison_md(v1: dict, v2: dict, path: Path) -> None:
    seqs = {s["seq_id"]: s for s in v2["sequences"]}

    def dist_rows(manifest: dict) -> list[tuple[str, str, str, int]]:
        rows = []
        for sp in ("train-int", "val-int", "val-official"):
            combo = Counter(
                (seqs[s]["scene"]["value"], seqs[s]["daytime"]["value"])
                for s in manifest["splits"][sp]["seqs"]
            )
            for (scene, dt), n in sorted(combo.items()):
                rows.append((sp, scene, dt, n))
        return rows

    def tags_row(manifest: dict, sp: str) -> str:
        seq_ids = manifest["splits"][sp]["seqs"]
        cells = []
        for tag in ("crossing", "occlusion", "static_target", "distractor_present",
                    "motion_blur"):
            n = sum(1 for s in seq_ids if seqs[s]["difficulty_tags"][tag]["value"])
            cells.append(f"{n}/{len(seq_ids)}")
        return " | ".join(cells)

    lines = [
        "# ITTD 划分重平衡对照表（manifest v1 → v2）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（scripts/build_manifest.py）",
        "- 依据：方案 1.3 v1.6 条款——val-int（65–76）全为傍晚、train-int 无外场傍晚，",
        "  天时×场地两轴同时失衡，直接使用会把早停与阈值校准带偏。",
        "",
        "## 调整动作",
        "",
        f"- 调入 train-int（外场傍晚 ×6）：{TO_TRAIN_FROM_VAL}（取自原 val-int 的 67-76 外场傍晚段）",
        f"- 调入 val-int（白天 ×6）：外场 {TO_VAL_FROM_TRAIN[:3]} + 内场 {TO_VAL_FROM_TRAIN[3:]}",
        "- val-official（77–87）不动，保持只读。",
        "",
        "## scene×daytime 构成对照（段数）",
        "",
        "| split | scene | daytime | v1 | v2 |",
        "| --- | --- | --- | --- | --- |",
    ]
    v1_rows = {(sp, sc, dt): n for sp, sc, dt, n in dist_rows(v1)}
    v2_rows = {(sp, sc, dt): n for sp, sc, dt, n in dist_rows(v2)}
    for sp in ("train-int", "val-int", "val-official"):
        for sc in ("outdoor", "indoor"):
            for dt in ("day", "dusk"):
                lines.append(
                    f"| {sp} | {sc} | {dt} | {v1_rows.get((sp, sc, dt), 0)} "
                    f"| {v2_rows.get((sp, sc, dt), 0)} |"
                )
    lines += [
        "",
        "## 难点标签覆盖对照（有该难点的段数/总段数）",
        "",
        "| split | crossing | occlusion | static_target | distractor_present | motion_blur |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for sp in ("train-int", "val-int", "val-official"):
        lines.append(f"| v1 {sp} | {tags_row(v1, sp)} |")
        lines.append(f"| v2 {sp} | {tags_row(v2, sp)} |")
    lines += [
        "",
        "## 实例数对照",
        "",
        "| split | v1 实例 | v2 实例 |",
        "| --- | --- | --- |",
    ]
    for sp in ("train-int", "val-int", "val-official"):
        lines.append(
            f"| {sp} | {v1['summary']['n_instances'][sp]} "
            f"| {v2['summary']['n_instances'][sp]} |"
        )
    lines += [
        "",
        "## v2 三子集序列清单",
        "",
    ]
    for sp in ("train-int", "val-int", "val-official"):
        lines.append(f"- **{sp}**: {v2['splits'][sp]['seqs']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_chart(v1: dict, v2: dict, path: Path) -> None:
    from dsld.utils.vis import setup_cjk

    setup_cjk()
    import matplotlib.pyplot as plt
    import numpy as np

    seqs = {s["seq_id"]: s for s in v2["sequences"]}
    combos = [("outdoor", "day"), ("indoor", "day"), ("indoor", "dusk"),
              ("outdoor", "dusk")]
    combo_labels = ["外场-白天", "内场-白天", "内场-傍晚", "外场-傍晚"]
    colors = ["#4c9f70", "#88c9a1", "#d9a441", "#b3623a"]
    splits = ("train-int", "val-int", "val-official")

    def counts(manifest: dict, sp: str) -> list[int]:
        return [
            sum(
                1
                for s in manifest["splits"][sp]["seqs"]
                if seqs[s]["scene"]["value"] == sc and seqs[s]["daytime"]["value"] == dt
            )
            for sc, dt in combos
        ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, manifest, title in ((axes[0], v1, "v1（官方直切，失衡）"),
                                (axes[1], v2, "v2（重平衡后）")):
        x = np.arange(len(splits))
        bottom = np.zeros(len(splits))
        for (sc, dt), lab, c in zip(combos, combo_labels, colors):
            vals = np.array([counts(manifest, sp) for sp in splits])[
                :, combos.index((sc, dt))
            ]
            ax.bar(x, vals, 0.55, bottom=bottom, label=lab, color=c)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            bottom += vals
        ax.set_xticks(x, splits)
        ax.set_title(title)
        ax.set_ylim(0, 70)
    axes[0].set_ylabel("段数")
    axes[0].legend(loc="upper right", fontsize=8, title="scene×daytime")
    fig.suptitle("ITTD 内部划分重平衡前后构成对照（val-official 不变）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=REPO / "data/cache/ittd_parse_stats.json")
    ap.add_argument("--attr", type=Path,
                    default=REPO / "data/manifests/ittd_official_attributes.csv")
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    by_seq = {s["seq_id"]: s for s in stats["sequences"]}
    attrs = load_official_attributes(args.attr)

    # 交叉校验（1.6 步骤3）：属性表 87 行已在 load 断言；此处对齐解析统计
    sequences = [make_sequence_entry(sid, by_seq[sid], attrs[sid]) for sid in range(1, 88)]

    extra_common = {
        "created": time.strftime("%Y-%m-%d"),
        "source": {
            "attributes": "ittd_official_attributes.csv (provenance=official, "
                          "ITTD 数据论文附录1, 傅瑞罡等 2022)",
            "parse_stats": "data/cache/ittd_parse_stats.json (scripts/check_ittd.py)",
        },
    }

    v1 = build_manifest_object(
        "v1", V1_SPLITS, sequences,
        extra={**extra_common, "note": "官方口径直切，仅作对照；v1.6 实测天时×场地失衡"},
    )
    v2_splits = make_v2_splits()
    v2 = build_manifest_object(
        "v2", v2_splits, sequences,
        extra={
            **extra_common,
            "rebalance": {
                "reason": "val-int(65-76) 全为傍晚；train-int(1-64) 无外场傍晚（方案 1.3 v1.6）",
                "moved_to_train_int": TO_TRAIN_FROM_VAL,
                "moved_to_val_int": TO_VAL_FROM_TRAIN,
                "target": "val-int = 白天6(内3外3)/傍晚6(内2外4)；train-int 含外场傍晚>=6",
            },
        },
    )

    out_dir = REPO / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ittd_split_v1.json").write_text(
        json.dumps(v1, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "ittd_split_v2.json").write_text(
        json.dumps(v2, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    report_dir = REPO / "reports" / "m0"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_comparison_md(v1, v2, report_dir / "split_rebalance_table.md")
    write_comparison_chart(v1, v2, report_dir / "split_rebalance_comparison.png")

    # 控制台摘要
    seqs = {s["seq_id"]: s for s in sequences}
    print("===== manifest v1 vs v2 构成摘要 =====")
    for sp in ("train-int", "val-int", "val-official"):
        for ver, m in (("v1", v1), ("v2", v2)):
            ids = m["splits"][sp]["seqs"]
            n_outdoor = sum(1 for s in ids if seqs[s]["scene"]["value"] == "outdoor")
            dt = Counter(seqs[s]["daytime"]["value"] for s in ids)
            print(f"  {ver} {sp:13s} n={len(ids):2d} day={dt['day']:2d} dusk={dt['dusk']:2d} "
                  f"outdoor={n_outdoor:2d} instances={m['summary']['n_instances'][sp]}")
    print(f"\nmanifest v1 checksum: {v1['checksum']}")
    print(f"manifest v2 checksum: {v2['checksum']}")
    print("产出: data/manifests/ittd_split_v{1,2}.json, reports/m0/split_rebalance_*.{md,png}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as e:
        print(f"[ManifestError] {e}")
        raise SystemExit(2)
