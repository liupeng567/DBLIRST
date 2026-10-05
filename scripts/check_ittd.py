"""M0-2: ITTD 本地完整性校验（方案 1.1 / 10.8 M0 增补任务）。

校验项（全部来自方案断言口径）：
  A. 目录结构：Images / Annotation / Evaluation/cm_GT 各 87 段（1..87 无补零）
  B. 帧数：每段 250 帧 → 全量 21,750；帧名 3 位补零且连续
  C. 标注解析：实例总数 89,174；轨迹总数（seq, track_id 去重）393
  D. 交叉一致：cm_GT txt 与 XML 逐实例框一致；GT targetnum == XML 轨迹数
  E. 图像头：640×480 8bit BMP
  F. 空标注帧清单：视频 1:1-4 / 7:206-250 / 15:1-20 / 79:1-28（97 帧，合法负样本）

产出：
  - data/cache/ittd_parse_stats.json（逐段统计，供 build_manifest 复用，避免二次解析）
  - reports/m0/ittd_integrity_report.json + 控制台摘要 + 逐段实例/轨迹分布图

用法：conda activate Alirst && python scripts/check_ittd.py [--ittd-root PATH]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.ittd_parse import (  # noqa: E402
    FRAMES_PER_SEQ,
    N_SEQS,
    TOTAL_INSTANCES,
    TOTAL_TRACKS,
    load_seq_annotation,
    parse_official_txt,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ITTD = Path(r"D:\Datasets\面向空地应用的红外时敏目标检测跟踪数据集")

# 方案 10.9 固化的空标注帧清单（合法负样本，非缺失数据）
EXPECTED_EMPTY_FRAMES: dict[int, set[int]] = {
    1: set(range(1, 5)),
    7: set(range(206, 251)),
    15: set(range(1, 21)),
    79: set(range(1, 29)),
}


def read_bmp_header(path: Path) -> tuple[int, int, int]:
    """读 BMP 头 (width, height, bpp)，不解码像素。"""
    with open(path, "rb") as fp:
        head = fp.read(30)
    if head[:2] != b"BM":
        raise ValueError(f"非 BMP 文件: {path}")
    w, h = struct.unpack("<ii", head[18:26])
    (bpp,) = struct.unpack("<H", head[28:30])
    return w, h, bpp


def check_seq(seq_id: int, ittd_root: Path, check_bmp: bool) -> dict:
    """单段校验，返回逐段统计与问题清单。"""
    problems: list[str] = []
    img_dir = ittd_root / "Images" / str(seq_id)
    ann_dir = ittd_root / "Annotation" / str(seq_id)
    gt_path = ittd_root / "Evaluation" / "cm_GT" / f"{seq_id}.txt"

    for d in (img_dir, ann_dir):
        if not d.is_dir():
            problems.append(f"缺目录: {d}")
            return {"seq_id": seq_id, "problems": problems}
    if not gt_path.exists():
        problems.append(f"缺 GT: {gt_path}")

    # B. 图像帧枚举与连续性
    img_frames = sorted(p.stem for p in img_dir.glob("*.bmp"))
    expected = [f"{f:03d}" for f in range(1, FRAMES_PER_SEQ + 1)]
    if img_frames != expected:
        problems.append(
            f"Images/{seq_id} 帧列表异常: n={len(img_frames)} "
            f"(首尾 {img_frames[:1]}...{img_frames[-1:]})"
        )

    # E. BMP 头抽验（每段首/中/尾 3 帧）或全量
    bmp_checked = [1, FRAMES_PER_SEQ // 2, FRAMES_PER_SEQ] if check_bmp else [1]
    for f in bmp_checked:
        p = img_dir / f"{f:03d}.bmp"
        if p.exists():
            w, h, bpp = read_bmp_header(p)
            if (w, h, bpp) != (640, 480, 8):
                problems.append(f"BMP 头异常 {p}: {w}x{h}@{bpp}bit")

    # C. XML 标注解析
    ann = load_seq_annotation(ann_dir)
    n_xml_frames = sum(1 for v in ann.frames.values() if v is not None)
    xml_files = len(list(ann_dir.glob("*.xml")))

    # D. GT txt 交叉校验
    gt = parse_official_txt(gt_path) if gt_path.exists() else None
    gt_box_mismatch = 0
    if gt is not None:
        if gt.n_target != ann.n_tracks:
            problems.append(
                f"GT targetnum={gt.n_target} != XML 轨迹数={ann.n_tracks}"
            )
        if set(gt.frames) != set(range(1, FRAMES_PER_SEQ + 1)):
            problems.append(f"GT 帧号集异常: n={len(gt.frames)}")
        for f, gt_insts in gt.frames.items():
            xml_insts = {(i.track_id, i.box) for i in ann.frames.get(f, [])}
            gt_inst_set = {(i.track_id, i.box) for i in gt_insts}
            if xml_insts != gt_inst_set:
                gt_box_mismatch += 1
        if gt_box_mismatch:
            problems.append(f"GT 与 XML 框不一致帧数: {gt_box_mismatch}")

    # F. 空标注帧清单核对
    empty_xml = {f for f, v in ann.frames.items() if len(v) == 0}
    expected_empty = EXPECTED_EMPTY_FRAMES.get(seq_id, set())
    if empty_xml != expected_empty:
        problems.append(
            f"空标注帧与方案清单不符: 实际 {sorted(empty_xml)[:8]}..."
            f" 期望 {sorted(expected_empty)[:8]}..."
        )

    return {
        "seq_id": seq_id,
        "n_frames_img": len(img_frames),
        "n_frames_xml": n_xml_frames,
        "n_xml_files": xml_files,
        "n_instances": ann.n_instances,
        "n_tracks": ann.n_tracks,
        "track_ids": sorted(ann.track_ids),
        "n_empty_frames": len(empty_xml),
        "gt_box_mismatch_frames": gt_box_mismatch,
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ITTD 本地完整性校验（M0）")
    ap.add_argument("--ittd-root", type=Path, default=DEFAULT_ITTD)
    ap.add_argument("--bmp-all", action="store_true", help="全量 BMP 头校验（默认抽验）")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[check_ittd] root = {args.ittd_root}")
    for sub in ("Images", "Annotation", "Evaluation/cm_GT"):
        if not (args.ittd_root / sub).is_dir():
            print(f"[FATAL] 缺子目录: {args.ittd_root / sub}")
            return 1

    seqs = sorted(
        int(p.name) for p in (args.ittd_root / "Annotation").iterdir() if p.name.isdigit()
    )
    print(f"[check_ittd] 发现 {len(seqs)} 段: {seqs[:5]}...{seqs[-3:]}")

    results = []
    for i, sid in enumerate(seqs, 1):
        results.append(check_seq(sid, args.ittd_root, check_bmp=args.bmp_all))
        if i % 10 == 0 or i == len(seqs):
            print(f"  ... {i}/{len(seqs)} 段完成 ({time.time() - t0:.0f}s)")

    # 全局断言
    total_img = sum(r.get("n_frames_img", 0) for r in results)
    total_inst = sum(r.get("n_instances", 0) for r in results)
    total_tracks = sum(r.get("n_tracks", 0) for r in results)
    all_problems = [(r["seq_id"], p) for r in results for p in r["problems"]]

    checks = {
        "A_段数87": len(seqs) == N_SEQS,
        "B_总帧数21750": total_img == 21750,
        "C_实例数89174": total_inst == TOTAL_INSTANCES,
        "C_轨迹数393": total_tracks == TOTAL_TRACKS,
        "D_GT交叉一致": all(r.get("gt_box_mismatch_frames", 1) == 0 for r in results),
        "E_BMP头8bit_640x480": not any("BMP 头异常" in p for _, p in all_problems),
        "F_空标注帧清单": not any("空标注帧" in p for _, p in all_problems),
        "无其他问题": len(all_problems) == 0,
    }

    val_inst = sum(r["n_instances"] for r in results if r["seq_id"] >= 77)
    print("\n===== ITTD 完整性校验摘要 =====")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  段数={len(seqs)}  帧数={total_img}  实例={total_inst}  轨迹={total_tracks}")
    print(f"  官方验证集(77-87)实例数 = {val_inst} → 满分假设 2×N = {2 * val_inst}"
          f"（论文口径 22,418）")
    for sid, p in all_problems[:20]:
        print(f"  [问题] 序列{sid}: {p}")

    # 产出 1：逐段统计缓存（build_manifest 复用）
    cache_dir = REPO / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / "ittd_parse_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "ittd_root": str(args.ittd_root),
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sequences": results,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    # 产出 2：报告 JSON
    report_dir = REPO / "reports" / "m0"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checks": checks,
        "totals": {
            "seqs": len(seqs),
            "frames": total_img,
            "instances": total_inst,
            "tracks": total_tracks,
            "val_instances_77_87": val_inst,
            "full_score_hypothesis_2x_val": 2 * val_inst,
        },
        "n_problems": len(all_problems),
        "problems": all_problems,
    }
    (report_dir / "ittd_integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # 产出 3：逐段实例/轨迹分布图
    from dsld.utils.vis import setup_cjk

    setup_cjk()
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(14, 4.5))
    ids = [r["seq_id"] for r in results]
    ax1.bar(ids, [r["n_instances"] for r in results], color="#3778ae", label="实例数")
    ax1.set_xlabel("序列 ID")
    ax1.set_ylabel("实例数（帧×目标）", color="#3778ae")
    ax2 = ax1.twinx()
    ax2.plot(ids, [r["n_tracks"] for r in results], color="#c44e52", marker=".",
             lw=1, label="轨迹数")
    ax2.set_ylabel("轨迹数", color="#c44e52")
    for x in (64.5, 76.5):
        ax1.axvline(x, color="gray", ls="--", lw=0.8)
    ax1.text(32, ax1.get_ylim()[1] * 0.95, "train-int 1-64", ha="center")
    ax1.text(70.5, ax1.get_ylim()[1] * 0.95, "val-int 65-76", ha="center")
    ax1.text(82, ax1.get_ylim()[1] * 0.95, "val-official 77-87", ha="center")
    ax1.set_title(f"ITTD 逐段标注分布（实例 {total_inst} / 轨迹 {total_tracks}）")
    fig.tight_layout()
    fig.savefig(report_dir / "ittd_per_seq_distribution.png", dpi=150)
    plt.close(fig)

    ok = all(checks.values())
    print(f"\n[check_ittd] {'全部通过' if ok else '存在失败项'}，"
          f"统计缓存 → {stats_path.name}，报告 → reports/m0/")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
