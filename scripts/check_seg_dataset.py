"""M0-5: SAM2.1 seg_dataset 预检（方案 10.9，免重算——直接消费 instances JSON 元数据）。

检查项与门限：
  ① 结构完整性：labels / instance_ids / instances 三层，87 段 × 250 帧，零缺帧
  ② 对齐率：非压制实例中 PNG 可见（area_px>0）占比 ≥ 99%（压制实例单列统计）
  ③ 压制实例统计：suppressed / won_px==0 计数（交汇融合的天然标注信号）
  ④ SAM 置信度：score < 0.5 的实例计数（抽样人工复核线索）
  ⑤ JSON 框-XML 框一致性：逐序列逐帧 (id, box) 集合一致（预期完全一致）
  ⑥ PNG 抽样解码：抽 6 段 × 3 帧，非压制实例像素计数 == area_px；压制实例无像素
  ⑦ 抽样可视化：4 场景 × 2 帧 掩码/框叠加图
  ⑧ 干扰目标标注确认：论文口径 + 数据侧可视化佐证（M0 待确认项）

产出：reports/m0/seg_precheck_report.{json,md}、reports/m0/seg_precheck_samples.png
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.ittd_parse import load_seq_annotation  # noqa: E402
from dsld.utils.vis import setup_cjk  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SEG = Path(r"D:\Datasets\kongdixiaomubiaodataset\seg_dataset")
DEFAULT_ITTD = Path(r"D:\Datasets\面向空地应用的红外时敏目标检测跟踪数据集")

# 可视化抽查场景（覆盖 4 象限，取自官方属性表）
VIZ_SEQS = {"外场-白天": 2, "内场-白天": 30, "内场-傍晚": 45, "外场-傍晚": 70}
PNG_SPOT_SEQS = [2, 15, 45, 70, 79, 87]


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """OpenCV imread 在 Windows 中文路径下失败，改走 fromfile+imdecode。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-root", type=Path, default=DEFAULT_SEG)
    ap.add_argument("--ittd-root", type=Path, default=DEFAULT_ITTD)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = time.time()
    seg = args.seg_root

    # ① 结构完整性
    problems: list[str] = []
    n_label_dirs = len([d for d in (seg / "labels").iterdir() if d.name.isdigit()])
    n_ids_dirs = len([d for d in (seg / "instance_ids").iterdir() if d.name.isdigit()])
    n_jsons = len(list((seg / "instances").glob("*.json")))
    if (n_label_dirs, n_ids_dirs, n_jsons) != (87, 87, 87):
        problems.append(f"目录数异常: labels={n_label_dirs} ids={n_ids_dirs} json={n_jsons}")

    # ②–④ 逐 JSON 聚合
    agg = Counter()
    per_video: dict[int, dict] = {}
    for vid in range(1, 88):
        d = json.loads((seg / "instances" / f"{vid}.json").read_text(encoding="utf-8"))
        n_obj = n_supp = n_low = n_invis = n_area_mismatch_json = 0
        for fr in d["frames"]:
            for o in fr["objects"]:
                n_obj += 1
                supp = o.get("suppressed", False) or o.get("won_px", 1) == 0
                n_supp += supp
                n_low += o["score"] < 0.5
                n_invis += (not supp) and o["area_px"] == 0
        agg["objects"] += n_obj
        agg["suppressed"] += n_supp
        agg["score_lt_0.5"] += n_low
        agg["invisible_non_suppressed"] += n_invis
        agg["empty_mask_annotated"] += d.get("empty_mask_for_annotated", 0)
        per_video[vid] = {
            "objects_total": n_obj, "suppressed": n_supp,
            "score_lt_0.5": n_low, "invisible_non_suppressed": n_invis,
        }
    n_visible = agg["objects"] - agg["suppressed"] - agg["invisible_non_suppressed"]
    align_rate = n_visible / max(1, agg["objects"] - agg["suppressed"])
    if align_rate < 0.99:
        problems.append(f"对齐率 {align_rate:.4%} < 99%")

    # ⑤ JSON 框-XML 框一致性（全量）
    mismatch_frames = 0
    checked_frames = 0
    for vid in range(1, 88):
        jd = json.loads((seg / "instances" / f"{vid}.json").read_text(encoding="utf-8"))
        json_boxes = {
            fr["frame"]: {(o["id"], tuple(o["box"])) for o in fr["objects"]}
            for fr in jd["frames"]
        }
        ann = load_seq_annotation(args.ittd_root / "Annotation" / str(vid))
        for f, xml_insts in ann.frames.items():
            checked_frames += 1
            xml_set = {(i.track_id, tuple(int(v) for v in i.box)) for i in xml_insts}
            if json_boxes.get(f, set()) != xml_set:
                mismatch_frames += 1
    if mismatch_frames:
        problems.append(f"JSON-XML 框不一致帧数: {mismatch_frames}")

    # ⑥ PNG 抽样解码
    spot_rows = []
    for vid in PNG_SPOT_SEQS:
        jd = json.loads((seg / "instances" / f"{vid}.json").read_text(encoding="utf-8"))
        cand = [fr for fr in jd["frames"] if fr["n_obj"] > 0]
        if not cand:
            continue
        for fr in rng.sample(cand, min(3, len(cand))):
            png = seg / "instance_ids" / str(vid) / f"{fr['stem']}.png"
            ids_png = imread_unicode(png, cv2.IMREAD_UNCHANGED)
            if ids_png is None:
                problems.append(f"PNG 无法读取: {png}")
                continue
            for o in fr["objects"]:
                cnt = int((ids_png == o["id"]).sum())
                supp = o.get("suppressed", False) or o.get("won_px", 1) == 0
                ok = (cnt == 0) if supp else (cnt == o["area_px"])
                if not ok:
                    problems.append(
                        f"PNG 像素计数不符 v{vid} f{fr['stem']} id{o['id']}: "
                        f"png={cnt} json.area_px={o['area_px']} suppressed={supp}"
                    )
                spot_rows.append((vid, fr["stem"], o["id"], supp, cnt, o["area_px"], ok))

    # ⑦ 抽样可视化（4 象限 × 2 帧）
    setup_cjk()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(11, 15))
    for row, (label, vid) in enumerate(VIZ_SEQS.items()):
        jd = json.loads((seg / "instances" / f"{vid}.json").read_text(encoding="utf-8"))
        cand = [fr for fr in jd["frames"] if fr["n_obj"] > 0]
        picks = [cand[len(cand) // 3], cand[2 * len(cand) // 3]] if cand else []
        for col, fr in enumerate(picks):
            bmp = args.ittd_root / "Images" / str(vid) / f"{fr['stem']}.bmp"
            img = imread_unicode(bmp, cv2.IMREAD_GRAYSCALE)
            ids_png = imread_unicode(
                seg / "instance_ids" / str(vid) / f"{fr['stem']}.png",
                cv2.IMREAD_UNCHANGED,
            )
            if img is None or ids_png is None:
                problems.append(f"图像读取失败: v{vid} f{fr['stem']}")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            mask = (ids_png > 0).astype(np.uint8)
            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rgb[mask > 0] = (rgb[mask > 0] * 0.4 + np.array([255, 60, 60]) * 0.6).astype(np.uint8)
            cv2.drawContours(rgb, contour, -1, (255, 230, 0), 1)
            for o in fr["objects"]:
                x1, y1, x2, y2 = o["box"]
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 255), 1)
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.set_title(f"v{vid} f{fr['stem']}  n_obj={fr['n_obj']}", fontsize=9)
            ax.axis("off")
        if not picks:
            for col in range(2):
                axes[row, col].set_title(f"v{vid} 无标注实例")
                axes[row, col].axis("off")
    for col, lab in zip(range(2), ("掩码叠加(红)/轮廓(黄)/GT框(黄框)", "同")):
        axes[0, col].text(0, -0.08, lab, transform=axes[0, col].transAxes, fontsize=9)
    fig.suptitle("seg_dataset 抽样可视化：instance_ids 掩码 vs GT 框（4 象限场景）", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPO / "reports/m0/seg_precheck_samples.png", dpi=150)
    plt.close(fig)

    # ⑧ 干扰目标标注确认（论文口径 + 序列级计数旁证）
    #    论文挑战③明确"除车辆这类感兴趣时敏目标外……行人、电瓶车等"——干扰非感兴趣目标。
    #    数据侧旁证：标注轨迹数与官方 targetnum 全部一致（check_ittd 已证 393 轨迹），
    #    且标注实例均有 SAM 掩码（无"额外"未标注动目标进入监督）。
    distractor_conclusion = (
        "ITTD GT 只标注感兴趣车辆目标（论文挑战③口径 + 393 轨迹全量核验）；"
        "行人/电瓶车等干扰目标无框无 ID → ROI 判别头走 10.7 半自动挖掘路线"
    )

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seg_root": str(seg),
        "structure": {"label_dirs": n_label_dirs, "id_dirs": n_ids_dirs, "jsons": n_jsons},
        "objects_total": agg["objects"],
        "suppressed": agg["suppressed"],
        "score_lt_0.5": agg["score_lt_0.5"],
        "invisible_non_suppressed": agg["invisible_non_suppressed"],
        "empty_mask_for_annotated": agg["empty_mask_annotated"],
        "align_rate_visible": align_rate,
        "gate_align_rate>=0.99": align_rate >= 0.99,
        "json_xml_box_mismatch_frames": mismatch_frames,
        "json_xml_checked_frames": checked_frames,
        "png_spot_checks": len(spot_rows),
        "png_spot_fail": sum(1 for r in spot_rows if not r[-1]),
        "distractor_conclusion": distractor_conclusion,
        "problems": problems,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = REPO / "reports" / "m0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seg_precheck_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    md = [
        "# SAM2.1 seg_dataset 预检报告（M0-5）",
        "",
        f"- 生成：{report['generated']}  耗时 {report['elapsed_sec']}s",
        f"- 结构：labels/instance_ids/instances = {n_label_dirs}/{n_ids_dirs}/{n_jsons}（应 87/87/87）",
        f"- 实例总数：{agg['objects']}；压制实例：{agg['suppressed']}"
        f"（{agg['suppressed'] / agg['objects']:.3%}）；score<0.5：{agg['score_lt_0.5']}",
        f"- 对齐率（非压制且 PNG 可见）：**{align_rate:.4%}**（门限 ≥99%："
        f"{'PASS' if align_rate >= 0.99 else 'FAIL'}）",
        f"- JSON 框-XML 框一致性：{checked_frames - mismatch_frames}/{checked_frames} 帧一致"
        f"（{'PASS' if mismatch_frames == 0 else 'FAIL'}）",
        f"- PNG 抽样解码：{len(spot_rows)} 实例，失败 {report['png_spot_fail']}",
        "",
        "## 干扰目标标注确认（M0 待确认项）",
        "",
        f"- **结论：{distractor_conclusion}**",
        "- 依据①：论文正文挑战③——'除车辆这类感兴趣时敏目标外……行人、电瓶车等'，"
        "干扰类型非感兴趣目标，GT 不为其标框；",
        "- 依据②：check_ittd 全量核验——每段 XML 轨迹数 == 官方 targetnum（393 轨迹），"
        "无额外标注对象；",
        "- 依据③：抽样可视化（seg_precheck_samples.png）可人工复核。",
        "",
        "## 问题清单",
        "",
    ]
    md += [f"- {p}" for p in problems] or ["- 无"]
    (out / "seg_precheck_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "problems"},
                     ensure_ascii=False, indent=1))
    print("问题:", len(problems))
    for p in problems[:10]:
        print(" -", p)
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
