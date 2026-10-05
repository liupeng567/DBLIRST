"""M0-5 附加探针：干扰目标（行人/电瓶车）是否被 GT 标注——数据侧佐证。

原理：选"官方属性=有其他动目标干扰 且 无动目标"的序列（GT 只含静/缓动车辆），
对相邻帧做差分找运动区域；落在 GT 框外的显著运动区 = 未标注的移动干扰目标。

产出：reports/m0/distractor_probe.png（差分图 + GT 框叠加）、结论写入预检 md。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.ittd_parse import load_seq_annotation  # noqa: E402
from dsld.utils.vis import setup_cjk  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ITTD = Path(r"D:\Datasets\面向空地应用的红外时敏目标检测跟踪数据集")
# 官方属性表：干扰=有 且 动目标=无（白天外场 8/9，傍晚内场 42）→ GT 内应只有静目标
PROBE_SEQS = [8, 9, 42]
FRAME_PAIRS = {8: (100, 102), 9: (120, 122), 42: (100, 102)}  # 任取中段两帧


def imread_u8(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def moving_regions(img1: np.ndarray, img2: np.ndarray, gt_boxes, thr: float = 12.0):
    """帧间差分 → 高斯平滑 → 阈值 → 连通域；返回 (框, 是否在任一 GT 框内)。"""
    diff = cv2.absdiff(img1, img2).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (5, 5), 1.5)
    mask = (diff > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    H, W = mask.shape
    for c in contours:
        area = cv2.contourArea(c)
        if area < 6:  # 过滤噪声小块（<6px）
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        inside = any(
            x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in gt_boxes
        )
        out.append({"rect": (x, y, w, h), "area": area, "inside_gt": inside,
                    "center": (cx, cy)})
    return out, mask


def main() -> int:
    setup_cjk()
    import matplotlib.pyplot as plt

    n_rows = len(PROBE_SEQS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.6 * n_rows))
    summary = {}
    for row, vid in enumerate(PROBE_SEQS):
        f1, f2 = FRAME_PAIRS[vid]
        img1 = imread_u8(ITTD / "Images" / str(vid) / f"{f1:03d}.bmp")
        img2 = imread_u8(ITTD / "Images" / str(vid) / f"{f2:03d}.bmp")
        ann = load_seq_annotation(ITTD / "Annotation" / str(vid))
        gt_boxes = [tuple(int(v) for v in i.box) for i in ann.frames[f1]]
        # GT 框向外交互容差 2px，避免边缘抖动误判
        gt_pad = [(x1 - 2, y1 - 2, x2 + 2, y2 + 2) for x1, y1, x2, y2 in gt_boxes]
        regions, mask = moving_regions(img1, img2, gt_pad)
        outside = [r for r in regions if not r["inside_gt"]]
        summary[vid] = {
            "gt_boxes_at_f1": len(gt_boxes),
            "moving_regions": len(regions),
            "outside_gt": len(outside),
            "outside_detail": [
                {"center": [round(c, 1) for c in r["center"]],
                 "area_px": round(r["area"], 1)}
                for r in outside
            ],
        }

        for col, (im, title) in enumerate(
            ((img2, f"v{vid} f{f2:03d} 原图（GT 框={len(gt_boxes)}）"),
             (mask, f"v{vid} 帧间差分 {f1}->{f2}（框外运动区={len(outside)}）"))
        ):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
                for x1, y1, x2, y2 in gt_boxes:
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, edgecolor="yellow", lw=1.2))
                for r in outside:
                    x, y, w, h = r["rect"]
                    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                               edgecolor="red", lw=1.0))
            else:
                ax.imshow(im, cmap="hot")
                for x1, y1, x2, y2 in gt_pad:
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, edgecolor="cyan", lw=1.2))
                for r in outside:
                    x, y, w, h = r["rect"]
                    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                               edgecolor="lime", lw=1.0))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle(
        "干扰目标探针：官方属性[干扰=有, 动目标=无]的序列，差分运动区 vs GT 框\n"
        "（黄/青=GT框，红/绿=框外运动区=未标注移动干扰）", fontsize=11)
    fig.tight_layout()
    out = REPO / "reports" / "m0" / "distractor_probe.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    import json

    (REPO / "reports" / "m0" / "distractor_probe.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
