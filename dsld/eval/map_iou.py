"""框级评测主口径（方案 8.1 / 10.9：IoU≥0.5 框级 P/R/F1 + AP50/mAP@[.5:.95]）。

实现约定（dsld/eval/map_iou.py，方案指定模块名）：
  - 匹配：逐帧、置信度降序、IoU≥thr 贪心（每 GT 至多配一次）——VOC/COCO 单类检测口径；
  - AP：COCO 式全点插值（101 个召回率栅格上取右侧最大精度均值），
    mAP = IoU 0.50:0.05:0.95 十档均值；pycocotools 交叉校验（tests/test_map_iou.py）；
  - 中心命中（辅助诊断口径，8.1）：预测框中心落入 GT 框内（含边界），
    与官方评分"框内有检测点"同判据，双口径差异 = 定位质量诊断信号；
  - F_a 两口径：F_a_frm = FP/帧数；F_a_pix = FP/(帧数×640×480)（×10⁻⁶ 主口径）。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dsld.eval.mask_to_boxes import Box, iou_inclusive

IOU_THRS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95
IMG_W, IMG_H = 640, 480  # F_a_pix 口径（ITTD 原生分辨率）


def _match_frame(
    preds: list[Box], gts: list[tuple[int, int, int, int]], thr: float
) -> tuple[list[bool], list[bool]]:
    """单帧贪心匹配。返回 (pred 是否 TP 标记, gt 是否被配对标记)。"""
    flags = [False] * len(preds)
    gt_used = [False] * len(gts)
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    for i in order:
        best, best_iou = -1, thr
        for g, gt in enumerate(gts):
            if gt_used[g]:
                continue
            v = iou_inclusive(preds[i], gt)
            if v >= best_iou:
                best, best_iou = g, v
        if best >= 0:
            flags[i] = True
            gt_used[best] = True
    return flags, gt_used


def _center_hit_match(
    preds: list[Box], gts: list[tuple[int, int, int, int]]
) -> tuple[list[bool], list[bool]]:
    """中心命中口径贪心：pred 中心落入 GT 框内即候选，分数序每 GT 至多一次。"""
    flags = [False] * len(preds)
    gt_used = [False] * len(gts)
    for i in sorted(range(len(preds)), key=lambda i: -preds[i].score):
        best = -1
        for g, gt in enumerate(gts):
            if gt_used[g]:
                continue
            if gt[0] <= preds[i].cx <= gt[2] and gt[1] <= preds[i].cy <= gt[3]:
                best = g
                break
        if best >= 0:
            flags[i] = True
            gt_used[best] = True
    return flags, gt_used


def _group_by_frame(preds: list[Box], gts: list[Box]):
    preds_by = defaultdict(list)
    gts_by = defaultdict(list)
    for p in preds:
        preds_by[p.frame].append(p)
    for g in gts:
        gts_by[g.frame].append((g.x1, g.y1, g.x2, g.y2))
    return preds_by, gts_by


def evaluate_boxes(
    preds: list[Box],
    gts: list[Box],
    n_frames: int,
    iou_thr: float = 0.5,
    img_wh: tuple[int, int] = (IMG_W, IMG_H),
) -> dict:
    """固定 IoU 阈值的 P/R/F1 + 中心命中 + F_a 双口径。"""
    preds_by, gts_by = _group_by_frame(preds, gts)
    tp = fp = fn = 0
    hit = 0
    for f in set(preds_by) | set(gts_by):
        flags, gt_used = _match_frame(preds_by.get(f, []), gts_by.get(f, []), iou_thr)
        tp += sum(flags)
        fp += len(flags) - sum(flags)
        fn += len(gt_used) - sum(gt_used)
        hflags, _ = _center_hit_match(preds_by.get(f, []), gts_by.get(f, []))
        hit += sum(hflags)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n_gt = len(gts)
    w, h = img_wh
    return {
        "iou_thr": iou_thr,
        "n_pred": len(preds), "n_gt": n_gt,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "center_hit_rate": round(hit / n_gt, 6) if n_gt else 0.0,
        "fa_frm": round(fp / n_frames, 6) if n_frames else 0.0,
        "fa_pix_e6": round(fp / (n_frames * w * h) * 1e6, 4) if n_frames else 0.0,
    }


def _ap_from_flags(tp_flags: list[bool], fp_flags: list[bool], n_gt: int) -> float:
    """COCO 式全点插值 AP（101 召回栅格）。"""
    if n_gt == 0:
        return 0.0
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(fp_flags)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    # 右侧最大精度包络（COCO 口径）
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    recall_pts = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, recall_pts, side="left")
    ap = 0.0
    for r_i, i in enumerate(recall_pts):
        j = idx[r_i]
        ap += precision[j] if j < len(precision) else 0.0
    return float(ap / 101.0)


def evaluate_ap(
    preds: list[Box], gts: list[Box], iou_thrs: list[float] | None = None
) -> dict:
    """AP50 与 mAP@[.5:.95]（COCO 式全点插值）。"""
    iou_thrs = iou_thrs or IOU_THRS
    preds_by, gts_by = _group_by_frame(preds, gts)
    for f in preds_by:  # 帧内按分数降序（匹配与全局排序共用该序）
        preds_by[f].sort(key=lambda b: -b.score)
    global_order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    pos = {}  # 全局序 → 帧内名次
    counters: dict[int, int] = defaultdict(int)
    for i in global_order:
        f = preds[i].frame
        pos[i] = counters[f]
        counters[f] += 1

    aps = {}
    for thr in iou_thrs:
        flags_cache = {
            f: _match_frame(pf, gts_by.get(f, []), thr)[0]
            for f, pf in preds_by.items()
        }
        tp_sorted = [flags_cache[preds[i].frame][pos[i]] for i in global_order]
        fp_sorted = [not t for t in tp_sorted]
        aps[thr] = _ap_from_flags(tp_sorted, fp_sorted, len(gts))
    ap50 = aps.get(0.5, 0.0)
    mmap = float(np.mean([aps[t] for t in iou_thrs]))
    return {
        "ap50": round(ap50, 6),
        "map_5095": round(mmap, 6),
        "ap_per_thr": {str(t): round(v, 6) for t, v in aps.items()},
    }
