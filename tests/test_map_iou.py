"""map_iou 单测（8.2 合成用例思路延伸到框级口径 + pycocotools 交叉校验）。"""

import numpy as np
import pytest

from dsld.eval.map_iou import evaluate_ap, evaluate_boxes
from dsld.eval.mask_to_boxes import Box, iou_inclusive


def mk(frame, x1, y1, x2, y2, score=1.0):
    return Box(frame=frame, x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def test_iou_inclusive_basic():
    a, b = (10, 10, 19, 19), (15, 15, 24, 24)
    inter = 5 * 5
    union = 100 + 100 - inter
    assert abs(iou_inclusive(a, b) - inter / union) < 1e-9
    assert iou_inclusive((0, 0, 9, 9), (10, 0, 19, 9)) == 0.0  # 相邻不交


def test_perfect_predictions():
    gts = [mk(1, 100, 100, 109, 109), mk(2, 200, 100, 209, 109)]
    preds = [mk(1, 100, 100, 109, 109, 0.9), mk(2, 200, 100, 209, 109, 0.8)]
    r = evaluate_boxes(preds, gts, n_frames=2)
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["center_hit_rate"] == 1.0
    ap = evaluate_ap(preds, gts)
    assert ap["ap50"] == pytest.approx(1.0)
    assert ap["map_5095"] == pytest.approx(1.0)


def test_false_alarm_and_miss():
    gts = [mk(1, 100, 100, 109, 109), mk(2, 200, 100, 209, 109)]
    preds = [
        mk(1, 100, 100, 109, 109, 0.9),        # TP
        mk(1, 400, 400, 409, 409, 0.8),        # FP（空区域）
        mk(2, 205, 105, 214, 114, 0.7),        # IoU = 25/175 < 0.5 → FP，且 GT 漏检
    ]
    r = evaluate_boxes(preds, gts, n_frames=2)
    assert r["tp"] == 1 and r["fp"] == 2 and r["fn"] == 1
    assert abs(r["precision"] - 1 / 3) < 1e-6
    assert abs(r["recall"] - 0.5) < 1e-6
    assert abs(r["fa_frm"] - 1.0) < 1e-6
    assert abs(r["fa_pix_e6"] - 2e6 / (2 * 640 * 480)) < 0.01  # FP=2


def test_center_hit_auxiliary():
    # 回归框略偏但中心仍在 GT 框内 → 中心命中 1.0，而 IoU = 15²/(20²+20²−15²) ≈ 0.39 < 0.5
    gts = [mk(1, 100, 100, 119, 119)]
    preds = [mk(1, 105, 105, 124, 124, 0.9)]
    r = evaluate_boxes(preds, gts, n_frames=1)
    assert r["center_hit_rate"] == 1.0
    assert r["tp"] == 0 and r["fp"] == 1  # IoU 口径不匹配 → 双口径差异即定位质量信号


def test_greedy_takes_higher_score_first():
    gts = [mk(1, 0, 0, 19, 19), mk(2, 0, 0, 19, 19)]
    preds = [
        mk(1, 0, 0, 19, 19, 0.9),   # TP（与 GT 完全重合，分数高）
        mk(1, 1, 1, 20, 20, 0.6),   # IoU ≈ 0.82 ≥0.5 但 GT 已被高分占配 → FP
    ]
    r = evaluate_boxes(preds, gts, n_frames=2)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1
    ap = evaluate_ap(preds, gts)
    # 全局分数序：0.9 TP（P=1, R=0.5）→ 0.6 FP（P=0.5, R=0.5）
    # 101 召回栅格中 r≤0.5 的 51 点取精度 1.0 → AP50 = 51/101
    assert ap["ap50"] == pytest.approx(51 / 101)


def _pycocotools_ap(preds, gts, n_frames, iou_thr):
    """pycocotools 交叉校验（含端点整数框 → xywh = [x, y, x2-x1+1, y2-y1+1]）。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = [{"id": f} for f in range(1, n_frames + 1)]
    anns = [
        {"id": i + 1, "image_id": g.frame, "category_id": 1, "iscrowd": 0,
         "bbox": [g.x1, g.y1, g.x2 - g.x1 + 1, g.y2 - g.y1 + 1], "area": g.area}
        for i, g in enumerate(gts)
    ]
    dets = [
        {"image_id": p.frame, "category_id": 1,
         "bbox": [p.x1, p.y1, p.x2 - p.x1 + 1, p.y2 - p.y1 + 1], "score": p.score}
        for p in preds
    ]
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1}]}
    coco.createIndex()
    if not dets:
        return 0.0
    e = COCOeval(coco, iouType="bbox")
    e.params.imgIds = list(range(1, n_frames + 1))
    e.params.iouThrs = np.array([iou_thr])
    import json

    e.cocoDt = coco.loadRes(dets)
    e.params.useCats = 1
    e.evaluate()
    e.accumulate()
    e.summarize()
    return float(e.stats[0])


def test_cross_check_with_pycocotools():
    rng = np.random.default_rng(0)
    n_frames = 20
    gts, preds = [], []
    for f in range(1, n_frames + 1):
        n = int(rng.integers(0, 4))
        for _ in range(n):
            x1, y1 = int(rng.integers(0, 500)), int(rng.integers(0, 360))
            w, h = int(rng.integers(4, 30)), int(rng.integers(4, 30))
            gts.append(mk(f, x1, y1, x1 + w, y1 + h))
            if rng.random() < 0.85:  # 85% 检出率，小抖动
                jx, jy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
                preds.append(mk(f, x1 + jx, y1 + jy, x1 + w + jx, y1 + h + jy,
                                score=float(rng.uniform(0.3, 1.0))))
        if rng.random() < 0.2:  # 偶发虚警
            x1, y1 = int(rng.integers(0, 500)), int(rng.integers(0, 360))
            preds.append(mk(f, x1, y1, x1 + 9, y1 + 9, score=float(rng.uniform(0.1, 0.5))))
    ours = evaluate_ap(preds, gts)["ap_per_thr"]["0.5"]
    ref = _pycocotools_ap(preds, gts, n_frames, 0.5)
    assert abs(ours - ref) < 0.05, f"ours={ours:.4f} pycocotools={ref:.4f}"
