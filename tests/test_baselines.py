"""mask_to_boxes / tracker / SLSIoU 单测（M2 eval 全链合成用例）。"""

import numpy as np
import pytest
import torch

from dsld.eval.mask_to_boxes import mask_to_boxes
from dsld.eval.tracker import GreedyTracker
from dsld.train.losses import SLSIoULoss, location_loss, soft_iou


def _blob(prob, cx, cy, r=5, amp=0.9):
    yy, xx = np.mgrid[0:prob.shape[0], 0:prob.shape[1]]
    prob += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r * r)))


def test_mask_to_boxes_two_blobs():
    prob = np.zeros((480, 640), dtype=np.float32)
    _blob(prob, 100, 100)
    _blob(prob, 400, 300)
    boxes = mask_to_boxes(prob, frame=1, thr=0.5, min_area=4)
    assert len(boxes) == 2
    b1 = min(boxes, key=lambda b: b.x1)
    b2 = max(boxes, key=lambda b: b.x1)
    assert abs(b1.cx - 100) <= 2 and abs(b1.cy - 100) <= 2
    assert abs(b2.cx - 400) <= 2 and abs(b2.cy - 300) <= 2
    assert all(0.0 < b.score <= 1.0 for b in boxes)


def test_mask_to_boxes_empty_and_min_area():
    prob = np.zeros((480, 640), dtype=np.float32)
    assert mask_to_boxes(prob, frame=1) == []
    prob[50, 50] = 1.0  # 单像素
    assert len(mask_to_boxes(prob, frame=1, min_area=1)) == 1
    assert mask_to_boxes(prob, frame=1, min_area=4) == []


def test_tracker_straight_and_gap():
    tr = GreedyTracker(vmax=8.0, coast_max=10)
    ids = []
    for f in range(1, 11):  # 匀速直线 1 px/帧
        from dsld.eval.mask_to_boxes import Box

        d = Box(frame=f, x1=100 + f, y1=100, x2=105 + f, y2=105, score=0.9)
        tr.update([d], f)
        ids.append(d.track_id)
    assert len(set(ids)) == 1
    # 中断 5 帧（coast ≤ 10）→ 同 ID 恢复
    from dsld.eval.mask_to_boxes import Box

    d = Box(frame=16, x1=100 + 16, y1=100, x2=105 + 16, y2=105, score=0.9)
    tr.update([d], 16)
    assert d.track_id == ids[-1]


def test_tracker_reid_after_long_gap():
    tr = GreedyTracker(vmax=8.0, coast_max=10)
    from dsld.eval.mask_to_boxes import Box

    d1 = Box(frame=1, x1=100, y1=100, x2=105, y2=105, score=0.9)
    tr.update([d1], 1)
    for f in range(2, 40):
        tr.update([], f)  # 长空窗 → 航迹删除
    d2 = Box(frame=40, x1=100, y1=100, x2=105, y2=105, score=0.9)
    tr.update([d2], 40)
    assert d2.track_id != d1.track_id


def test_soft_iou_manual():
    # pred 仅在目标处为 1 → SoftIoU = 1 → loss ≈ 0
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 10:15, 10:15] = 1.0
    pred_log = torch.where(target > 0, 10.0, -10.0)
    assert soft_iou(pred_log, target).item() < 0.05
    # pred 处处 0（logits −10）：iou = 1/26 → loss ≈ 0.96
    assert abs(soft_iou(torch.full((1, 1, 32, 32), -10.0), target).item() - 25 / 26) < 0.01


def test_slsiou_empty_frame_protection():
    """空对空 = 完美（官方 smooth=0 在此 0/0 产生 NaN，本仓约定取 1）。"""
    loss_fn = SLSIoULoss()
    empty = torch.zeros(2, 1, 64, 64)
    pred_log = torch.full((2, 1, 64, 64), -1000.0)  # sigmoid 下溢 → 精确零预测
    v = loss_fn(pred_log, empty, warm_epoch=1, epoch=5)
    assert torch.isfinite(v) and v.item() < 0.05
    # 空标注但预测有响应 → 应受罚
    v2 = loss_fn(torch.full((2, 1, 64, 64), 10.0), empty, warm_epoch=1, epoch=5)
    assert v2.item() > 0.1


def test_slsiou_perfect_and_scale_term():
    loss_fn = SLSIoULoss()
    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 20:30, 20:30] = 1.0
    v = loss_fn(torch.where(target > 0, 10.0, -10.0), target, warm_epoch=1, epoch=5)
    assert v.item() < 0.1
    # 尺度差一半 → α < 1，损失高于完美
    pred_half = torch.zeros(1, 1, 64, 64)
    pred_half[0, 0, 20:25, 20:25] = 10.0
    v_half = loss_fn(pred_half, target, warm_epoch=1, epoch=5)
    assert v_half.item() > v.item()


def test_location_loss_zero_for_aligned():
    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 20:30, 20:40] = 1.0
    pred = target.clone()
    assert location_loss(pred, target).item() < 1e-6
