"""掩码 → 预测框（M2 eval 全链第一环）。

基线输出为分割 logits（MSHNet final / T-MSD3D 逐帧），评测主口径是 IoU≥0.5
框级匹配（8.1/10.9），故推理期将概率图按阈值二值化后做 8 连通域分析：
  - 框 = 连通域紧致外接框（含端点像素口径，与 ITTD VOC XML 标注一致）；
  - score = 域内概率均值（置信度，供 AP 排序）；
  - min_area 过滤单像素噪声响应（默认 4 px²，1×4 及以上目标保留）。

GT 框同为含端点整数，IoU 计算统一 +1 口径（map_iou.py）。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Box:
    frame: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    track_id: int = -1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> int:
        return (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)


def iou_inclusive(a: Box | tuple, b: Box | tuple) -> float:
    """含端点整数框 IoU（+1 口径）。元组输入 = (x1,y1,x2,y2)。"""
    ax1, ay1, ax2, ay2 = (a.x1, a.y1, a.x2, a.y2) if isinstance(a, Box) else a
    bx1, by1, bx2, by2 = (b.x1, b.y1, b.x2, b.y2) if isinstance(b, Box) else b
    iw = min(ax2, bx2) - max(ax1, bx1) + 1
    ih = min(ay2, by2) - max(ay1, by1) + 1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return inter / float(area_a + area_b - inter)


def mask_to_boxes(
    prob: np.ndarray,
    frame: int,
    thr: float = 0.5,
    min_area: int = 4,
) -> list[Box]:
    """单帧概率图 → 预测框列表（score = 连通域内概率均值）。"""
    binary = (prob >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[Box] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp = prob[y : y + h, x : x + w][lab[y : y + h, x : x + w] == i]
        boxes.append(
            Box(frame=frame, x1=int(x), y1=int(y), x2=int(x + w - 1), y2=int(y + h - 1),
                score=float(comp.mean()))
        )
    return boxes
