"""SLSIoU 损失（M2 基线；官方 loss.py 逐行移植 + 向量化 + 空帧数值保护）。

来源：CVPR 2024 MSHNet（github.com/ying-fu/MSHNet model/loss.py）：
  - SoftIoU：sigmoid(pred) 与 target 的交并比软版；
  - SLSIoULoss：warmup 后乘尺度敏感权重 α（预测/目标像素总量之比，含 dis 惩罚）
    并加位置敏感项 LLoss（质量中心归一化坐标的角度差 + 模长比）。

与官方的偏差（均在 ITTD 数据适配必需范围，其余逐行对齐）：
  ① LLoss 向量化（官方逐样本 Python 循环 → 批量算子，数值语义一致）；
  ② 空标注帧保护：官方 smooth=0.0 在 target_sum=pred_sum=0 时 0/0 产生 NaN——
     ITTD 全背景帧是合法负样本（10.9 约定），此时约定 IoU 项取 1、α 取 1
     （空对空 = 完美匹配），仅在两和同时为零分支生效，不影响含目标样本梯度。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def soft_iou(pred_log: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """官方 SoftIoULoss（1 − mean(IoU)），pred_log 为未过 sigmoid 的 logits。"""
    pred = torch.sigmoid(pred_log)
    smooth = 1
    intersection = pred * target
    inter_sum = intersection.sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    iou = (inter_sum + smooth) / (pred_sum + target_sum - inter_sum + smooth)
    return 1 - iou.mean()


def location_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """官方 LLoss 向量化：归一化坐标下质量中心的角度差 + 模长比（逐样本均值）。"""
    n, _, h, w = pred.shape
    x_index = torch.arange(w, device=pred.device, dtype=pred.dtype).view(1, 1, 1, w) / w
    y_index = torch.arange(h, device=pred.device, dtype=pred.dtype).view(1, 1, h, 1) / h
    smooth = 1e-8
    px = (x_index * pred).sum(dim=(2, 3)).sum(dim=1)  # 质量中心 x（∑x·p / hw 的逐样本和）
    py = (y_index * pred).sum(dim=(2, 3)).sum(dim=1)
    tx = (x_index * target).sum(dim=(2, 3)).sum(dim=1)
    ty = (y_index * target).sum(dim=(2, 3)).sum(dim=1)
    angle = (4 / torch.pi**2) * torch.square(
        torch.arctan(py / (px + smooth)) - torch.arctan(ty / (tx + smooth))
    )
    plen = torch.sqrt(px * px + py * py + smooth)
    tlen = torch.sqrt(tx * tx + ty * ty + smooth)
    length = torch.minimum(plen, tlen) / (torch.maximum(plen, tlen) + smooth)
    return ((1 - length + angle) / n).sum()


class SLSIoULoss(nn.Module):
    """官方 SLSIoULoss：warmup 前 1−mean(IoU)；之后 1−mean(α·IoU)+LLoss。"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_log: torch.Tensor,
        target: torch.Tensor,
        warm_epoch: int,
        epoch: int,
        with_shape: bool = True,
    ) -> torch.Tensor:
        pred_log = pred_log.float()
        target = target.float()
        pred = torch.sigmoid(pred_log)

        inter_sum = (pred * target).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))

        iou = inter_sum / (pred_sum + target_sum - inter_sum).clamp_min(1e-6)
        empty = (pred_sum + target_sum) == 0  # 空对空 = 完美（偏差②）
        iou = torch.where(empty, torch.ones_like(iou), iou)

        if epoch > warm_epoch:
            dis = torch.pow((pred_sum - target_sum) / 2, 2)
            alpha = (torch.minimum(pred_sum, target_sum) + dis) / (
                torch.maximum(pred_sum, target_sum) + dis
            ).clamp_min(1e-6)
            alpha = torch.where(empty, torch.ones_like(alpha), alpha)
            loss = 1 - (alpha * iou).mean()
            if with_shape:
                loss = loss + location_loss(pred, target)
        else:
            loss = 1 - iou.mean()
        return loss
