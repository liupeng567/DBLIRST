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
import torch.nn.functional as F


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


# ---- M3 DSLD 损失分量（方案 7.1 ①②③；④⑤⑥ 随 M4/M5 接入） --------------------

def focal_dice_loss(pred_log: torch.Tensor, target: torch.Tensor,
                    alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """L_seg（7.1①）：Focal(α=0.75, γ=2，极端前景不平衡) + soft-Dice 各半。

    pred_log: [N,1,H,W] logits；target: [N,1,H,W] 框填充掩码。空帧（全背景）Dice
    项约定为 0（无前景无监督）、Focal 的负例项正常参与（背景学习同样重要）。
    """
    pred_log = pred_log.float()
    target = target.float()
    pred = torch.sigmoid(pred_log)
    # Focal（逐像素，p_t 形式）
    bce = F.binary_cross_entropy_with_logits(pred_log, target, reduction="none")
    p_t = target * pred + (1 - target) * (1 - pred)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal = (alpha_t * (1 - p_t) ** gamma * bce).mean()
    # soft-Dice（逐样本）
    inter = (pred * target).sum(dim=(1, 2, 3))
    card = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    nonempty = card > 0
    dice = torch.zeros_like(card)
    dice[nonempty] = 1 - (2 * inter[nonempty] + 1.0) / (card[nonempty] + 1.0)
    return focal * 0.5 + dice.mean() * 0.5


def recon_loss(y_b: torch.Tensor, x_feat: torch.Tensor, m_tgt: torch.Tensor,
               target_box: torch.Tensor, box_dilate: int = 3,
               top_ratio: float = 0.05) -> torch.Tensor:
    """L_recon（7.1②）：L1(ŷ_B, x)，仅在背景掩码内计算。

    排除三类像素（防背景通道吸收目标/亮点虚警）：
      ① 目标框膨胀 3px（GT 框填充掩码 maxpool）；
      ② 内环反馈掩码 m_tgt；
      ③ 剩余区域残差 |x−ŷ_B| top-5%（可能尚未标注的瞬时亮点，7.1② "最后一项是关键"）。
    x_feat/y_b/m_tgt 为 stride-2 域 [B,T,C(1),H,W]；target_box 为原生分辨率
    [B,T,1,2H,2W]，经 maxpool(2) 下采样（膨胀先在全分辨率做，保几何精度）。
    """
    y_b = y_b.float()
    x_feat = x_feat.float()
    B, T = y_b.shape[:2]
    box = target_box.float().flatten(0, 1)  # max_pool2d 仅支持 4D
    box_excl = F.max_pool2d(box, 2 * box_dilate + 1, stride=1, padding=box_dilate)
    box_excl = F.max_pool2d(box_excl, 2, 2).view(B, T, 1, *y_b.shape[-2:])
    excl = (box_excl > 0).float() + (m_tgt > 0.1).float()
    valid = (excl == 0).float()
    # ③ top-5% 残差剔除（按通道均值残差排序，仅在候选背景区内竞争名额）
    H, W = y_b.shape[-2:]
    res_mag = (x_feat - y_b).abs().mean(dim=2, keepdim=True)  # [B,T,1,H,W]
    v_flat = valid.flatten(2)                                 # [B,T,H·W]
    r_flat = res_mag.flatten(2).masked_fill(v_flat == 0, float("inf"))
    k = int(v_flat.shape[-1] * top_ratio)
    if k > 0:
        thr = r_flat.topk(k, dim=-1).values[..., -1:]  # [B,T,1]
        v_flat = v_flat * (r_flat < thr)
    valid = v_flat.view(B, T, 1, H, W)
    denom = (valid.sum() * x_feat.shape[2]).clamp_min(1.0)  # 逐元素均值（含通道维）
    return ((x_feat - y_b).abs() * valid).sum() / denom


def decouple_loss(h_t: torch.Tensor, h_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L_dec（7.1③）：逐帧逐位置 cos²(h̄_T, h̄_B)，防双通道坍缩到同一表示。

    h̄ 取通道隐状态本身（C 维向量）。范围 [0,1]，正交 → 0。
    """
    h_t = h_t.float()
    h_b = h_b.float()
    dot = (h_t * h_b).sum(dim=2)                                   # [B,T,H,W]
    nt = h_t.norm(dim=2)
    nb = h_b.norm(dim=2)
    cos2 = (dot / (nt * nb + eps)) ** 2
    return cos2.mean()
