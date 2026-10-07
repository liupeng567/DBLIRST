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

import warnings

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
               top_ratio: float = 0.05, min_valid_frac: float = 0.01,
               strict: bool = True) -> torch.Tensor:
    """L_recon（7.1②，返工 B-2）：L1(ŷ_B, x)，仅在背景掩码内计算。

    排除三类像素（防背景通道吸收目标/亮点虚警）：
      ① 目标框膨胀 3px（GT 框填充掩码 maxpool）；
      ② 内环反馈掩码 m_tgt；
      ③ 剩余区域残差 |x−ŷ_B| top-5%（可能尚未标注的瞬时亮点，7.1② "最后一项是关键"）。

    窒息拦截（返工 B-2，评审 2.1）：有效像素占比 < min_valid_frac 说明反馈掩码铺满
    全图、本损失已失去监督意义——旧实现静默返回 0（loss_recon=0 实录）。strict=True
    抛 RuntimeError 与 trainer 的"loss NaN 即停"同策略（坏状态立即终止定位，不烧 GPU）；
    长跑无人值守可切 strict=False（只告警属性落盘）。
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
        # `<=`：并列值（如平场残差全零）不整片剔除——top-5% 是软守卫不是精确截断
        v_flat = v_flat * (r_flat <= thr)
    valid = v_flat.view(B, T, 1, H, W)
    recon_loss.last_valid_frac = float(valid.mean())  # 窒息哨兵：≈0 即反馈掩码铺满全图
    if recon_loss.last_valid_frac < min_valid_frac:
        msg = (f"L_recon 有效像素占比 {recon_loss.last_valid_frac:.4f} < {min_valid_frac}"
               "——反馈掩码疑似铺满全图（背景通道窒息），L_recon 已失去监督意义")
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    denom = (valid.sum() * x_feat.shape[2]).clamp_min(1.0)  # 逐元素均值（含通道维）
    return ((x_feat - y_b).abs() * valid).sum() / denom


def gate_teacher_loss(alpha_map: torch.Tensor, target_box: torch.Tensor,
                      box_dilate: int = 3) -> torch.Tensor:
    """α 门控教师监督（解 ④b，方案 v1.6.1 教师掩码落地）：平衡 BCE(α, GT框膨胀掩码)。

    死锁诊断（5k 诊断，2026-10-07）：α 仅经 seg 头间接收梯度 → 永不开 → 掩码不形成
    → 背景吸收目标 → 残差无信号。本项给 α 直接梯度打开正环：α 在目标处被推高 →
    掩码形成 → 吸收停止 → 残差出信号。

    alpha_map: [B,T,1,h,w] 门控概率图（stride-2，post-sigmoid）；
    target_box: [B,T,1,2h,2w] 原生框填充掩码。
    平衡正负项——目标像素极稀少，朴素 BCE 会被背景项淹没、把 α 推成全局 0（恰是死锁）：
        L = BCE(α[gt], 1).mean() + BCE(α[bg], 0).mean()
    空帧（无目标）只计背景项。平衡性不变量：α≡0.5 时 L ≡ 2·ln2，与框大小无关。
    """
    B, T = alpha_map.shape[:2]
    box = target_box.float().flatten(0, 1)
    box = F.max_pool2d(box, 2 * box_dilate + 1, stride=1, padding=box_dilate)
    gt = F.max_pool2d(box, 2, 2).view(B, T, 1, *alpha_map.shape[-2:])
    a = alpha_map.float().clamp(1e-4, 1.0 - 1e-4)
    gt = (gt > 0.5)
    pos = a[gt]
    neg = a[~gt]
    loss_pos = (-torch.log(pos)).mean() if pos.numel() > 0 else a.new_zeros(())
    loss_neg = (-torch.log(1.0 - neg)).mean() if neg.numel() > 0 else a.new_zeros(())
    return loss_pos + loss_neg


def decouple_loss(h_t: torch.Tensor, h_b: torch.Tensor, eps: float = 1e-6,
                  min_norm: float = 1e-2, norm_weight: float = 1.0) -> torch.Tensor:
    """L_dec（7.1③，返工 B-3）：带范数下限的软正交，防双通道坍缩到同一表示。

    旧式 cos² = dot²/(n_t·n_b+eps)² 在任一状态塌缩到 0 时给出 0——恰好放过它本该
    防止的坍缩（h_T≡0 时损失为 0）。修复两点：
      ① 分母 clamp_min 设下限，"双零"不再把 cos² 抹平；
      ② 显式惩罚范数低于 min_norm 的单元（norm floor，VICReg variance 项的同族替代），
        坍缩必然产生正损失。
    值域：cos² ∈ [0,1]（Cauchy–Schwarz，clamp 仅作数值保险）+ 非负 norm floor。
    """
    h_t = h_t.float()
    h_b = h_b.float()
    dot = (h_t * h_b).sum(dim=2)                                   # [B,T,H,W]
    nt = h_t.norm(dim=2)
    nb = h_b.norm(dim=2)
    cos2 = (dot / (nt * nb).clamp_min(min_norm * min_norm)) ** 2
    cos2 = cos2.clamp(max=1.0)
    norm_floor = F.relu(min_norm - torch.minimum(nt, nb)).mean()   # 坍缩哨兵
    return cos2.mean() + norm_weight * norm_floor
