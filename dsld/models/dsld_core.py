"""DSLD 双状态液态检测模型（M3 装配：GhostNetV2 编码器 + FPN-lite 颈 + 液态核心 + 头）。

方案 3.2 表 M / 4.1 结构：
  - 编码器：GhostNetV2 截断至 stride-8，引出 S1(/2,16ch) S2(/4,24ch) S3(/8,40ch)；
  - 融合颈：FPN 式轻量上采样 S1 + up(S2) + up(S3) → 主尺度 stride-2 C₂=32；
    （with_region=True 时另出区域尺度 stride-8 C₈=64，M3 后期双尺度接入）；
  - 液态核心：DualStateLiquidCore（dual/single 消融开关，方案 10.8-a）；
  - 输出：seg logits（stride-2 → 上采样 ×2 至原生 480×640 参与损失与评测）、
    ŷ_B（特征域重构，L_recon 用）、α（临时门控图，τ/范数监控附属输出）。

时序接口与 M2 基线对齐：forward([B,T,1,H,W]) → dict(logits=[B,T,1,H,W], ...)。
编码器/颈走外部 autocast（bf16），核心强制 fp32（4.8-①）。

M3 阶段 A 返工（2026-10-07）：核心换 λ 域指数泄漏动力学（liquid_core.py 顶部
docstring），装配层新增 s_max/kappa_max/m_scale/state_dependent 与掩码
m_max/softness 透传；τ 区间改不相交 [24,192]/[2,8]；旧 ckpt 不兼容。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsld.models.encoder_ghostnetv2 import GhostNetV2Backbone
from dsld.models.liquid_core import DualStateLiquidCore


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, c)


class FpnLite(nn.Module):
    """FPN 式轻量融合颈（3.2）：S3→P8(1×1) → ×2 融 S2 → ×2 融 S1 → 3×3 smooth。

    with_region=True 输出 (main[stride-2,C₂], region[stride-8,C₈])；否则仅 main。
    归一化 GN(8)+SiLU（方案 3.2 颈规范）。
    """

    def __init__(self, c1: int = 16, c2: int = 24, c3: int = 40,
                 c_main: int = 32, c_region: int = 64, with_region: bool = False):
        super().__init__()
        self.with_region = with_region
        self.lat3 = nn.Sequential(nn.Conv2d(c3, c_region, 1, bias=False), _gn(c_region), nn.SiLU())
        self.lat2 = nn.Sequential(nn.Conv2d(c2, c_region, 1, bias=False), _gn(c_region), nn.SiLU())
        self.smooth4 = nn.Sequential(nn.Conv2d(c_region, c_region, 3, padding=1, bias=False),
                                     _gn(c_region), nn.SiLU())
        self.red4 = nn.Sequential(nn.Conv2d(c_region, c_main, 1, bias=False), _gn(c_main), nn.SiLU())
        self.lat1 = nn.Sequential(nn.Conv2d(c1, c_main, 1, bias=False), _gn(c_main), nn.SiLU())
        self.smooth2 = nn.Sequential(nn.Conv2d(c_main, c_main, 3, padding=1, bias=False),
                                     _gn(c_main), nn.SiLU())

    def forward(self, s1, s2, s3):
        p8 = self.lat3(s3)
        p4 = self.smooth4(F.interpolate(p8, scale_factor=2.0, mode="bilinear",
                                        align_corners=False) + self.lat2(s2))
        p2 = self.smooth2(F.interpolate(self.red4(p4), scale_factor=2.0, mode="bilinear",
                                        align_corners=False) + self.lat1(s1))
        return (p2, p8) if self.with_region else (p2, None)


class DsldCore(nn.Module):
    """DSLD 核心模型（M3）。forward([B,T,1,H,W]) → dict：

    logits [B,T,1,H,W]（原生分辨率）、y_b [B,T,C,·,·]（stride-2 特征域 ŷ_B）、
    alpha [B,T,1,h,w]（stride-2 临时门控图）、h_t/h_b（解耦正则用）、
    m_tgt（内环反馈掩码，L_recon 排除用）、norms（隐状态范数）。
    """

    def __init__(
        self,
        in_ch: int = 1,
        width: float = 1.0,
        c_main: int = 32,
        c_h: int = 32,
        liquid_mode: str = "dual",
        tau_b: tuple[float, float, float] = (24.0, 192.0, 48.0),
        tau_t: tuple[float, float, float] = (2.0, 8.0, 6.0),
        mask_radius: int = 5,
        mask_decay: float = 0.9,
        alpha_th: float = 0.6,
        mask_m_max: float = 0.8,
        mask_softness: float = 0.0,
        detach_every: int = 0,
        use_checkpoint: bool = False,
        s_max: float = 0.7,
        kappa_max: float = 0.5,
        m_scale: float = 0.5,
        state_dependent: bool = False,
        mask_source: str = "gate",   # gate|teacher（④a 机制阳性对照）
        bg_mode: str = "learned",    # learned|ema（B1 非学习 EMA 背景对照臂）
        ema_momentum: float = 0.9,
        out_scale: int = 2,  # logits 上采样倍率（stride-2 → 原生）
    ):
        super().__init__()
        self.encoder = GhostNetV2Backbone(in_ch=in_ch, width=width)
        c1, c2, c3 = self.encoder.out_channels
        self.neck = FpnLite(c1, c2, c3, c_main, with_region=False)
        self.core = DualStateLiquidCore(
            c_in=c_main, c_h=c_h, mode=liquid_mode, tau_b=tau_b, tau_t=tau_t,
            mask_radius=mask_radius, mask_decay=mask_decay, alpha_th=alpha_th,
            mask_m_max=mask_m_max, mask_softness=mask_softness,
            detach_every=detach_every, use_checkpoint=use_checkpoint,
            s_max=s_max, kappa_max=kappa_max, m_scale=m_scale,
            state_dependent=state_dependent,
            mask_source=mask_source, bg_mode=bg_mode, ema_momentum=ema_momentum,
        )
        self.out_scale = out_scale
        self.liquid_mode = liquid_mode

    def forward(self, x: torch.Tensor, quality: torch.Tensor | None = None,
                teacher_mask: torch.Tensor | None = None) -> dict:
        B, T = x.shape[:2]
        frames = x.flatten(0, 1)                       # [B·T,1,H,W]
        s1, s2, s3 = self.encoder(frames)
        main, _ = self.neck(s1, s2, s3)                # [B·T,C,240,320]
        main = main.view(B, T, *main.shape[1:])
        if teacher_mask is not None and self.core.mask_source == "teacher":
            # ④a：GT 掩码膨胀 3px（与 L_recon 排除同几何）→ maxpool 下采样到 stride-2
            tm = teacher_mask.flatten(0, 1).float()
            tm = F.max_pool2d(tm, 2 * 3 + 1, stride=1, padding=3)
            tm = F.max_pool2d(tm, 2, 2)
            core_teacher = tm.view(B, T, 1, *tm.shape[-2:])
        else:
            core_teacher = None  # gate 模式忽略教师（语义与历史一致）
        out = self.core(main, quality=quality, teacher_mask=core_teacher)
        out["x_main"] = main  # 核心输入特征（L_recon 的重构目标 x）
        h, w = out["logits"].shape[-2:]
        out["logits"] = F.interpolate(
            out["logits"].flatten(0, 1), scale_factor=self.out_scale,
            mode="bilinear", align_corners=False).view(B, T, 1, h * self.out_scale,
                                                       w * self.out_scale)
        out["norms"] = dict(self.core.last_norms)
        return out

    def tau_report(self) -> dict[str, float]:
        return self.core.tau_report()
