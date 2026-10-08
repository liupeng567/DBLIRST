"""DSLD 核心装配（M3 v2.0 方案 §8.1：EncoderLite + DualStateLiquidCore + 头接口）。

时序接口与 M2 基线对齐：forward([B,T,1,H,W]) → dict（logits [B,T,1,h,w]，h,w = 原生/2）。

职能边界（方案对前版的两条修订）：
  - **框头移除**（属 M4）：本文件只出 stride-2 逐像素 logits，掩码→框在 eval 侧做；
  - **不做上采样**：损失与快评全部在 stride-2 域（IttdWindows 的 target/teacher_mask 就是
    stride-2），全分辨率输出属 P4 推理档，届时再随消费者加开关（L5：不留无人读的配置）。

teacher_mask 直接透传：几何（膨胀 3px + 块最大下采样）由数据侧 register 的
dilate_mask/downsample_max 一次性产出，**装配层不再二次膨胀**——"与 L_recon 剔除区同几何"
这条口径必须只有一份实现（L4：两处各写一份迟早重演坐标系/采样语义分叉）。

精度分工（§4.3.7）：编码器由外部 autocast 决定（训练用 bf16），核心内部强制 fp32。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dsld.models.encoder_lite import EncoderLite
from dsld.models.liquid_core import DualStateLiquidCore


class DsldCore(nn.Module):
    """编码器 + 双状态液态核。forward 输出键见 DualStateLiquidCore（外加 x_main）。"""

    def __init__(self, in_ch: int = 1, enc_mid: int = 16, enc_out: int = 24,
                 c_h: int = 24, state_mode: str = "dual", tau_mode: str = "normal",
                 mask_source: str = "teacher", bg_mode: str = "learned",
                 tau_b: tuple[float, float, float] = (24.0, 192.0, 48.0),
                 tau_t: tuple[float, float, float] = (2.0, 8.0, 6.0),
                 mask_decay: float = 0.9, mask_radius: int = 5, m_max: float = 0.8,
                 alpha_th: float = 0.5, ema_momentum: float = 0.9, s_max: float = 0.7,
                 beta_max: float = 1.0, state_dependent: bool = False,
                 gate_hidden: int = 16, gate_bias_init: float = -2.0,
                 use_checkpoint: bool = False):
        super().__init__()
        self.encoder = EncoderLite(in_ch=in_ch, mid_ch=enc_mid, out_ch=enc_out)
        self.core = DualStateLiquidCore(
            c_in=enc_out, c_h=c_h, state_mode=state_mode, tau_mode=tau_mode,
            mask_source=mask_source, bg_mode=bg_mode, tau_b=tau_b, tau_t=tau_t,
            mask_decay=mask_decay, mask_radius=mask_radius, m_max=m_max, alpha_th=alpha_th,
            ema_momentum=ema_momentum, s_max=s_max, beta_max=beta_max,
            state_dependent=state_dependent, gate_hidden=gate_hidden,
            gate_bias_init=gate_bias_init, use_checkpoint=use_checkpoint)
        self.stride = self.encoder.stride

    def forward(self, x: torch.Tensor, dt: torch.Tensor | None = None,
                quality: torch.Tensor | None = None,
                teacher_mask: torch.Tensor | None = None) -> dict:
        """x [B,T,1,H,W] 归一化帧 → dict。dt/quality/teacher_mask 语义见 DualStateLiquidCore。"""
        B, T = x.shape[:2]
        feats = self.encoder(x.flatten(0, 1))              # [B·T,C,h,w]（stride-2）
        feats = feats.view(B, T, *feats.shape[1:])
        out = self.core(feats, dt=dt, quality=quality, teacher_mask=teacher_mask)
        out["x_main"] = feats                              # L_recon 的重构目标 x
        out["norms"] = dict(self.core.last_norms)
        out["tau"] = self.core.tau_report()
        return out
