"""Dry-run 占位模型（M0 临时，M3 由真实双状态液态核心替换）。

保留与正式模型一致的模块分组（encoder / liquid / gating / heads），
使 main.py 的参数量报表结构从 M0 起即可工作。参数量刻意做小。
"""

from __future__ import annotations

import torch.nn as nn


class DryRunNet(nn.Module):
    """最小可训练卷积占位：验证 main.py 启动链路 / 训练循环 / 断点保存，无物理意义。"""

    def __init__(self, in_ch: int = 1, width: int = 8):
        super().__init__()
        w = width
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.SiLU(),
            nn.Conv2d(w, w * 2, 3, padding=1, stride=2), nn.SiLU(),
        )
        self.liquid = nn.Conv2d(w * 2, w * 2, 3, padding=1)  # 占位"动力学"
        self.gating = nn.Conv2d(w * 2, w, 1)
        self.heads = nn.Conv2d(w, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,1,H,W] 时序窗 → 时间维并入批（真实模型 M3 在时间维递推，接口一致）
        import torch

        B, T = x.shape[:2]
        z = self.encoder(x.flatten(0, 1))
        z = z + self.liquid(z)
        out = self.heads(self.gating(z))
        out = torch.nn.functional.interpolate(
            out, size=x.shape[-2:], mode="bilinear", align_corners=False
        )  # 输出回全分辨率（对齐真实模型 5.3 抑制图上采样通路）
        return out.unflatten(0, (B, T))
