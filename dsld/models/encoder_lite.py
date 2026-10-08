"""EncoderLite：stride-2 单尺度轻量编码器（M3 v2.0 方案 §4.2）。

结构（§4.2 原文）：stem 3×3/2（1ch→16）→ 2 × DW 可分离块（16→24）→ GN(8)+SiLU。
裁定：**GhostNetV2 不在 M3 重建**——骨干与核心解耦是总方案 10.2 原则，M3 验收对象是
核心动力学，编码器只需提供够用的时空特征；M4/M6 换骨干时核心零改动。

接口约定（换骨干的唯一契约）：输入 [N, in_ch, H, W] 原始归一化帧（N = B·T，时间维
共享权重、每帧前向一次），输出 [N, out_ch, H/2, W/2] 主尺度特征。核心吃
`feats: [B,T,C,H,W]`（§4.2 明示），故时间递推在特征维进行、空间逐位置并行。

参数量：≈2×10³（§4.6 估的 0.03M 是从旧 GhostNetV2 全栈继承的量级，按 §8.4 的
out_ch=24 实际远小于它——见 P1 报告的 params 实测与文档修订建议）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    """GN(8)（§4.2 归一化规范）；groups 取 gcd 防通道数不整除时构造失败。"""
    return nn.GroupNorm(min(groups, c), c)


class _DWBlock(nn.Module):
    """深度可分离块：DW 3×3（逐通道邻域上下文）→ PW 1×1（通道混合）→ GN+SiLU。"""

    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.norm = _gn(c_out, groups)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class EncoderLite(nn.Module):
    """stride-2 轻量编码器（§4.2）。小目标禁止缩放铁律（总方案 3.4）⇒ 只到 stride-2。"""

    def __init__(self, in_ch: int = 1, mid_ch: int = 16, out_ch: int = 24,
                 groups: int = 8):
        super().__init__()
        self.stride = 2
        self.out_channels = out_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=2, padding=1, bias=False),
            _gn(mid_ch, groups), nn.SiLU())
        self.b1 = _DWBlock(mid_ch, out_ch, groups)
        self.b2 = _DWBlock(out_ch, out_ch, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b2(self.b1(self.stem(x)))
