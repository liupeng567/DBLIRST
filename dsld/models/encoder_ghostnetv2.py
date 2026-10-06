"""GhostNetV2 骨干（方案 3.2 编码器；官方结构移植 + 截断至 stride-8 + 单通道 stem）。

来源：GhostNetV2（Tang et al., NeurIPS 2022, openreview vhKaBdOOobB）。
PyTorch 移植底本 github.com/likyoo/GhostNetV2-PyTorch（自 MindSpore 官方实现改写），
模块逐块保留：GhostModule / GhostModuleMul（DFC 长程注意力门）/ SE / GhostBottleneck。

本文件改动（ITTD 适配必需范围）：
  ① stem 输入通道 1（LWIR 单通道），kernel 3×3 / stride 2 / 16ch 不变；
  ② 截断至 stride-8（stage3 末，40ch），弃用 stage4+ 与分类头——小目标必须保
     stride-2 高分辨率（方案 10.2）；
  ③ 引出三处特征 S1/S2/S3：/2·16ch（stage1 末）、/4·24ch（stage2 末）、
     /8·40ch（stage3 末），与方案 3.2 表 M 完全一致；
  ④ DFC（GhostModuleMul）启用规则保持官方 layer_id>1：前两个 bottleneck（stem 后
     与 stage2 首块）用原始 Ghost 模块，即 DFC 自第 3 个 bottleneck（stride-4 块内
     深处）起启用——对应方案"stride≥4 启用、stride-2 关闭"（偏差：官方首个
     stride-4 块仍为原始模块，忠实官方口径，v1.3 允许）；
  ⑤ 宽度倍率 multiplier ∈ {0.8, 1.0, 1.3}，_make_divisible(divisor=4) 官方口径；
  ⑥ 骨干保留官方 BN（方案 3.2：每 GPU 前向 4 窗 × 32 帧 = 128 帧，统计充足）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(x, divisor: int = 4) -> int:
    return int(np.ceil(x * 1.0 / divisor) * divisor)


class ConvUnit(nn.Module):
    """conv-bn(-act) 封装（官方实现原样保留，act_type 仅 relu/hswish 两用）。"""

    def __init__(self, num_in, num_out, kernel_size: int = 1, stride: int = 1,
                 padding: int = 0, num_groups: int = 1, use_act: bool = True,
                 act_type: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size, stride, padding,
                              groups=num_groups, bias=False, padding_mode="zeros")
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act: nn.Module | None
        if use_act:
            self.act = nn.Hardswish() if act_type == "hswish" else nn.ReLU()
        else:
            self.act = None

    def forward(self, x):
        out = self.bn(self.conv(x))
        return self.act(out) if self.use_act else out


class GhostModule(nn.Module):
    """Ghost 模块：主卷积 + 廉价深度卷积拼合（官方原样）。"""

    def __init__(self, num_in, num_out, kernel_size: int = 1, stride: int = 1,
                 ratio: int = 2, dw_size: int = 3, use_act: bool = True,
                 act_type: str = "relu"):
        super().__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size, stride,
                                     kernel_size // 2, 1, use_act, act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, dw_size, 1,
                                        dw_size // 2, init_channels, use_act, act_type)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostModuleMul(nn.Module):
    """GhostModule + DFC 长程注意力门（官方短卷积版：avgpool↓2 → 1×k/k×1 → sigmoid ↑2）。

    即 GhostNetV2 的 DFC attention 的省内存实现（下采样空间上估计注意力再上采样），
    数值语义与官方 MindSpore 实现一致。
    """

    def __init__(self, num_in, num_out, kernel_size: int = 1, stride: int = 1,
                 ratio: int = 2, dw_size: int = 3, use_act: bool = True,
                 act_type: str = "relu"):
        super().__init__()
        self.avgpool2d = nn.AvgPool2d(2, 2)
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size, stride,
                                     kernel_size // 2, 1, use_act, act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, dw_size, 1,
                                        dw_size // 2, init_channels, use_act, act_type)
        self.short_conv = nn.Sequential(
            ConvUnit(num_in, num_out, kernel_size, stride, kernel_size // 2, 1, False),
            ConvUnit(num_out, num_out, (1, 5), 1, (0, 2), num_out, False),
            ConvUnit(num_out, num_out, (5, 1), 1, (2, 0), num_out, False),
        )

    def forward(self, x):
        res = self.avgpool2d(x)
        res = self.short_conv(res)
        res = torch.sigmoid(res)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out * F.interpolate(res, size=out.shape[-2:], mode="bilinear",
                                   align_corners=True)


class SE(nn.Module):
    """Squeeze-Excitation（官方原样，hsigmoid 门）。"""

    def __init__(self, num_out, ratio: int = 4):
        super().__init__()
        num_mid = _make_divisible(num_out // ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(num_out, num_mid, 1, bias=True)
        self.conv_expand = nn.Conv2d(num_mid, num_out, 1, bias=True)

    def forward(self, x):
        out = self.pool(x)
        out = torch.relu(self.conv_reduce(out))
        out = F.relu6(self.conv_expand(out) + 3.0) * 0.16666667
        return x * out


class GhostBottleneck(nn.Module):
    """Ghost bottleneck（官方原样；layer_id≤1 用原始 Ghost 模块=DFC 关闭）。"""

    def __init__(self, num_in, num_mid, num_out, kernel_size: int, stride: int = 1,
                 act_type: str = "relu", use_se: bool = False, layer_id: int = 0):
        super().__init__()
        ghost1_cls = GhostModule if layer_id <= 1 else GhostModuleMul
        self.ghost1 = ghost1_cls(num_in, num_mid, 1, 1, act_type=act_type)
        self.use_dw = stride > 1
        self.dw = None
        if self.use_dw:
            pad = {3: 1, 5: 2, 7: 3}[kernel_size]
            self.dw = ConvUnit(num_mid, num_mid, kernel_size, stride, pad,
                               num_mid, False, act_type)
        self.use_se = use_se
        if use_se:
            self.se = SE(num_mid)
        self.ghost2 = GhostModule(num_mid, num_out, 1, 1, use_act=False)
        self.shortcut = None
        if num_in != num_out or stride != 1:
            pad = {3: 1, 5: 2, 7: 3}[kernel_size]
            self.shortcut = nn.Sequential(
                ConvUnit(num_in, num_in, kernel_size, stride, pad, num_in, False),
                ConvUnit(num_in, num_out, 1, 1, 0, 1, False),
            )

    def forward(self, x):
        shortcut = x
        out = self.ghost1(x)
        if self.use_dw:
            out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        return shortcut + out


# 官方 1.0× 配置表截断至 stride-8（stage1–3；k, exp, c, se, act, s）
GHOSTNETV2_CFGS_STRIDE8 = [
    # stage1（stride-2）
    [3, 16, 16, False, "relu", 1],
    # stage2（首块 s2 → stride-4）
    [3, 48, 24, False, "relu", 2],
    [3, 72, 24, False, "relu", 1],
    # stage3（首块 s2 → stride-8）
    [5, 72, 40, True, "relu", 2],
    [5, 120, 40, True, "relu", 1],
]


class GhostNetV2Backbone(nn.Module):
    """GhostNetV2 截断骨干：输入 [B,1,H,W] → (S1[/2,16ch], S2[/4,24ch], S3[/8,40ch])。

    width=1.0 时 S1/S2/S3 = (16, 24, 40)（方案 3.2 表 M 原值）；0.8×→(16,20,32)、
    1.3×→(24,32,52)（_make_divisible div=4）。
    """

    def __init__(self, in_ch: int = 1, width: float = 1.0):
        super().__init__()
        c_stem = _make_divisible(width * 16)
        self.conv_stem = nn.Conv2d(in_ch, c_stem, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_stem)
        self.act1 = nn.ReLU()

        blocks = []
        in_ch_blk = c_stem
        for layer_id, (k, exp, c, se, act, s) in enumerate(GHOSTNETV2_CFGS_STRIDE8):
            c_mid, c_out = _make_divisible(width * exp), _make_divisible(width * c)
            blocks.append(GhostBottleneck(in_ch_blk, c_mid, c_out, k, s, act, se, layer_id))
            in_ch_blk = c_out
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = (c_stem, _make_divisible(width * 24), _make_divisible(width * 40))
        self._initialize_weights()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        taps = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in (0, 2, 4):  # stage1 末 / stage2 末 / stage3 末
                taps.append(x)
        s1, s2, s3 = taps
        return s1, s2, s3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, std=np.sqrt(2.0 / n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
