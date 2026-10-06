"""T-MSD3D 离散时序基线（M2：多尺度帧差 + 3D conv，与 MSHNet 等参数量）。

方案 9.2-M2 定义："一个离散时序基线（多尺度帧差 + 3D conv，等参数量）"。
设计原则：与 MSHNet 逐块同构（CBAM-ResNet 块 / 五级 U 型 / 多尺度头 + final 融合），
仅三处替换——保证对比公平、参数量可对齐：

  ① 前端多尺度帧差：Δ∈{1,4,8,16} 的符号帧差 x_t − x_{t−Δ}（窗口前端补零），
     与原帧堆叠为 5 通道输入（配准后残余背景运动已被 M1 消解，帧差主要响应运动目标）；
  ② 全部卷积换 3D（k=(3,3,3)，时间维不池化、步幅 1，保持逐帧输出对齐；
     空间下采样同 MSHNet MaxPool），CBAM 注意力按帧展开为 2D（与 MSHNet 逐块同构）；
  ③ 通道宽度按 3D 卷积参数膨胀系数（×3/卷积）缩放，使总参数量 ≈ MSHNet（±5%），
     具体宽度在实例化时以 MSHNet 实测参数量校准（scripts/bench_baselines.py 输出对照）。

输出：逐帧 logits [B,T,1,H,W]；训练期 warm_flag 深监督同 MSHNet。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dsld.models.mshnet import ChannelAttention, SpatialAttention

DIFF_SCALES = (1, 4, 8, 16)  # 多尺度帧差（帧）


def multi_scale_frame_diff(x: torch.Tensor) -> torch.Tensor:
    """[B,T,1,H,W] → [B,T,5,H,W]：原帧 + Δ∈{1,4,8,16} 符号帧差（前段补零）。"""
    feats = [x]
    for d in DIFF_SCALES:
        diff = torch.zeros_like(x)
        diff[:, d:] = x[:, d:] - x[:, :-d]
        feats.append(diff)
    return torch.cat(feats, dim=2)


class ResBlock3D(nn.Module):
    """MSHNet ResNet 块的 3D 同构版：conv3d-bn-relu-conv3d-bn + 逐帧 CBAM + shortcut。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channels)
        if out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1), nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        B, C, T, H, W = out.shape
        f = out.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # 逐帧 CBAM（2D 同构）
        f = self.ca(f) * f
        f = self.sa(f) * f
        out = f.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return self.relu(out + residual)


def _up_spatial():
    return nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)


class TemporalBaseline3D(nn.Module):
    """3D U 型编解码 + 多尺度逐帧输出；channels 由参数量对齐决定（默认 ≈MSHNet）。"""

    def __init__(self, in_ch: int = 1, channels: tuple[int, ...] | None = None, **_):
        super().__init__()
        ch = list(channels or (10, 19, 38, 76, 152))  # 实测 4.196M vs MSHNet 4.066M（ratio 1.033）
        c0, c1, c2, c3, c4 = ch
        n_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.up = _up_spatial()

        self.conv_init = nn.Conv3d(1 + len(DIFF_SCALES), c0, 1, 1)

        self.encoder_0 = nn.Sequential(ResBlock3D(c0, c0))
        self.encoder_1 = nn.Sequential(ResBlock3D(c0, c1), ResBlock3D(c1, c1))
        self.encoder_2 = nn.Sequential(ResBlock3D(c1, c2), ResBlock3D(c2, c2))
        self.encoder_3 = nn.Sequential(ResBlock3D(c2, c3), ResBlock3D(c3, c3))
        self.middle_layer = nn.Sequential(ResBlock3D(c3, c4), ResBlock3D(c4, c4))

        self.decoder_3 = nn.Sequential(ResBlock3D(c3 + c4, c3), ResBlock3D(c3, c3))
        self.decoder_2 = nn.Sequential(ResBlock3D(c2 + c3, c2), ResBlock3D(c2, c2))
        self.decoder_1 = nn.Sequential(ResBlock3D(c1 + c2, c1), ResBlock3D(c1, c1))
        self.decoder_0 = nn.Sequential(ResBlock3D(c0 + c1, c0))

        self.output_0 = nn.Conv3d(c0, 1, 1)
        self.output_1 = nn.Conv3d(c1, 1, 1)
        self.output_2 = nn.Conv3d(c2, 1, 1)
        self.output_3 = nn.Conv3d(c3, 1, 1)
        self.final = nn.Conv3d(4, 1, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor, warm_flag: bool):
        """x: [B,T,1,H,W]。返回 ([aux0..3 逐帧 logits], final [B,T,1,H,W])。"""
        feats = multi_scale_frame_diff(x)  # [B,T,5,H,W]
        feats = feats.permute(0, 2, 1, 3, 4)  # → [B,5,T,H,W]

        e0 = self.encoder_0(self.conv_init(feats))
        e1 = self.encoder_1(self.pool(e0))
        e2 = self.encoder_2(self.pool(e1))
        e3 = self.encoder_3(self.pool(e2))
        m = self.middle_layer(self.pool(e3))

        d3 = self.decoder_3(torch.cat([e3, self.up(m)], 1))
        d2 = self.decoder_2(torch.cat([e2, self.up(d3)], 1))
        d1 = self.decoder_1(torch.cat([e1, self.up(d2)], 1))
        d0 = self.decoder_0(torch.cat([e0, self.up(d1)], 1))

        # 逐尺度头 → 时间维恒定，空间上采样到输入分辨率后融合（MSHNet final 同构）
        m0 = self.output_0(d0)
        m1 = self.output_1(d1)
        m2 = self.output_2(d2)
        m3 = self.output_3(d3)
        output = self.final(torch.cat([m0, self.up(m1), self.up(self.up(m2)), self.up(self.up(self.up(m3)))], dim=1))
        output = output.permute(0, 2, 1, 3, 4)  # → [B,T,1,H,W]

        if warm_flag:
            aux = [
                m0.permute(0, 2, 1, 3, 4),
                self.up(m1).permute(0, 2, 1, 3, 4),
                self.up(self.up(m2)).permute(0, 2, 1, 3, 4),
                self.up(self.up(self.up(m3))).permute(0, 2, 1, 3, 4),
            ]
            return aux, output
        return [], output

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, warm_flag=True)[1]


class TemporalBaseline(nn.Module):
    """训练器接口包装（model.type=msd3d）。"""

    def __init__(self, in_ch: int = 1, channels: tuple[int, ...] | None = None, **_):
        super().__init__()
        self.net = TemporalBaseline3D(in_ch=in_ch, channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # [B,1,H,W] → 补时间维
            x = x.unsqueeze(1)
        return self.net.forward_logits(x)

    def forward_train(self, x: torch.Tensor, warm_flag: bool):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        return self.net(x, warm_flag)
