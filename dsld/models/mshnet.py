"""MSHNet 单帧基线（M2 忠实复现）。

来源：CVPR 2024《Infrared Small Target Detection with Scale and Location
Sensitivity》（arXiv:2403.19366，官方实现 github.com/ying-fu/MSHNet，
model/MSHNet.py 逐块移植）：
  - CBAM 注意力 ResNet 基本块（conv-bn-relu-conv-bn + CA·SA + shortcut）；
  - 五级 U 型编解码，通道 [16,32,64,128,256]，块数 [1,2,2,2,2]（含 middle）；
  - 多尺度头 output_0..3（1×1 conv）+ final 3×3 conv 融合四尺度上采样 logits；
  - warm_flag 训练期深监督（官方 main.py：epoch > warm_epoch 后启用辅助头）。

移植差异（仅两处，均为 ITTD 适配，其余逐行对齐官方）：
  ① input_channels 参数化（官方 3 通道 RGB → 本仓 1 通道 IR，conv_init 1×1 平凡支持）；
  ② 官方 forward 返回 (aux_masks, output) 二元组语义保持，另提供
     forward_logits(x) 推理便捷入口（返回 final 融合 logits）。
损失 SLSIoULoss 见 dsld/train/losses.py（同样逐行移植）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_planes // ratio, 1)  # 官方 ratio=16；宽度缩放网络保证 ≥1 隐通道
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class ResNet(nn.Module):
    """官方同名基本块：双层 3×3 + BN + CBAM + shortcut（官方文件命名保持）。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        return self.relu(out)


class MSHNet(nn.Module):
    """官方结构移植：param_channels [16,32,64,128,256]，param_blocks [2,2,2,2]。"""

    def __init__(self, input_channels: int = 1, block=ResNet):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels: int, out_channels: int, block, block_num: int = 1):
        layer = [block(in_channels, out_channels)]
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag: bool):
        """官方签名。warm_flag=True 返回 ([mask0..3], output) 供深监督；False 仅 output。"""
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(
                torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1)
            )
            return [mask0, mask1, mask2, mask3], output
        return [], self.output_0(x_d0)

    def forward_logits(self, x) -> torch.Tensor:
        """推理便捷入口：final 融合 logits（官方 test 路径同口径，tag=True）。"""
        return self.forward(x, warm_flag=True)[1]


class MSHNetBaseline(nn.Module):
    """训练器接口包装：输入 [B,1,H,W]（T=1 展开批次维）。"""

    def __init__(self, in_ch: int = 1, **_):
        super().__init__()
        self.net = MSHNet(input_channels=in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # [B,T,1,H,W] → 单帧模式取 T=1
            assert x.shape[1] == 1, "MSHNet 单帧基线要求 T=1"
            x = x[:, 0]
        return self.net.forward_logits(x)

    def forward_train(self, x: torch.Tensor, warm_flag: bool):
        if x.dim() == 5:
            x = x[:, 0]
        return self.net(x, warm_flag)
