"""M0 dry-run 合成窗口数据集：验证训练链路，无物理意义（M1 替换为真实 ITTD 窗口）。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class DryRunWindows(Dataset):
    """生成 [T,1,H,W] 合成窗口：缓变背景 + 瞬时亮点 + 目标轨迹，目标为窗口本身（自回归式占位）。"""

    def __init__(self, n_windows: int = 64, T: int = 8, H: int = 240, W: int = 320,
                 seed: int = 0):
        self.n_windows = n_windows
        self.T, self.H, self.W = T, H, W
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict:
        rng = self.rng  # 单进程 dry-run 直接用共享 rng
        t_axis = np.arange(self.T)[:, None, None]
        bg = 0.4 + 0.1 * rng.normal() + 0.05 * np.sin(t_axis / 3 + rng.normal() * 6.28)
        win = np.repeat(bg, self.H * self.W).reshape(self.T, self.H, self.W)
        # 瞬时亮点（1-2 帧）
        for _ in range(rng.integers(1, 4)):
            f = int(rng.integers(0, self.T))
            y, x = int(rng.integers(8, self.H - 8)), int(rng.integers(8, self.W - 8))
            win[f, y - 1 : y + 2, x - 1 : x + 2] += 0.5
        win += rng.normal(0, 0.02, win.shape)
        x = torch.from_numpy(win.astype(np.float32)).unsqueeze(1)  # [T,1,H,W]
        return {"windows": x, "target": x.clone()}
