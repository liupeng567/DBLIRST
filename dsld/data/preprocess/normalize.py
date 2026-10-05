"""2.2 坏点/NUC + 2.3 鲁棒标准化。

流水线顺序（方案 2.1）：原始帧 → 坏点替换 → NUC 平场 → 逐帧 median/MAD 标准化。

缓存约定（2.8）：
  - frames.u8.npy 存原始帧；本模块的校正全部在线施加（减场 + 少量像素替换，开销可忽略）
  - nuc_field.npy = 时序中值图的 15×15 高斯低通（float16 [H,W]）
  - deadpix.npy = 坏点位置 [K,2]
  - norm_stats.npy = 每帧 (median, sigma)（校正后域，float32 [N,2]）

关键决策（方案 2.3 依据）：不用全局 min-max；σ 下限 1.0 灰度级——平坦帧 MAD=0
时防止 1/ε 放大量化噪声成盐椒噪声（ε 的工程实现取下限钳制）。
"""

from __future__ import annotations

import cv2
import numpy as np

DEAD_DIFF_GATE = 40.0     # 与 3×3 邻域中值差（灰度级）
SIGMA_FLOOR = 1.0         # σ 下限（灰度级），防平坦帧爆炸
CLIP = 8.0                # x̂ 裁剪区间 [−8, 8]


def temporal_median(frames: np.ndarray) -> np.ndarray:
    """时序中值图 M(u,v)（目标移动不污染中值）。frames: [N,H,W] uint8。

    uint8 双分区实现（避免 np.median 的 float64 中间数组，~3×）；
    N=偶数时取中间两值均值（与 np.median 一致，±0.5 灰度级内）。
    """
    n = frames.shape[0]
    lo, hi = (n - 1) // 2, n // 2
    part = np.partition(frames, (lo, hi), axis=0)
    if lo == hi:
        return part[lo].copy()
    med = (part[lo].astype(np.uint16) + part[hi]) // 2
    return med.astype(np.uint8)


def detect_dead_pixels(med: np.ndarray) -> np.ndarray:
    """孤立异常像素（与时序中值图 3×3 邻域中值差 > 40，位置固定 → 传感器坏点）。"""
    med3 = cv2.medianBlur(med, 3)
    dead = np.abs(med.astype(np.int16) - med3.astype(np.int16)) > DEAD_DIFF_GATE
    return np.argwhere(dead).astype(np.int32)  # [K,2] (y,x)


def _hist_median_u16(hist: np.ndarray) -> float:
    """直方图中位数（np.median 口径：偶数取中间两值均值）。"""
    c = np.cumsum(hist)
    total = int(c[-1])
    k1 = (total - 1) // 2  # 0-based 下中位
    k2 = total // 2
    v1 = int(np.searchsorted(c, k1 + 1))
    v2 = int(np.searchsorted(c, k2 + 1))
    return (v1 + v2) / 2.0


def nuc_field(med: np.ndarray) -> np.ndarray:
    """平场近似：lowpass(M)，15×15 高斯；在线做 x' = x − field。"""
    return cv2.GaussianBlur(med, (15, 15), 0).astype(np.float16)


def correct_frame(frame: np.ndarray, nuc: np.ndarray, dead: np.ndarray) -> np.ndarray:
    """单帧校正：减平场 + 坏点用 3×3 空间中值替换。返回 float32 [H,W]。

    nuc 传 float32（调用方对每段缓存一次转换，避免逐帧 float16→float32）。
    """
    x = frame.astype(np.float32)
    x -= nuc if nuc.dtype == np.float32 else nuc.astype(np.float32)
    if len(dead):
        ys, xs = dead[:, 0], dead[:, 1]
        pad = np.pad(x, 1, mode="edge")
        for y, x_ in zip(ys, xs):
            nb = pad[y : y + 3, x_ : x_ + 3].reshape(-1)
            x[y, x_] = np.median(nb)
    return x


_OFF = 2048  # int16 量化域平移（校正后帧幅度 ≈ ±300，余量充足）


def frame_stats(x_corr: np.ndarray) -> tuple[float, float]:
    """逐帧稳健统计 (median, σ=1.4826·MAD)，σ 下限钳制。

    int16 量化域走 bincount 精确直方图快路径（O(N)）；量化误差 ≤0.5 灰度级，
    远小于 σ（≈10），且统计与应用同域，口径统一。
    """
    if x_corr.dtype == np.int16:
        v = x_corr.ravel() + np.int16(_OFF)          # int16 域内平移（值域 ±300+2048 安全）
        med = _hist_median_u16(np.bincount(v, minlength=2 * _OFF)) - _OFF
        mad_v = np.abs(x_corr - np.int16(int(round(med)))).ravel() + np.int16(_OFF)
        mad = _hist_median_u16(np.bincount(mad_v, minlength=2 * _OFF))
        sigma = max(1.4826 * mad, SIGMA_FLOOR)
        return med, sigma
    med = float(np.median(x_corr))
    mad = float(np.median(np.abs(x_corr - med)))
    sigma = max(1.4826 * mad, SIGMA_FLOOR)
    return med, sigma


def normalize_frame(x_corr: np.ndarray, med: float, sigma: float) -> np.ndarray:
    """x̂ = clip((x−med)/σ, −8, 8)；x_n = (x̂+8)/16 ∈ [0,1]。返回 float32。"""
    x_hat = np.clip((x_corr - med) / (sigma + 1e-6), -CLIP, CLIP)
    return (x_hat + CLIP) / (2 * CLIP)
