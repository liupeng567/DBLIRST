"""normalize / encode 单元测试：坏点、NUC、标准化口径、热图峰值、RLE、轨迹索引。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.preprocess.encode import (  # noqa: E402
    build_heatmap_frame,
    build_track_index,
    rle_decode,
    rle_encode,
    track_index_to_arrays,
)
from dsld.data.preprocess.normalize import (  # noqa: E402
    correct_frame,
    detect_dead_pixels,
    frame_stats,
    normalize_frame,
    nuc_field,
    temporal_median,
)

RNG = np.random.default_rng(3)


class TestNormalize:
    def make_seq(self, n=40, h=120, w=160) -> np.ndarray:
        bg = 100 + RNG.normal(0, 5, (1, h, w))  # 缓变背景
        seq = np.repeat(bg, n, axis=0).astype(np.uint8)
        for t in range(n):  # 移动目标
            seq[t, 50 + t : 54 + t, 80 : 84] = 200
        return seq

    def test_temporal_median_ignores_moving_target(self):
        seq = self.make_seq()
        med = temporal_median(seq)
        assert med[52, 81] < 150  # 目标扫过处仍是背景值

    def test_dead_pixel_detected_and_replaced(self):
        seq = self.make_seq()
        seq[:, 30, 40] = 255  # 恒定坏点（时序不变）
        med = temporal_median(seq)
        dead = detect_dead_pixels(med)
        assert any((dead == [30, 40]).all(1)), f"坏点未检出: {dead}"
        nuc = nuc_field(med)
        x = correct_frame(seq[5], nuc, dead)
        assert abs(x[30, 40] - np.median(x[29:32, 39:42])) < 20  # 已被邻域中值替换

    def test_stats_and_normalize_range(self):
        x = RNG.normal(100, 10, (120, 160)).astype(np.float32)
        med, sigma = frame_stats(x)
        assert abs(med - np.median(x)) < 1e-6 and 9.0 < sigma < 11.0  # 1.4826·MAD≈σ
        xn = normalize_frame(x, med, sigma)
        assert xn.min() >= 0 and xn.max() <= 1
        assert abs(xn[int(np.argmin(np.abs(x - med)) // 160),
                      int(np.argmin(np.abs(x - med)) % 160)] - 0.5) < 0.01

    def test_flat_frame_is_all_050(self):
        x = np.full((60, 80), 42.0)
        med, sigma = frame_stats(x)
        xn = normalize_frame(x, med, sigma)
        assert np.allclose(xn, 0.5)  # σ 下限钳制 → 无盐椒噪声

    def test_clipped_outlier_maps_to_bounds(self):
        x = np.zeros((60, 80), np.float32)
        x[0, 0] = 1000.0
        med, sigma = frame_stats(x)
        xn = normalize_frame(x, med, sigma)
        assert xn[0, 0] == 1.0 and xn[10, 10] == 0.5


class TestEncode:
    def test_heatmap_single_peak_255(self):
        hm = build_heatmap_frame(np.array([[100, 50, 110, 60]]))
        assert hm[55, 105] == 255
        assert (hm > 128).sum() >= 1

    def test_heatmap_peak_count_equals_boxes(self):
        boxes = np.array([[10, 10, 14, 14], [200, 100, 208, 106], [400, 300, 404, 306]])
        hm = build_heatmap_frame(boxes)
        from scipy.ndimage import maximum_filter

        peaks = ((hm == maximum_filter(hm, 3)) & (hm > 100)).sum()
        assert peaks == len(boxes), f"峰值 {peaks} != 框数 {len(boxes)}"

    def test_rle_roundtrip(self):
        m = np.zeros((48, 64), np.uint8)
        m[10:14, 20:26] = 1
        m[30, 0:5] = 1
        runs = rle_encode(m)
        rec = rle_decode(runs, (48, 64))
        assert (rec == m).all()

    def test_track_index_roundtrip(self):
        boxes = np.array([[0, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [5, 0, 0, 0]])
        tids = np.array([1, 1, 1, 2])
        idx = build_track_index(boxes, tids)
        assert idx == {1: [0, 2], 2: [5]}
        ids, offsets, frames = track_index_to_arrays(idx)
        assert ids.tolist() == [1, 2]
        assert frames[offsets[0] : offsets[1]].tolist() == [0, 2]
        assert frames[offsets[1] : offsets[2]].tolist() == [5]
