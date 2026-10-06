"""ITTD 窗口数据集集成测试（真实缓存 seq_0001；M1 口径一致性检查）。"""

import numpy as np
import pytest

from dsld.data.ittd_window_dataset import IttdWindows

MANIFEST = "data/manifests/ittd_split_v3.json"
CACHE = "data/cache/ittd"


def test_frame_mode_shapes_and_mask_alignment():
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", mode="frame", T=1,
                     augment=False, seed=0)
    assert len(ds) == 16000  # manifest v3 train-int 16k 帧
    item = ds[0]
    assert item["windows"].shape == (1, 1, 480, 640)
    assert item["target"].shape == (1, 1, 480, 640)
    x = item["windows"]
    assert x.min() >= 0.0 and x.max() <= 1.0  # 归一化域 [0,1]
    m = item["target"]
    assert set(np.unique(m)) <= {0.0, 1.0}    # seq_0001 帧 5（train-int 首段首个含框帧，无双叠）掩码像素 = 框面积（含端点）
    lab = np.load(f"{CACHE}/seq_0001/labels.npz")
    boxes = lab["boxes"]
    f5 = boxes[boxes[:, 0] == 5]
    assert len(f5) == 1
    b = f5[0]
    expect = (int(b[3]) - int(b[1]) + 1) * (int(b[4]) - int(b[2]) + 1)
    item = ds[4]  # idx = 帧号 − 1（首段偏移 0）
    assert int(item["target"].sum()) == expect


def test_window_mode_T32():
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", mode="window", T=32,
                     stride=8, crop=(240, 320), augment=False, seed=0)
    assert len(ds) == 64 * ((250 - 32) // 8 + 1)  # 64 段 × 28 窗
    item = ds[100]
    assert item["windows"].shape == (32, 1, 240, 320)
    assert item["target"].shape == (32, 1, 240, 320)
    assert item["windows"].dtype == np.float32


def test_deterministic_per_item():
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", mode="window", T=32,
                     stride=8, augment=True, seed=0)
    a = ds[7]
    b = ds[7]
    assert np.array_equal(a["windows"], b["windows"])
    assert np.array_equal(a["target"], b["target"])


def test_empty_frames_are_legal_negatives():
    """seq_0001 有 4 个空标注帧 → 对应样本掩码全 0 且仍可用（10.9 约定）。"""
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", mode="frame", augment=False)
    lab = np.load(f"{CACHE}/seq_0001/labels.npz")
    frames_with_boxes = set(lab["boxes"][:, 0].tolist())
    empty = [f for f in range(1, 251) if f not in frames_with_boxes]
    assert empty, "seq_0001 应含空标注帧"
    idx = empty[0] - 1  # seq_0001 是 train-int 第一段，idx = 帧号−1
    item = ds[idx]
    assert item["target"].sum() == 0
    assert item["windows"].shape[-2:] == (480, 640)
