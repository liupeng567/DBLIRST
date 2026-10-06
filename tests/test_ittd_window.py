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


# ---- 窗口对齐（M2 修复）：合成缓存端到端，帧与掩码同步对齐 ----------------------

import json
import hashlib
from pathlib import Path

import cv2


def _make_aligned_cache(root: Path, seq_id: int = 9001, n: int = 60,
                        h: int = 64, w: int = 80):
    """合成一段"相机匀速平移"缓存：帧 t 内容 = 基准纹理平移 (0.5t, 0.25t)。

    reg.npz 按矩阵语义（M[t]: ref→第 t 帧正向）人工写死，两块 + 真实桥；
    labels 在第 5 帧放一个已知框，用于验证掩码随帧同步 warp。
    """
    rng = np.random.default_rng(3)
    base = np.zeros((h, w), np.uint8)
    for _ in range(60):
        y, x = rng.integers(0, h - 10), rng.integers(0, w - 10)
        s = int(rng.integers(6, 11))
        base[y : y + s, x : x + s] = int(rng.integers(60, 255))
    base = cv2.GaussianBlur(base, (0, 0), 3.0)  # 平滑纹理：对齐残差阈值不受重采样误差支配

    def shift(dx, dy):
        A = np.float32([[1, 0, dx], [0, 1, dy]])
        return A

    frames = np.stack(
        [cv2.warpAffine(base, shift(0.5 * t, 0.25 * t), (w, h)) for t in range(n)]
    ).astype(np.uint8)

    d = root / f"seq_{seq_id:04d}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "frames.u8.npy", frames)
    np.save(d / "nuc_field.npy", np.zeros((h, w), np.float16))
    np.save(d / "deadpix.npy", np.zeros((0, 2), np.int32))
    np.save(d / "norm_stats.npy", np.tile(np.array([[10.0, 20.0]], np.float32), (n, 1)))
    np.save(d / "quality.npy", np.ones(n, np.float32))

    M = np.zeros((n, 2, 3), np.float32)
    for t in range(n):
        dx, dy = 0.5 * t - 0.5 * (25 if t >= 25 else 0), 0.25 * t - 0.25 * (25 if t >= 25 else 0)
        M[t] = [[1, 0, dx], [0, 1, dy]]  # 块内相对各自参考的正向矩阵
    bridge = np.full((n, 2, 3), np.nan)
    bridge[25] = [[1, 0, 12.5], [0, 1, 6.25]]  # ref_0 → ref_25（内容平移累计）
    np.savez_compressed(
        d / "reg.npz", M=M, rmse=np.zeros(n, np.float32),
        failed=np.zeros(n, bool), method=np.array(["klt"] * n),
        ref_idx=np.array([0] * 25 + [25] * (n - 25), np.int32),
        bridge_M=bridge, bridge_rmse=np.zeros(n, np.float32),
    )
    boxes = np.array([[5, 30, 20, 35, 25]], np.int32)  # 帧号 1-based
    np.savez_compressed(
        d / "labels.npz", boxes=boxes, track_ids=np.array([1], np.int32),
        track_ids_index=np.array([0, 1], np.int32),
        track_offsets=np.array([0, 1], np.int64), track_frames=np.array([5], np.int32),
    )
    return base, boxes


def _make_manifest(root: Path, seq_id: int, n: int) -> Path:
    data = {
        "splits": {"train-int": {"seqs": [seq_id]}},
        "sequences": [{"seq_id": seq_id, "n_frames": n}],
    }
    ck = "md5:" + hashlib.md5(json.dumps(
        {"splits": data["splits"], "sequences": data["sequences"]},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    data["checksum"] = ck
    p = root / "manifest_test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_window_mode_aligns_frames_and_mask(tmp_path):
    """window 模式：整窗对齐锚点后，背景逐帧静止（合成平移序列应回收基准纹理），
    掩码框随同一 W warp（框中心位移 = 锚点系下的目标位置）。"""
    seq_id, n = 9001, 60
    base, _ = _make_aligned_cache(tmp_path, seq_id, n)
    mf = _make_manifest(tmp_path, seq_id, n)
    ds = IttdWindows(str(mf), str(tmp_path), split="train-int", mode="window",
                     T=32, stride=8, augment=False, seed=0)
    item = ds[0]  # 窗 [0,32)，跨块 0→25
    win = item["windows"][:, 0]  # [32,64,80]
    anchor = win[0]
    for t in range(1, 32):
        d = float(np.abs(win[t] - anchor).mean())
        assert d < 0.05, f"对齐后帧 {t} 背景残差过大: {d:.4f}（应≈静止）"
        # 反证：不对齐的原始帧差异显著
    ds_raw = IttdWindows(str(mf), str(tmp_path), split="train-int", mode="window",
                         T=32, stride=8, augment=False, align=False, seed=0)
    win_raw = ds_raw[0]["windows"][:, 0]
    assert float(np.abs(win_raw[31] - win_raw[0]).mean()) > 3 * float(
        np.abs(win[31] - win[0]).mean()), "对照失败：未对齐窗应有明显背景漂移"

    # 掩码同步：帧 idx4 的框中心（帧自身坐标 q=(32.5,22.5)）对齐后在锚点系应为
    # W_4⁻¹·q = q − d_4 = (30.5, 21.5)（dst(u)=mask(W·u)，目标锚点位置取逆映射）
    m = item["target"][:, 0]
    assert m[4].sum() > 0
    ys, xs = np.nonzero(m[4])
    cy, cx = ys.mean(), xs.mean()
    reg = dict(np.load(tmp_path / f"seq_{seq_id:04d}" / "reg.npz"))
    from dsld.data.preprocess.register import window_anchor_warps
    Ws = window_anchor_warps(reg, 0, 32)
    W4h = np.vstack([Ws[4], [0, 0, 1.0]])
    p = np.linalg.inv(W4h) @ np.array([32.5, 22.5, 1.0])  # 锚点系下的框中心
    assert abs(cx - p[0]) < 1.5 and abs(cy - p[1]) < 1.5, (
        f"掩码中心 ({cx:.1f},{cy:.1f}) 未跟随对齐位移 ({p[0]:.1f},{p[1]:.1f})")


def test_frame_mode_unaffected_by_align_flag(tmp_path):
    """frame 模式不做对齐（与 MSHNet 推理口径一致），align 参数不影响结果。"""
    seq_id, n = 9002, 40
    _make_aligned_cache(tmp_path, seq_id, n)
    mf = _make_manifest(tmp_path, seq_id, n)
    a = IttdWindows(str(mf), str(tmp_path), mode="frame", augment=False)[0]
    b = IttdWindows(str(mf), str(tmp_path), mode="frame", augment=False, align=False)[0]
    assert np.array_equal(a["windows"], b["windows"])
    assert isinstance(a["quality"], float)
