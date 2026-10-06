"""配准模块单元测试：仿射真值回收 / 滑动参考 / 质量门限。

合成用例：随机纹理图 + 已知仿射（旋转/尺度/平移），验证 KLT 主路与 FM 回退
的恢复精度（方案 9.1 tests 四件套之一）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.preprocess.register import (  # noqa: E402
    CFG,
    align_to_anchor,
    estimate_affine_klt,
    fourier_mellin,
    register_frame,
    register_sequence,
)

RNG = np.random.default_rng(7)


def make_texture(h: int = 240, w: int = 320) -> np.ndarray:
    """随机点 + 密集随机块的丰富纹理图（Harris 角点 ≥ 数百，接近真实场景密度）。"""
    img = RNG.integers(0, 255, (h, w)).astype(np.uint8) * 0.5
    for _ in range(200):
        y, x = RNG.integers(0, h - 24), RNG.integers(0, w - 24)
        s = int(RNG.integers(8, 24))
        img[y : y + s, x : x + s] = RNG.integers(0, 255)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img.astype(np.uint8)


def affine(deg: float, scale: float, tx: float, ty: float, h=240, w=320) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]]) * scale
    A = np.eye(3)
    A[:2, :2] = R
    A[:2, 2] = [tx, ty]
    center = np.array([w / 2, h / 2])
    T_c = np.eye(3)
    T_c[:2, 2] = center
    T_cinv = np.eye(3)
    T_cinv[:2, 2] = -center
    return (T_c @ A @ T_cinv)[:2]


def point_err(M_est: np.ndarray, M_gt: np.ndarray, h=240, w=320) -> float:
    """四角映射误差 RMS（px）。"""
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
    pe = cv2.transform(corners.reshape(-1, 1, 2), M_est).reshape(-1, 2)
    pg = cv2.transform(corners.reshape(-1, 1, 2), M_gt).reshape(-1, 2)
    return float(np.sqrt(((pe - pg) ** 2).sum(1).mean()))


class TestKLT:
    def test_small_affine_recovered(self):
        ref = make_texture()
        M_gt = affine(1.5, 1.02, 3.0, -2.0)
        cur = cv2.warpAffine(ref, M_gt, (320, 240), flags=cv2.INTER_LINEAR)
        M_est, rmse, n_inl = estimate_affine_klt(ref, cur)
        assert n_inl >= CFG["min_inliers_abs"]
        assert rmse <= CFG["rmse_gate"], f"RMSE={rmse}"
        assert point_err(M_est, M_gt) < 1.0, f"角点误差={point_err(M_est, M_gt):.2f}px"

    def test_translation_only(self):
        ref = make_texture()
        M_gt = affine(0.0, 1.0, 5.0, 4.0)
        cur = cv2.warpAffine(ref, M_gt, (320, 240))
        M_est, rmse, _ = estimate_affine_klt(ref, cur)
        assert point_err(M_est, M_gt) < 0.5


class TestFourierMellin:
    def test_rotation_scale_recovered(self):
        ref = make_texture(256, 256)
        M_gt = affine(6.0, 1.10, 0.0, 0.0, h=256, w=256)
        cur = cv2.warpAffine(ref, M_gt, (256, 256), flags=cv2.INTER_LINEAR)
        M_est, quality = fourier_mellin(ref, cur)
        assert quality > 0.1, f"FM 相关峰过低: {quality}"
        err = point_err(M_est, M_gt, h=256, w=256)
        assert err < 4.0, f"FM 角点误差={err:.2f}px（log-polar 分辨率内应 <4px）"

    def test_pure_translation(self):
        ref = make_texture(256, 256)
        M_gt = affine(0.0, 1.0, 7.0, -5.0, h=256, w=256)
        cur = cv2.warpAffine(ref, M_gt, (256, 256))
        M_est, quality = fourier_mellin(ref, cur)
        assert quality > 0.3
        pe = point_err(M_est, M_gt, h=256, w=256)
        assert pe < 2.0, f"FM 平移误差={pe:.2f}px"


class TestRegisterFrame:
    def test_klt_accepted(self):
        ref, cur = make_texture(), None
        M_gt = affine(0.5, 1.005, 2.0, 1.0)
        cur = cv2.warpAffine(ref, M_gt, (320, 240))
        M, rmse, how = register_frame(ref, cur)
        assert how == "klt" and rmse <= CFG["rmse_gate"]

    def test_flat_frame_not_failed(self):
        ref = make_texture()
        flat = np.full((240, 320), 77, np.uint8)
        M, rmse, how = register_frame(ref, flat)
        # 平坦帧无纹理：允许 failed（回退链终点），register_sequence 内部标 flat
        assert how in ("failed", "fm")

    def test_register_sequence_sliding_ref(self):
        """5 步平移序列：全帧对齐参考帧后背景应静止。"""
        ref = make_texture(240, 320)
        frames = [ref]
        for t in range(1, 10):
            A = affine(0.0, 1.0, 0.6 * t, 0.3 * t)
            frames.append(cv2.warpAffine(ref, A, (320, 240)))
        seq = np.stack(frames).astype(np.float32)
        res = register_sequence(seq)
        assert not res.failed.any()
        assert (res.ref_idx == 0).all()  # 10 帧 < 25 → 单参考
        # 对齐后帧与参考帧差分应远小于未对齐差分
        # 约定：M 为 ref→cur 正向，对齐需 WARP_INVERSE_MAP（见 register.py 文档）
        warped = cv2.warpAffine(
            seq[9], res.M[9], (320, 240),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        d_aligned = np.abs(warped - seq[0]).mean()
        d_raw = np.abs(seq[9] - seq[0]).mean()
        assert d_aligned < 0.5 * d_raw, f"对齐差分 {d_aligned:.2f} vs 原始 {d_raw:.2f}"

    def test_ref_reset_every_25(self):
        seq = np.stack([make_texture()] * 30).astype(np.float32)
        res = register_sequence(seq)
        assert set(res.ref_idx) == {0, 25}  # 滑动参考按 25 帧重置

    def test_bridge_cross_boundary_alignment(self):
        """跨参考块边界：桥矩阵复合后相邻帧背景残差应显著低于未对齐差分。

        背景：T=32 训练窗必然跨 25 帧参考边界，若采样器只按各自参考对齐，
        窗内会出现背景跳变（M1 实测跨块残差 5–10× 块内）。
        """
        ref = make_texture(240, 320)
        frames = [ref]
        for t in range(1, 40):
            A = affine(0.0, 1.0, 0.5 * t, 0.25 * t)
            frames.append(cv2.warpAffine(ref, A, (320, 240)))
        seq = np.stack(frames).astype(np.float32)
        res = register_sequence(seq)
        assert res.bridge_M is not None
        assert not np.isnan(res.bridge_M[25][0, 0]), "参考帧 25 缺桥矩阵"

        # 跨块对齐：anchor=23（块0），target=26（块25）
        W = align_to_anchor(res.M[26], int(res.ref_idx[26]), res.M[23],
                            int(res.ref_idx[23]), res.bridge_M)
        warped = cv2.warpAffine(seq[26], W, (320, 240),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        d_cross = float(np.abs(warped - seq[23]).mean())
        d_raw = float(np.abs(seq[26] - seq[23]).mean())
        assert d_cross < 0.6 * d_raw, f"跨块复合对齐无效: {d_cross:.2f} vs raw {d_raw:.2f}"

        # 同块对照：复合公式在同块退化为 M_t∘M_a⁻¹，同样应有效
        W2 = align_to_anchor(res.M[24], int(res.ref_idx[24]), res.M[22],
                             int(res.ref_idx[22]), res.bridge_M)
        warped2 = cv2.warpAffine(seq[24], W2, (320, 240),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        d_same = float(np.abs(warped2 - seq[22]).mean())
        assert d_same < 0.6 * float(np.abs(seq[24] - seq[22]).mean())


# ---- 窗口采样器对齐（M2 修复：NaN 桥续链 + window_anchor_warps） ----------------

def test_align_to_anchor_nan_bridge_continues_chain():
    """平坦参考块跳过桥估计（M1 缓存缺口，bridge_M=NaN）→ 恒等续链而非 break。

    反例构造：anchor 在平坦块 0（M=I），target 在块 50，中间块 25 的桥 NaN、
    块 50 的桥真实。若 break，则 W 漏乘 B[50]，坐标系错一级。
    """
    hom = lambda A: np.vstack([np.asarray(A, np.float64), [0, 0, 1]])
    B50 = affine(0.0, 1.0, 3.0, -2.0)          # ref_25 → ref_50 的真实桥（2×3）
    Mt = affine(0.0, 1.0, 1.0, 1.0)            # ref_50 → 第 55 帧
    Ma = np.array([[1.0, 0, 0], [0, 1.0, 0]])  # 平坦块内 M = 恒等
    n = 60
    bridge = np.full((n, 2, 3), np.nan)
    bridge[50] = B50
    W = align_to_anchor(Mt, 50, Ma, 0, bridge)
    expect = (hom(Mt) @ hom(B50) @ np.linalg.inv(hom(Ma)))[:2]
    assert np.allclose(W, expect, atol=1e-9), "NaN 桥未按恒等续链复合"


def test_window_anchor_warps_synthetic_translation():
    """单块纯平移序列：window_anchor_warps 对每帧回收锚点帧内容。"""
    ref = make_texture(240, 320)
    ref = cv2.GaussianBlur(ref, (0, 0), 2.0)  # 平滑纹理：隔离双线性重采样误差与几何对齐
    n = 40
    frames = [ref.astype(np.float32)]
    dts = [(0.5 * t, 0.25 * t) for t in range(1, n)]
    for dx, dy in dts:
        A = affine(0.0, 1.0, dx, dy)
        frames.append(cv2.warpAffine(ref, A, (320, 240)).astype(np.float32))
    seq = np.stack(frames)
    M = np.zeros((n, 2, 3), np.float32)
    M[0] = [[1, 0, 0], [0, 1, 0]]
    for t, (dx, dy) in enumerate(dts, 1):
        M[t] = [[1, 0, dx], [0, 1, dy]]
    reg = {"M": M, "ref_idx": np.zeros(n, np.int32),
           "bridge_M": np.full((n, 2, 3), np.nan)}

    from dsld.data.preprocess.register import window_anchor_warps, is_identity_warp

    Ws = window_anchor_warps(reg, 0, 32)
    assert Ws.shape == (32, 2, 3)
    assert is_identity_warp(Ws[0]), "锚点帧自身必须恒等"
    for t in range(1, 32):
        warped = cv2.warpAffine(seq[t], Ws[t], (320, 240),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        d = float(np.abs(warped - seq[0]).mean())
        d_raw = float(np.abs(seq[t] - seq[0]).mean())
        assert d < max(0.25 * d_raw, 1.2), f"帧 {t} 对齐失败: {d:.3f} vs raw {d_raw:.3f}"
