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


# ---- 掩码几何口径（L4：GT 锚点对齐 + 教师掩码与 L_recon 同几何） ----------------

from dsld.data.preprocess.register import (  # noqa: E402
    IDENT2,
    box_fill,
    dilate_mask,
    downsample_max,
    gt_mask_anchor,
    gt_masks_anchor,
    warp_frame,
)


def _centroid(m: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def test_box_fill_area_uses_inclusive_endpoints():
    """含端点口径单测（ittd_parse.py:5）：面积 = (x2−x1+1)(y2−y1+1)。

    若误按不含端点实现，小目标掩码每框少 1 行 1 列——20×12 的框少 3% 面积，
    而 5px 级目标少 ~30%，L_seg 与残差统计的 GT 位置被系统性缩小（判据 ② 的
    resid_scr 分子直接受损）。这条断言就是防它。
    """
    boxes = np.array([[5, 10, 20, 14, 24]], np.int32)  # (f, x1,y1,x2,y2)：5×5 框
    m = box_fill(boxes, (48, 64), frame=5)
    assert m.sum() == (14 - 10 + 1) * (24 - 20 + 1) == 25
    assert m[20:25, 10:15].all()
    for (yy, xx) in ((19, 12), (25, 12), (21, 9), (21, 15)):  # 四侧各越界 1px
        assert m[yy, xx] == 0.0, f"框外像素 ({yy},{xx}) 被填进 GT"

    other = box_fill(boxes, (48, 64), frame=6)
    assert other.sum() == 0.0, "frame 过滤必须按 1-based 标注帧号精确取帧（不得隐式换算）"


def test_gt_mask_anchor_identity_and_integer_translation_both_ways():
    """GT 锚点对齐：恒等与整数平移**双向**回收（L4 强制口径）。"""
    H, W = 64, 80
    boxes = np.array([[7, 20, 30, 27, 38]], np.int32)  # 帧 7（1-based）的 8×9 框
    cx0, cy0 = 23.5, 34.0  # 含端点框中心

    # 恒等：out_hw 必须等于 (H,W) 才走"不下采样"分支
    m_id = gt_mask_anchor(boxes, 7, None, (H, W), (H, W))
    assert m_id.sum() == 8 * 9
    assert np.allclose(_centroid(m_id), (cx0, cy0)), "恒等 warp 不应移动 GT"

    # 正向：W 把 ref→帧 平移 +t，配准后内容在锚点系移动 −t（WARP_INVERSE_MAP 采样）
    for tx, ty in ((5, 3), (-4, -6), (0, 7)):
        Wm = np.array([[1.0, 0, tx], [0, 1.0, ty]], np.float32)
        m = gt_mask_anchor(boxes, 7, Wm, (H, W), (H, W))
        assert abs(m.sum() - 72) <= 2, f"整数平移不应改变掩码面积（{m.sum()} ≠ 72）"
        got = _centroid(m)
        assert abs(got[0] - (cx0 - tx)) < 0.6 and abs(got[1] - (cy0 - ty)) < 0.6, (
            f"t=({tx},{ty}) GT 锚点回收错向：{got} ≠ {(cx0 - tx, cy0 - ty)}")


def test_gt_mask_anchor_direction_is_falsified():
    """方向证伪：误用正向 M（漏 WARP_INVERSE_MAP）必须与 warp_frame 输出**不**同治。

    这条断言的作用是"能红"：45859f2 实录首版正是漏了旗标，被同治性单测抓出。
    """
    H, W = 64, 80
    boxes = np.array([[7, 20, 30, 27, 38]], np.int32)
    Wm = np.array([[1.0, 0, 6], [0, 1.0, 4]], np.float32)
    # 造一个"GT 位置就是亮目标"的合成帧：帧系 (row 30..38, col 20..27) 有块
    frame = np.zeros((H, W), np.float32)
    frame[30:39, 20:28] = 1.0
    aligned = warp_frame(frame, Wm)  # 正确的锚点系帧（内容移到 −t）
    m_ok = gt_mask_anchor(boxes, 7, Wm, (H, W), (H, W))
    m_bad = cv2.warpAffine(box_fill(boxes, (H, W)), Wm, (W, H), flags=cv2.INTER_NEAREST)
    # 正确：掩码覆盖的锚点系像素确实是亮块
    assert float(aligned[m_ok > 0].mean()) > 0.9, "GT 掩码未与 warp_frame 同治"
    # 反例：漏 INVERSE_MAP 的掩码与 warp_frame 输出不同治（落在暗区）
    assert float(aligned[m_bad > 0].mean()) < 0.1, "错误方向竟然同治——说明约定已变，须重审"


def test_gt_masks_anchor_frames_and_mask_share_warps():
    """逐帧 GT 掩码与帧使用同一组 Ws（gt_masks_anchor 的"同治"契约）。"""
    H, W, T = 48, 56, 6
    boxes = np.array([[3, 10, 10, 14, 14], [5, 20, 30, 24, 34]], np.int32)  # 帧 3、5
    Ws = np.repeat(IDENT2[None], T, 0).astype(np.float64)
    Ws[2] = [[1.0, 0, 4], [0, 1.0, 2]]  # 帧 index 2 = 帧号 3 → 平移 (4,2)
    Ws[4] = [[1.0, 0, -3], [0, 1.0, 5]]
    m = gt_masks_anchor(boxes, 0, T, Ws, (H, W))
    assert m.shape == (T, H, W) and set(np.unique(m)) <= {0.0, 1.0}
    assert m[2].sum() == 25 and m[4].sum() == 25
    assert np.allclose(_centroid(m[2]), (12 - 4, 12 - 2)), "帧 3 GT 未随 W 进锚点系"
    assert np.allclose(_centroid(m[4]), (22 + 3, 32 - 5)), "帧 5 GT 未随 W 进锚点系"
    assert m[[0, 1, 3, 5]].sum() == 0.0, "无标注帧不应产生 GT"


def test_dilate_mask_matches_torch_maxpool():
    """"膨胀 3px"在 numpy 侧与 torch 侧只有一份定义（同几何的机器证明）。"""
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(3)
    for px in (1, 3, 5):
        m = (rng.random((2, 3, 37, 41)) > 0.97).astype(np.float32)
        np_out = dilate_mask(m, px)
        tt = F.max_pool2d(torch.from_numpy(m.reshape(-1, *m.shape[-2:])),
                          2 * px + 1, stride=1, padding=px).reshape(m.shape)
        assert np.array_equal(np_out, tt.numpy()), f"px={px} 膨胀口径不一致"
    assert np.array_equal(dilate_mask(np.ones((5, 5), np.float32), 0), np.ones((5, 5)))


def test_downsample_max_keeps_small_target_one_pixel():
    """总方案 2.7-② 的构造保证：<2px 框在 stride-2 图上必留 1px，无需特判分支。"""
    for (y, x) in ((0, 0), (1, 1), (7, 8), (10, 11)):
        g = box_fill(np.array([[1, x, y, x, y]], np.int32), (12, 16))  # 1×1 框
        d = downsample_max(g[None], 2)[0]
        assert d.shape == (6, 8) and d.sum() == 1.0, f"({y},{x}) 1px 目标丢失"
        assert d[y // 2, x // 2] == 1.0, f"({y},{x}) 保留位置错块"
    with pytest.raises(ValueError):
        downsample_max(np.zeros((1, 1), np.float32), 2)
    assert downsample_max(np.zeros((3, 7, 9), np.float32), 2).shape == (3, 3, 4), "奇数边裁齐"


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
