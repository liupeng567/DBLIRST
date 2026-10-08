"""2.4 帧间运动补偿：Harris + KLT + RANSAC 仿射配准。

方案 2.4 配置：
  - 特征点 Harris：blockSize=8, maxCorners=2000, qualityLevel=0.01
  - 跟踪：三层金字塔 KLT，窗口 21×21
  - 变换模型：仿射（2×3，6 自由度）；RANSAC 重投影阈值 1.0 px
  - 参考帧：滑动参考，每 25 帧更新（防止误差累积漂移）
  - 质量门限：内点 ≥ 100 且 RMSE ≤ 0.5 px → 接受；否则回退 Fourier-Mellin；
    仍失败 → reg_failed，恒等变换

矩阵语义（固定约定）：reg.M[t] 将"参考帧坐标"映射到"第 t 帧坐标"（KLT 正向），
即第 t 帧中内容出现在 M[t]·u 处。**把第 t 帧对齐到参考帧必须用
cv2.warpAffine(frame_t, M[t], (W,H), flags=… | cv2.WARP_INVERSE_MAP)**
（warpAffine 默认把 M 当 src→dst 正向变换内部取逆，与采样语义相反——
本约定经 np.roll 数值实验敲定，M2 窗口采样器不得再用 inv）。
平场/平坦帧（无纹理）标记 method='flat'，恒等变换，不计入 reg_failed。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CFG = {
    "max_corners": 2000,
    "quality": 0.001,        # 方案默认 0.01 在 LWIR 上被热目标边缘垄断（相对阈值），
                             # 实测seq2 仅 22 角点 → 全部退 FM；0.001 恢复数百角点（M1 报告记录偏离）
    "block_size": 8,
    "min_distance": 5,       # 方案默认 8；同因纹理平滑放松（偏离已记录）
    "use_harris": True,
    "klt_win": (21, 21),
    "klt_levels": 3,
    "ransac_thresh": 1.0,
    "min_inliers_abs": 100,  # 方案 2.4 门限（角点充足时生效）
    "min_inliers_floor": 30, # 角点贫乏序列的下限（LWIR 平滑场景，自适应门限的下界）
    "min_inlier_ratio": 0.4, # 内点/跟踪点比例下限（防 RANSAC 在噪声中共识）
    "rmse_gate": 0.5,        # px
    "fm_corr_gate": 0.5,     # FM 回退的互相关接受门限
    "fm_downscale": 2,       # FM 半分辨率估计（回退路径提速 ~4×，精度足够）
    "ref_every": 25,         # 滑动参考周期
    "flat_std_gate": 1.0,    # 帧内灰度 std < 1.0 视为平坦帧
}


def _inlier_gate(n_tracked: int) -> int:
    """自适应内点门限：角点充足 → 方案门限 100；贫乏 → max(30, 0.4×跟踪数)。"""
    return max(CFG["min_inliers_floor"],
               min(CFG["min_inliers_abs"], int(CFG["min_inlier_ratio"] * n_tracked)))


@dataclass
class RegResult:
    M: np.ndarray            # [N,2,3] float32，M[t]: ref(t) 坐标 → 第 t 帧坐标
    rmse: np.ndarray         # [N] float32（FM 帧存 1-corr 作质量代理）
    failed: np.ndarray       # [N] bool，True = 配准失败（恒等）
    method: np.ndarray       # [N] str: 'ref'/'klt'/'fm'/'flat'
    ref_idx: np.ndarray      # [N] int，每帧所属参考帧号
    bridge_M: np.ndarray | None = None   # [N,2,3] 参考帧相对前一参考的桥矩阵（非参考帧为 NaN）
    bridge_rmse: np.ndarray | None = None  # [N] 桥估计的 RMSE
    extras: dict = field(default_factory=dict)

    def save(self, path) -> None:
        np.savez_compressed(
            path, M=self.M, rmse=self.rmse, failed=self.failed,
            method=self.method.astype("U4"), ref_idx=self.ref_idx,
            bridge_M=self.bridge_M, bridge_rmse=self.bridge_rmse,
        )


def align_to_anchor(M_t, ref_t: int, M_a, ref_a: int, bridge_M) -> np.ndarray:
    """计算把第 t 帧对齐到锚点帧 a 的采样矩阵 W（M2 窗口采样器的唯一入口）。

    W 满足：cv2.warpAffine(frame_t, W, (W,H), flags=…|WARP_INVERSE_MAP) 输出即对齐到
    anchor 的帧。M_t/M_a 分别为 t/anchor 相对各自参考块的正向矩阵（ref→帧），
    bridge_M[r] 为参考帧 r 相对前一参考 (r−25) 的正向估计（reg.npz: bridge_M）。

        同块（ref_t == ref_a）：   W = M_t ∘ M_a⁻¹
        跨块（ref_t > ref_a）：    W = M_t ∘ B[ref_t] ∘ … ∘ B[ref_a+25] ∘ M_a⁻¹

    T=32 窗最多跨 1 个边界，复合链 ≤3 个矩阵；要求 t ≥ anchor（窗口锚定首帧）。

    桥缺失（NaN）语义：register_sequence 对平坦参考块跳过桥估计（M1 缓存中的已知
    缺口），此时该桥取恒等并**继续复合链**（平坦块本身即按"恒等即正确对齐"标注，
    register.py:260）——不能 break，否则后续 M_t 会作用在错误的参考坐标系上。
    配准失败的桥在缓存中已存为恒等（bridge_rmse=inf），不经过此分支。
    """
    def hom(M):
        return np.vstack([np.asarray(M, np.float64), [0, 0, 1]])

    W = np.linalg.inv(hom(M_a))          # anchor → ref_a
    r = ref_a
    while r < ref_t:                     # ref_a → ref_t 逐级过桥
        nxt = r + CFG["ref_every"]
        if nxt > len(bridge_M) - 1:
            break                        # 无桥可依（不应发生），退化为不跨
        B = bridge_M[nxt]
        if np.isnan(B[0, 0]):
            B = _identity()              # 平坦块缺口 → 恒等续链，保持参考系列一致
        W = hom(B) @ W
        r = nxt
    W = hom(M_t) @ W                     # ref_t → t
    return W[:2]


IDENT2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def is_identity_warp(W: np.ndarray, atol: float = 1e-3) -> bool:
    """准恒等判定（静态相机序列整窗免重采样，避免双重插值模糊）。"""
    return bool(np.allclose(np.asarray(W, np.float64), IDENT2, atol=atol))


def window_anchor_warps(reg, start: int, T: int) -> np.ndarray:
    """窗口 [start, start+T) 各帧对齐到锚点帧 start 的采样矩阵 [T,2,3]。

    reg 为载入的 reg.npz（dict，含 M/ref_idx/bridge_M）或 RegResult。返回的
    Ws[i] 满足 warpAffine(frame_{start+i}, Ws[i], flags=…|WARP_INVERSE_MAP) 输出
    即锚点坐标系帧；Ws[0] 恒为恒等（锚点帧本身不重采样）。要求 start+T ≤ N。
    """
    M = reg["M"] if isinstance(reg, dict) else reg.M
    ref_idx = reg["ref_idx"] if isinstance(reg, dict) else reg.ref_idx
    bridge = reg["bridge_M"] if isinstance(reg, dict) else reg.bridge_M
    n = len(M)
    if start < 0 or start + T > n:
        raise IndexError(f"窗口 [{start},{start+T}) 越界（N={n}）")
    Ws = np.repeat(IDENT2[None], T, axis=0).astype(np.float64)
    ref_a = int(ref_idx[start])
    for i in range(1, T):
        t = start + i
        Ws[i] = align_to_anchor(M[t], int(ref_idx[t]), M[start], ref_a, bridge)
    return Ws


def warp_frame(img: np.ndarray, W: np.ndarray, border_mode: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """把 img 按采样矩阵 W 对齐到锚点坐标系（WARP_INVERSE_MAP 采样，register.py 约定）。"""
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img, np.asarray(W, np.float32), (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border_mode,
    )


# ---- 掩码几何口径（窗口数据集 / 教师掩码 / L_recon 剔除区共用） ----------------
# 放在本模块的原因：这三者是同一个"锚点坐标系 + 采样语义"问题的三种投影。
# 配准口径（WARP_INVERSE_MAP）、GT 锚点对齐、膨胀/块最大下采样若各处各写一份，
# 就会重演 M3 审核的 L4 事故（残差统计被窗内平台位移污染、warp 方向取逆）。
# 任何消费者（dataset / core / losses / quick_eval）都必须调这里，不得自行实现。


def dilate_mask(m: np.ndarray, px: int) -> np.ndarray:
    """方核膨胀 px 像素（(2px+1)² 全结构元），二值/浮点掩码均按最大值传播。

    与 losses 的 F.max_pool2d(m, 2px+1, stride=1, padding=px) 逐位等价（单测锁定），
    故"膨胀 3px"这一几何口径在 numpy 侧与 torch 侧只有一份定义。
    支持任意前导维（[H,W] / [T,H,W] / [B,T,C,H,W]）：膨胀始终作用在最后两维。
    """
    m = np.asarray(m)
    if px <= 0:
        return m.copy()
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    if m.ndim <= 2:
        return cv2.dilate(m, k, borderType=cv2.BORDER_CONSTANT)
    flat = np.ascontiguousarray(m.reshape(-1, *m.shape[-2:]))
    out = np.empty_like(flat)
    for i in range(len(flat)):
        out[i] = cv2.dilate(flat[i], k, borderType=cv2.BORDER_CONSTANT)
    return out.reshape(m.shape)


def downsample_max(m: np.ndarray, stride: int = 2) -> np.ndarray:
    """块最大下采样（stride×stride 取块内最大值），作用于最后两维。

    对二值掩码语义 = "块内有任何前景像素则保留 1"，故原生域 <2px 的框在 stride-2
    特征图上必然保留 1px（总方案 2.7-② 的"小目标保中心 1px"由本性质构造满足，
    无需特判分支）。奇数边长右/下裁齐，不留半块。
    """
    m = np.asarray(m)
    h, w = m.shape[-2], m.shape[-1]
    hh, ww = h // stride, w // stride
    if hh == 0 or ww == 0:
        raise ValueError(f"掩码 {m.shape} 小于下采样步长 {stride}")
    v = m[..., : hh * stride, : ww * stride]
    return v.reshape(*v.shape[:-2], hh, stride, ww, stride).max(axis=(-3, -1))


def box_fill(boxes: np.ndarray, img_hw: tuple[int, int], frame: int | None = None) -> np.ndarray:
    """框填充掩码 float32 [H,W]（总方案 2.7-②：目标极小，填框优于分割轮廓）。

    boxes: [N,5] = (frame, x1, y1, x2, y2)；帧号 1-based（ITTD 标注口径），
    坐标**含端点**（ittd_parse.py:5 —— 面积 = (x2−x1+1)(y2−y1+1)，单测锁定）。
    frame 给出时只填该帧（**与 boxes 同口径的 1-based 帧号**，不做隐式换算——
    缓存数组下标才 0-based，混用即静默丢标注）；越界框取可见部分。
    """
    H, W = int(img_hw[0]), int(img_hw[1])
    g = np.zeros((H, W), np.float32)
    boxes = np.asarray(boxes)
    if not len(boxes):
        return g
    if frame is not None:
        boxes = boxes[boxes[:, 0].astype(int) == int(frame)]
    for f, x1, y1, x2, y2 in boxes:
        g[max(int(y1), 0):min(int(y2) + 1, H), max(int(x1), 0):min(int(x2) + 1, W)] = 1.0
    return g


def gt_masks_anchor(boxes: np.ndarray, start: int, T: int, warps: np.ndarray | None,
                    img_hw: tuple[int, int]) -> np.ndarray:
    """窗口 [start, start+T) 逐帧 GT 框填充掩码，统一到**锚点坐标系** [T,H,W]。

    L4 纪律的实现入口：帧被 warp 到锚点系，标注仍在原始帧坐标系，二者必须同治——
    掩码施加与该帧完全相同的 W（WARP_INVERSE_MAP + INTER_NEAREST 保二值、越界补 0，
    不伪造目标）。warps=None 时退化为不对齐（消融/调试用）。
    """
    H, W = int(img_hw[0]), int(img_hw[1])
    m = np.zeros((T, H, W), np.float32)
    boxes = np.asarray(boxes)
    if not len(boxes):
        return m
    fno = boxes[:, 0].astype(int)
    in_win = (fno >= start + 1) & (fno < start + T + 1)
    for t in range(T):
        sel = in_win & (fno == start + t + 1)
        if not sel.any():
            continue
        g = box_fill(boxes[sel], (H, W))
        if warps is not None and not is_identity_warp(warps[t]):
            g = cv2.warpAffine(
                g, np.asarray(warps[t], np.float32), (W, H),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
        m[t] = g
    return m


def gt_mask_anchor(boxes: np.ndarray, fno_last: int, warp, img_hw, out_hw,
                   dilate_px: int = 0) -> np.ndarray:
    """单帧 GT 框 → 锚点坐标系 →（可选膨胀）→ 块最大下采样到 out_hw 的掩码。

    与 gt_masks_anchor 同一采样语义（WARP_INVERSE_MAP；reg.M 是 ref→帧 正向矩阵，
    对齐到锚点必须逆采样——M1 教训"勿再取逆"）；膨胀在**下采样之前**于全分辨率做，
    保 3px 几何精度（教师掩码与 L_recon 剔除区同几何由此保证）。
    boxes: [(f, x1, y1, x2, y2), ...]；fno_last: 1-based 帧号（与 boxes 列同口径）；
    warp: 2×3 或 None（恒等）；img_hw=(H,W) 原生分辨率；out_hw=(h,w) 目标（特征）分辨率。
    """
    H, W = int(img_hw[0]), int(img_hw[1])
    g = box_fill(boxes, (H, W), frame=int(fno_last))
    if warp is not None and not is_identity_warp(warp):
        g = cv2.warpAffine(
            g, np.asarray(warp, np.float32), (W, H),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
    if dilate_px > 0:
        g = dilate_mask(g, int(dilate_px))
    h2, w2 = int(out_hw[0]), int(out_hw[1])
    return downsample_max(g[:h2 * 2, :w2 * 2], 2) if (h2, w2) != (H, W) else g


def _identity() -> np.ndarray:
    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


def _rmse_of(pts_ref: np.ndarray, pts_cur: np.ndarray, M: np.ndarray) -> float:
    if len(pts_ref) == 0:
        return float("inf")
    proj = cv2.transform(pts_ref.reshape(-1, 1, 2), M).reshape(-1, 2)
    err = np.linalg.norm(proj - pts_cur.reshape(-1, 2), axis=1)
    return float(np.sqrt((err ** 2).mean()))


def _to_u8(img: np.ndarray) -> np.ndarray:
    """KLT/Harris 需要 CV_8U；float 输入（NUC 校正域）裁剪到 [0,255]。"""
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def estimate_affine_klt(ref: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Harris 特征点 + KLT + RANSAC 仿射。返回 (M: ref→cur, rmse, n_inliers)。"""
    ref_u8, cur_u8 = _to_u8(ref), _to_u8(cur)
    pts = cv2.goodFeaturesToTrack(
        ref_u8, maxCorners=CFG["max_corners"], qualityLevel=CFG["quality"],
        minDistance=CFG["min_distance"], blockSize=CFG["block_size"],
        useHarrisDetector=CFG["use_harris"],
    )
    if pts is None or len(pts) < 50:
        return _identity(), float("inf"), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        ref_u8, cur_u8, pts, None, winSize=CFG["klt_win"], maxLevel=CFG["klt_levels"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    good = st.ravel() == 1
    pts_ref, pts_cur = pts[good], p1[good]
    if len(pts_ref) < 50:
        return _identity(), float("inf"), 0
    M, mask = cv2.estimateAffine2D(
        pts_ref, pts_cur, method=cv2.RANSAC,
        ransacReprojThreshold=CFG["ransac_thresh"], maxIters=2000, refineIters=10,
    )
    if M is None:
        return _identity(), float("inf"), 0
    inliers = int(mask.sum())
    gate = _inlier_gate(len(pts_ref))
    if inliers < gate:
        return _identity(), float("inf"), 0
    in_ref = pts_ref[mask.ravel() == 1]
    in_cur = pts_cur[mask.ravel() == 1]
    rmse = _rmse_of(in_ref, in_cur, M)
    return M, rmse, inliers


def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def fourier_mellin(ref: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, float]:
    """FM 相位相关回退：log-polar 幅度谱求旋转+尺度，再求平移。

    在 CFG['fm_downscale']× 缩小图上估计（回退路径提速 ~4×，亚像素平移精度仍足够），
    返回全分辨率 (M: ref→cur 对齐矩阵——与 KLT 主路同向, 相关峰质量 0-1)。
    推导：cur 的内容 = 旋转 θ₀ · 尺度 s₀ · (参考内容平移 t)，
    M(p) = A_rs⁻¹·(p + t)，A_rs = R(−θ₀)·(1/s₀)（把 cur 拉回 ref 姿态的校正）。
    """
    ds = CFG["fm_downscale"]
    if ds > 1:
        ref_s = cv2.resize(ref, (ref.shape[1] // ds, ref.shape[0] // ds),
                           interpolation=cv2.INTER_AREA)
        cur_s = cv2.resize(cur, (cur.shape[1] // ds, cur.shape[0] // ds),
                           interpolation=cv2.INTER_AREA)
    else:
        ref_s, cur_s = ref, cur
    h, w = ref_s.shape
    win = _hann2d(h, w)
    f1 = np.fft.fft2(ref_s.astype(np.float32) * win)
    f2 = np.fft.fft2(cur_s.astype(np.float32) * win)
    m1 = np.log1p(np.abs(np.fft.fftshift(f1))).astype(np.float32)
    m2 = np.log1p(np.abs(np.fft.fftshift(f2))).astype(np.float32)
    polar_flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG
    center = (w / 2, h / 2)
    max_r = min(w, h) / 2.0  # 内切圆半径：谱角部（r>W/2）不可靠，不进极坐标
    p1 = cv2.warpPolar(m1, (w, h), center, max_r, polar_flags)
    p2 = cv2.warpPolar(m2, (w, h), center, max_r, polar_flags)
    # 极域 Hann 窗（抑制角度 0/2π 环绕与半径边缘不连续）+ 9×9 高通
    w2d = _hann2d(h, w)
    p1 = ((p1 * w2d) - cv2.blur(p1 * w2d, (9, 9))).astype(np.float32)
    p2 = ((p2 * w2d) - cv2.blur(p2 * w2d, (9, 9))).astype(np.float32)
    (dpdx, dpdy), resp_ps = cv2.phaseCorrelate(p1, p2)
    # 实测（OpenCV warpPolar）：y 轴 = 角度（0..2π），x 轴 = log 半径
    theta = dpdy * (2 * np.pi / h)                    # cur 相对 ref 的旋转
    scale = float(np.exp(dpdx * np.log(max_r) / w))   # cur 相对 ref 的尺度
    # 把 cur 拉回 ref 姿态的校正（绕中心 旋转 −θ / 尺度 ×scale_est；
    # 尺度读数为 1/s₀——谱缩放定理，实验敲定为乘而非除）
    c, s = np.cos(-theta), np.sin(-theta)
    A_rs = np.eye(3)
    A_rs[:2, :2] = np.array([[c, -s], [s, c]]) * max(scale, 1e-9)
    A_rs[:2, 2] = np.array(center) - A_rs[:2, :2] @ np.array(center)
    cur_rs = cv2.warpPerspective(cur_s, A_rs, (w, h), flags=cv2.INTER_LINEAR)
    ((dx, dy), resp_t) = cv2.phaseCorrelate(
        (ref_s * win).astype(np.float32), (cur_rs * win).astype(np.float32)
    )
    T3 = np.eye(3)
    T3[:2, 2] = [dx, dy]
    M3 = np.linalg.inv(A_rs) @ T3          # p = A_rs⁻¹·(u + t)：ref 坐标 → cur 坐标
    quality = float(max(resp_ps * resp_t, 0.0))
    M_half = M3[:2].astype(np.float64)
    if ds > 1:  # 半分辨率 → 全分辨率：线性部分不变，平移 ×ds
        M_full = M_half.copy()
        M_full[:, 2] *= ds
        return M_full, quality
    return M_half, quality


def register_frame(ref: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, float, str]:
    """单帧配准：KLT 主路 → 质量门限 → FM 回退。返回 (M: ref→cur, rmse, method)。"""
    M, rmse, n_inl = estimate_affine_klt(ref, cur)
    if n_inl > 0 and rmse <= CFG["rmse_gate"]:  # inlier 门限已在 estimate 内自适应判定
        return M.astype(np.float32), rmse, "klt"
    M_fm, quality = fourier_mellin(ref, cur)
    if quality >= CFG["fm_corr_gate"]:
        return M_fm.astype(np.float32), 1.0 - quality, "fm"
    return _identity().astype(np.float32), float("inf"), "failed"


def _compose(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """复合仿射：先 M1 再 M2（p' = M2(M1(p))）。"""
    A1 = np.vstack([M1, [0, 0, 1]])
    A2 = np.vstack([M2, [0, 0, 1]])
    return (A2 @ A1)[:2]


def register_sequence(frames: np.ndarray) -> RegResult:
    """整段配准（滑动参考，每 CFG['ref_every'] 帧更新）。

    块内先试 ref→t 直接跟踪；失败则链式回退（从上一成功帧续推复合矩阵），
    平台累积运动超出 KLT 金字塔容量时兜底（链条每 25 帧被参考重置截断，不累积漂移）。

    桥矩阵（跨块对齐的关键）：每块额外估计"下一参考帧相对本参考"的矩阵并存入
    bridge_M[next_ref]，供 M2 采样器跨块复合（align_to_anchor）——否则 T=32 窗
    必然跨边界（25<32），窗内背景会对新参考跳变（实测跨块相邻帧残差 5–10×块内）。

    frames: [N,H,W]（uint8 或 float32，NUC 校正域）。返回 RegResult。
    """
    n = len(frames)
    M_out = np.repeat(_identity()[None], n, axis=0).astype(np.float32)
    rmse = np.full(n, np.inf, dtype=np.float32)
    failed = np.zeros(n, dtype=bool)
    method = np.empty(n, dtype="U6")
    ref_idx = np.zeros(n, dtype=np.int32)
    bridge_M = np.full((n, 2, 3), np.nan)
    bridge_rmse = np.full(n, np.inf, dtype=np.float32)
    n_chained = 0

    refs = list(range(0, n, CFG["ref_every"]))
    for r in refs:
        ref_img = frames[r]
        ref_idx[r] = r
        block_end = min(r + CFG["ref_every"], n)   # 本块末（不含下一参考帧）
        block = list(range(r + 1, block_end))
        # 平坦参考帧：整块标记 flat（无纹理，恒等即正确对齐）
        if float(np.asarray(ref_img).std()) < CFG["flat_std_gate"]:
            method[r], method[block] = "flat", "flat"
            rmse[r], rmse[block] = 0.0, 0.0
            continue
        method[r], rmse[r] = "ref", 0.0
        last_good = r
        for t in block:
            if float(np.asarray(frames[t]).std()) < CFG["flat_std_gate"]:
                method[t], rmse[t], ref_idx[t] = "flat", 0.0, r
                continue
            M, rms, how = register_frame(ref_img, frames[t])
            if how == "failed" and t > last_good:
                # 链式回退：上一成功帧 → 当前帧，复合到参考系
                M_p, rms_p, how_p = register_frame(frames[last_good], frames[t])
                if how_p != "failed":
                    M = _compose(M_out[last_good], M_p)
                    rms = rms_p
                    how = "klt2" if how_p == "klt" else "fm2"
                    n_chained += 1
            M_out[t], rmse[t], method[t], ref_idx[t] = M, min(rms, 1e6), how, r
            failed[t] = how == "failed"
            if how != "failed":
                last_good = t
        # 桥估计：下一参考帧相对本参考（下一块会把它置为恒等，故必须单独存）
        if block_end < n:
            nr = block_end
            if float(np.asarray(frames[nr]).std()) < CFG["flat_std_gate"]:
                bridge_M[nr] = _identity()
                bridge_rmse[nr] = 0.0
                continue
            Mb, rms_b, how_b = register_frame(ref_img, frames[nr])
            if how_b == "failed" and nr > last_good:
                M_p, rms_p, how_p = register_frame(frames[last_good], frames[nr])
                if how_p != "failed":
                    Mb = _compose(M_out[last_good], M_p)
                    rms_b, how_b = rms_p, "chained"
            bridge_M[nr] = Mb
            bridge_rmse[nr] = min(rms_b, 1e6) if how_b != "failed" else np.inf
    return RegResult(M=M_out, rmse=rmse, failed=failed, method=method,
                     ref_idx=ref_idx, bridge_M=bridge_M, bridge_rmse=bridge_rmse,
                     extras={"n_chained": n_chained})
