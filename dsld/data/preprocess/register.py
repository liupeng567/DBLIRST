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
    """
    def hom(M):
        return np.vstack([np.asarray(M, np.float64), [0, 0, 1]])

    W = np.linalg.inv(hom(M_a))          # anchor → ref_a
    r = ref_a
    while r < ref_t:                     # ref_a → ref_t 逐级过桥
        nxt = r + CFG["ref_every"]
        if nxt > len(bridge_M) - 1 or np.isnan(bridge_M[nxt][0, 0]):
            break                        # 无桥可依（不应发生），退化为不跨
        W = hom(bridge_M[nxt]) @ W
        r = nxt
    W = hom(M_t) @ W                     # ref_t → t
    return W[:2]


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
