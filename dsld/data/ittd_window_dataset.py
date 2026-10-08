"""ITTD 时序窗口数据集（M3 v2.0 方案 §8.1：align_to_anchor 对齐 + 整窗一致增强）。

职能（方案 §4.1 数据流、§4.7 尺度档、§5.2 采样与对齐、§2.6 增强全表）：
  1. 窗口采样：T=32、训练步距 8 + 时间窗随机起点（§5.2）；
  2. 对齐唯一入口 register.window_anchor_warps（禁逐帧各归各参考，M1 §7.4 约束）；
  3. GT 锚点对齐强制（L4 教训）：帧与 GT 掩码同治——掩码经 register.gt_masks_anchor
     （WARP_INVERSE_MAP + NEAREST）变换到锚点坐标系，残差统计 / L_recon 剔除区
     全部落在同一坐标系；
  4. 训练档整窗一致随机裁剪 [240,320]，并含 §4.7 新增的"≥1 完整 GT 框（含 3px 边距）"
     采样约束（防"目标被裁半"制造边界漏检/虚警偏差）；
  5. 输出监督张量：target（框填充→stride-2 块最大）、teacher_mask（膨胀 3px→stride-2，
     **与 L_recon 剔除区同几何**——同一对函数，杜绝两处各写一份口径）；
  6. 增强全表（§2.6）：几何整窗一致、逐帧独立模糊、光度/噪声、帧复制、
     合成瞬时亮点注入（门控 r_T 支路的可控负样本）、缓变热斑注入（慢通道剥离训练）；
  7. 合法负样本（§2.1/10.9）：无标注窗（含 97 空标注帧所在的窗）按正常样本产出，
     全 0 掩码，禁止当损坏剔除；
  8. R4 缓解：reg_failed 密度 >10% 的窗在索引阶段剔除（鬼影簇不进训练）；
  9. track_index 消费：裁剪可行性按**轨迹在窗内的并集矩形**判定（同一目标整窗不得被
     裁半），用 labels.npz 的 boxes + track_ids。

域约定（错一处即静默毒化动力学）：
  - 帧号 1-based（标注）/ 0-based（缓存数组）；框坐标**含端点**（ittd_parse.py:5）。
  - 归一化域 x_n = (clip((x−med)/σ, ±8) + 8)/16 ∈ [0,1]，故"μ+kσ"在本域精确等于
    0.5 + k/16（合成注入幅值不需 σ_raw 换算）；只有灰度级噪声需 σ_raw 换算。
  - 配准 W 与 cv2.WARP_INVERSE_MAP 配套：点由帧坐标进锚点坐标用的是 **W⁻¹**；
    而增强小仿射 M 走 warpAffine 默认正向（src→dst）——两套语义方向相反，
    本文件各只在一处出现，tests/test_ittd_window.py 有方向证伪断言（L4）。

M2 的 mode="frame" 单帧档不重建（方案 0.1 跳过 M2；单帧参照按 M6 届时文献重选）。
"""

from __future__ import annotations

import cv2
import numpy as np
from torch.utils.data import Dataset

from dsld.data.manifest import load_manifest
from dsld.data.preprocess.normalize import correct_frame, normalize_frame
from dsld.data.preprocess.register import (
    dilate_mask,
    downsample_max,
    gt_masks_anchor,
    is_identity_warp,
    warp_frame,
    window_anchor_warps,
)

_CLIP = 8.0  # normalize.CLIP（归一化域半宽）；μ+kσ ↔ 0.5 + k/(2·CLIP)
_IDENT_W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _level(k_sigma: float) -> float:
    """背景中位 + k·σ 在归一化域 [0,1] 中的绝对取值。"""
    return 0.5 + float(k_sigma) / (2.0 * _CLIP)


def _to_anchor_rect(W: np.ndarray, x1, y1, x2, y2):
    """含端点框的四角 → 锚点坐标系外接矩形（W⁻¹ 语义，与 warp_frame 同治）。"""
    inv = np.linalg.inv(np.vstack([np.asarray(W, np.float64), [0.0, 0.0, 1.0]]))
    pts = [inv @ np.array([px, py, 1.0])
           for px, py in ((x1, y1), (x2 + 1, y1), (x2 + 1, y2 + 1), (x1, y2 + 1))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)  # 右下为**不含**端点的边（x2+1 派生）


def _track_rects(boxes: np.ndarray, track_ids: np.ndarray, start: int, T: int,
                 warps: np.ndarray | None):
    """窗内每条轨迹的锚点系并集矩形 [(tid, x1, y1, x2r, y2r, n_frames), ...]。

    并集覆盖该轨迹窗内**全部**可见帧——裁剪可行性据此判定，被选中的目标在窗内任何
    一帧都不会被裁半（只看首帧则后半窗的边界漏检偏差会原样回归）。
    越界框不裁剪：贴到画面边的目标由可行区间自然判为不可行（给不出 3px 边距）。
    """
    if not len(boxes):
        return []
    fno = boxes[:, 0].astype(int)
    in_win = (fno >= start + 1) & (fno < start + T + 1)
    out = []
    for tid in np.unique(track_ids[in_win]):
        sel = in_win & (track_ids == tid)
        r = [np.inf, np.inf, -np.inf, -np.inf]
        for f, x1, y1, x2, y2 in boxes[sel]:
            W = _IDENT_W if warps is None else warps[int(f) - 1 - start]
            a = _to_anchor_rect(W, x1, y1, x2, y2)
            r = [min(r[0], a[0]), min(r[1], a[1]), max(r[2], a[2]), max(r[3], a[3])]
        out.append((int(tid), r[0], r[1], r[2], r[3], int(sel.sum())))
    return out


class IttdWindows(Dataset):
    """ITTD 缓存窗口数据集（训练/快评用；序列级推理走 dsld/eval/infer_seq.py，P4）。"""

    def __init__(
        self,
        manifest_path: str,
        cache_root: str,
        split: str = "train-int",
        T: int = 32,
        stride: int = 8,
        crop: tuple[int, int] | None = (240, 320),
        start_jitter: bool = True,
        augment: bool = True,
        align: bool = True,
        highlight: bool = True,
        hotspot: bool = True,
        hotspot_p: float = 0.2,
        teacher_dilate: int = 3,
        feature_stride: int = 2,
        crop_margin: int = 3,
        max_reg_fail_frac: float = 0.10,
        max_crop_attempts: int = 8,
        seed: int = 0,
        limit_seqs: int = 0,
    ):
        self.manifest = load_manifest(manifest_path)
        self.cache_root = cache_root
        self.split = split
        self.T = int(T)
        self.stride = int(stride)
        self.crop = tuple(int(v) for v in crop) if crop else None
        self.start_jitter = bool(start_jitter) and self.stride > 1
        self.augment = bool(augment)
        self.align = bool(align)
        self.highlight = bool(highlight)
        self.hotspot = bool(hotspot)
        self.hotspot_p = float(hotspot_p)
        self.teacher_dilate = int(teacher_dilate)
        self.fs = int(feature_stride)
        self.crop_margin = int(crop_margin)
        self.max_crop_attempts = int(max_crop_attempts)
        self.seed = int(seed)
        self.attrs = {s["seq_id"]: s for s in self.manifest["sequences"]}
        self._seq_cache: dict[int, dict] = {}
        self._reg_fail: dict[int, np.ndarray] = {}
        # 裁剪档位计数（§4.7 约束的执行情况）。DataLoader 多 worker 下各进程独立计数，
        # 故本统计仅在 num_workers=0（本项目 Windows 实测档）或 QC 脚本中可信。
        self.crop_stats = {"attempts": 0, "all_tracks": 0, "one_track": 0,
                           "one_frame": 0, "negative": 0, "fallback": 0}
        self.index = self._build_index(max_reg_fail_frac, limit_seqs)

    # ---- 索引（§5.2 步距 8；R4 reg_failed 密度门限） ----------------------
    def _load_failed(self, seq_id: int) -> np.ndarray:
        if seq_id not in self._reg_fail:
            with np.load(f"{self.cache_root}/seq_{seq_id:04d}/reg.npz") as z:
                self._reg_fail[seq_id] = np.asarray(z["failed"], bool)
        return self._reg_fail[seq_id]

    def _build_index(self, max_fail_frac: float, limit_seqs: int) -> list[tuple[int, int]]:
        seqs = list(self.manifest["splits"][self.split]["seqs"])
        if limit_seqs:
            seqs = seqs[:limit_seqs]
        idx, dropped = [], 0
        for sid in seqs:
            n = int(self.attrs[sid]["n_frames"])
            failed = self._load_failed(sid)
            for s in range(0, n - self.T + 1, self.stride):
                if float(failed[s : s + self.T].mean()) > max_fail_frac:
                    dropped += 1
                    continue
                idx.append((sid, s))
        self.n_dropped_reg = dropped
        return idx

    def __len__(self) -> int:
        return len(self.index)

    # ---- 缓存载入 ----------------------------------------------------------
    def _load_seq(self, seq_id: int) -> dict:
        if seq_id in self._seq_cache:
            return self._seq_cache[seq_id]
        d = f"{self.cache_root}/seq_{seq_id:04d}"
        lab = np.load(f"{d}/labels.npz")
        self._seq_cache[seq_id] = {
            "frames": np.load(f"{d}/frames.u8.npy", mmap_mode="r"),
            "stats": np.load(f"{d}/norm_stats.npy"),
            "nuc": np.load(f"{d}/nuc_field.npy").astype(np.float32),
            "dead": np.load(f"{d}/deadpix.npy"),
            "boxes": np.asarray(lab["boxes"], np.int32),
            "track_ids": np.asarray(lab["track_ids"], np.int32),
            "reg": dict(np.load(f"{d}/reg.npz")) if self.align else None,
            "quality": np.load(f"{d}/quality.npy"),
        }
        return self._seq_cache[seq_id]

    def _load_window(self, seq: dict, start: int, Ws) -> np.ndarray:
        """[T,H,W] float32 归一化帧，已对齐到锚点坐标系（t=0 即锚点不重采样）。"""
        T = self.T
        raw = np.asarray(seq["frames"][start : start + T], dtype=np.uint8)
        out = np.empty((T, *raw.shape[1:]), dtype=np.float32)
        stats = seq["stats"][start : start + T]
        for t in range(T):
            x = correct_frame(raw[t], seq["nuc"], seq["dead"])
            out[t] = normalize_frame(x, float(stats[t, 0]), float(stats[t, 1]))
        if Ws is not None:
            for t in range(1, T):
                if not is_identity_warp(Ws[t]):  # 准恒等免重采样（防双重插值模糊）
                    out[t] = warp_frame(out[t], Ws[t])
        return out

    # ---- §4.7 裁剪可行性采样（四档退化，档名随样本落盘可审计） -------------
    def _sample_crop(self, seq: dict, start: int, Ws, rects, rng: np.random.Generator):
        """在"≥1 完整 GT 框（含 crop_margin 边距）"约束下均匀取裁剪起点。

        rects 由调用方给出（同一批锚点系并集矩形还要用于覆盖率记账）。

        §4.7 的字面要求是"≥1 完整框"，本实现按强到弱分档，退化如实记账：
          all_tracks  窗内**每条**轨迹的整窗轨迹都能完整装下且可行区间有交
                      → 一次裁剪使所有目标全程不被裁半（最强，无边界偏差）；
          one_track   至少一条轨迹整窗可容纳（横跨全幅的长轨迹装不下 ⇒ 会退化到此，
                      真实 ITTD 上这是主档，见 P0 目检报告的分档计数）；
          one_frame   无整轨迹可容纳时退到"至少一个 (轨迹,帧) 框完整"——仍满足 §4.7
                      字面约束，但该窗带边界漏检偏差，单列计数供裁决；
          negative    窗内无 GT（合法负样本窗，任意偏移）；
          fallback    连单帧框都装不下 → 交回上层换窗重采（§4.7"否则重采"）。
        """
        H, W = seq["frames"].shape[1:3]
        if self.crop is None or tuple(self.crop) == (H, W):
            return 0, 0, (H, W), "no_crop"
        ch, cw = self.crop
        y_lim, x_lim = H - ch, W - cw
        self.crop_stats["attempts"] += 1
        if not rects:  # 合法负样本窗：无 GT 可保
            self.crop_stats["negative"] += 1
            return (int(rng.integers(0, y_lim + 1)), int(rng.integers(0, x_lim + 1)),
                    (ch, cw), "negative")

        def interval(lo, hi, size, limit):
            """[lo,hi)（锚点系、hi 不含）加 margin 完整落入 size 的起点区间；空则 None。"""
            a = max(int(np.ceil(hi + self.crop_margin - size)), 0)
            b = min(int(np.floor(lo - self.crop_margin)), limit)
            return (a, b) if a <= b else None

        def emit(iy, ix, tier):
            self.crop_stats[tier] += 1
            return (int(rng.integers(iy[0], iy[1] + 1)), int(rng.integers(ix[0], ix[1] + 1)),
                    (ch, cw), tier)

        per_track = []
        for _tid, x1, y1, x2, y2, _n in rects:
            iy, ix = interval(y1, y2, ch, y_lim), interval(x1, x2, cw, x_lim)
            if iy and ix:
                per_track.append((iy, ix))
        if per_track:
            ay = (max(iv[0] for iv, _ in per_track), min(iv[1] for iv, _ in per_track))
            ax = (max(iv[0] for _, iv in per_track), min(iv[1] for _, iv in per_track))
            if len(per_track) == len(rects) and ay[0] <= ay[1] and ax[0] <= ax[1]:
                return emit(ay, ax, "all_tracks")
            return emit(*per_track[int(rng.integers(0, len(per_track)))], "one_track")

        # 单帧级可行性：逐 (轨迹,帧) 框各自进锚点系判一次
        if len(seq["boxes"]):
            fno = seq["boxes"][:, 0].astype(int)
            sel = (fno >= start + 1) & (fno < start + self.T + 1)
            for f, x1, y1, x2, y2 in seq["boxes"][sel]:
                W_t = _IDENT_W if Ws is None else Ws[int(f) - 1 - start]
                ax1, ay1, ax2, ay2 = _to_anchor_rect(W_t, x1, y1, x2, y2)
                iy, ix = interval(ay1, ay2, ch, y_lim), interval(ax1, ax2, cw, x_lim)
                if iy and ix:
                    return emit(iy, ix, "one_frame")
        self.crop_stats["fallback"] += 1
        return (int(rng.integers(0, y_lim + 1)), int(rng.integers(0, x_lim + 1)),
                (ch, cw), "fallback")

    # ---- 增强（§2.6 全表；几何整窗一致，模糊/复制/注入逐帧独立） -----------
    def _geom(self, win: np.ndarray, mask: np.ndarray, rng):
        """整窗一致几何算子（翻转 / 180°(90° 仅方形) / 小幅仿射）；掩码同治。"""
        T, H, W = win.shape
        if rng.random() < 0.5:
            win, mask = win[:, :, ::-1].copy(), mask[:, :, ::-1].copy()
        if rng.random() < 0.5:
            win, mask = win[:, ::-1, :].copy(), mask[:, ::-1, :].copy()
        if rng.random() < 0.5:
            k = 2 if H != W else int(rng.integers(1, 4))
            win, mask = np.rot90(win, k, axes=(1, 2)).copy(), np.rot90(mask, k, axes=(1, 2)).copy()
        if rng.random() < 0.5:  # 平移 ±2px / 旋转 ±0.5° / 缩放 ±2%
            M = cv2.getRotationMatrix2D((W / 2, H / 2), float(rng.uniform(-0.5, 0.5)),
                                        float(rng.uniform(0.98, 1.02)))
            M[0, 2] += float(rng.uniform(-2, 2))
            M[1, 2] += float(rng.uniform(-2, 2))
            # 此处 M 为 src→dst 正向（warpAffine 默认语义），与配准 W 的 INVERSE_MAP 相反
            win = np.stack([cv2.warpAffine(f, M, (W, H)) for f in win])
            mask = np.stack([cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_NEAREST) for f in mask])
        return win, mask

    def _per_frame_blur(self, win: np.ndarray, rng):
        """逐帧独立退化（模拟平台晃动真实物理）：运动模糊核 3–9px 或散焦 σ0.5–1.5px，p=0.3。"""
        if rng.random() >= 0.3:
            return win
        for t in range(win.shape[0]):
            if rng.random() < 0.5:
                L, ang = int(rng.integers(3, 10)), float(rng.uniform(0, np.pi))
                k = np.zeros((L, L), np.float32)
                for s in np.linspace(-(L // 2), L // 2, L):
                    xx = int(round(L // 2 + s * np.cos(ang)))
                    yy = int(round(L // 2 + s * np.sin(ang)))
                    if 0 <= xx < L and 0 <= yy < L:
                        k[yy, xx] += 1.0
                k /= k.sum()
                win[t] = cv2.filter2D(win[t], -1, k, borderType=cv2.BORDER_REPLICATE)
            else:
                win[t] = cv2.GaussianBlur(win[t], (0, 0), float(rng.uniform(0.5, 1.5)))
        return np.clip(win, 0.0, 1.0)  # 防御：后续 gamma 幂的底数非负

    def _frame_copy(self, win: np.ndarray, mask: np.ndarray, rng):
        """帧复制（短时停滞/遮挡）：随机 1–2 帧替换为前一帧，p=0.3；掩码同治。"""
        T = win.shape[0]
        if T < 2 or rng.random() >= 0.3:
            return win, mask
        for _ in range(int(rng.integers(1, 3))):
            t = int(rng.integers(1, T))
            win[t], mask[t] = win[t - 1].copy(), mask[t - 1].copy()
        return win, mask

    def _photometric(self, win: np.ndarray, sigma_raw: float, rng):
        """整窗一致光度（gamma 0.7–1.4 / gain 0.8–1.2 / bias ±0.05）+ 灰度级噪声 σ0–3。"""
        if rng.random() < 0.5:
            win = win ** float(rng.uniform(0.7, 1.4))
        win = win * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.05, 0.05))
        g = float(rng.uniform(0, 3))  # 唯一需要 σ_raw 换算的算子（归一化域 1 灰度级 = 1/(16σ_raw)）
        if g > 0:
            win = win + rng.normal(0, g / (16.0 * max(sigma_raw, 1.0)), win.shape).astype(np.float32)
        return np.clip(win, 0.0, 1.0)

    @staticmethod
    def _blob(win: np.ndarray, t: int, y: int, x: int, amp: float, sigma_g: float):
        """在帧 t 的 (y,x) 贴一个类高斯**增量**亮点（峰值=amp，结果钳在 [0,1]）。"""
        H, W = win.shape[1:3]
        r = max(2, int(np.ceil(3.0 * sigma_g)))
        y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, H)
        x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, W)
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        win[t, y0:y1, x0:x1] = np.clip(
            win[t, y0:y1, x0:x1] + amp * np.exp(-d2 / (2.0 * sigma_g ** 2)), 0.0, 1.0)

    def _inject(self, win: np.ndarray, mask: np.ndarray, rng):
        """合成注入（瞬时亮点 / 缓变热斑）——只在**非目标区**落点。

        落点禁区 = 掩码膨胀(teacher_dilate+2)：若把可控负样本砸在真目标上，同一位置
        既被 L_seg 拉成正例又被门控当成虚警负例，监督自我矛盾（R7 的另一面）。
        注入在帧复制/光度之后进行，故幅值即 §2.6 口径（μ+4σ~μ+8σ → 增量 0.25~0.50）。
        """
        T, H, W = win.shape
        # 时域并集后膨胀 = 禁落区：膨胀半径取 teacher_dilate + 3σ_G,max(≈6)+2，
        # 使高斯拖尾到真目标膨胀区(3px)时已 <1% 幅值——注入是"孤立瞬现亮点"负样本，
        # 落在真目标上就不再是负样本，且会把 L_seg 的正例像素糊成不可控形状。
        excl = dilate_mask(mask.max(axis=0), self.teacher_dilate + 8) > 0
        assert excl.shape == (H, W), f"禁落区形状 {excl.shape} 与帧 {H}×{W} 不符"

        def free_spot():
            for _ in range(20):
                y, x = int(rng.integers(0, H)), int(rng.integers(0, W))
                if not excl[y, x]:
                    return y, x
            return None

        if self.highlight and rng.random() < 0.4:  # p=0.4/窗、1–3 个、存活 1–3 帧
            for _ in range(int(rng.integers(1, 4))):
                spot = free_spot()
                if spot is None:
                    break
                dur = int(rng.integers(1, 4))
                t0 = int(rng.integers(0, max(1, T - dur + 1)))
                amp = _level(rng.uniform(4.0, 8.0)) - 0.5
                sg = float(rng.uniform(0.8, 2.0))
                for t in range(t0, min(t0 + dur, T)):
                    self._blob(win, t, spot[0], spot[1], amp, sg)
        if self.hotspot and rng.random() < self.hotspot_p:  # 30 帧内升至 μ+5σ 后维持
            spot = free_spot()
            if spot is not None:
                sg = float(rng.uniform(4.0, 10.0))
                ramp0 = int(rng.integers(0, 8))
                amp = _level(5.0) - 0.5
                for t in range(T):
                    frac = min(1.0, max(0.0, (t - ramp0) / 30.0))
                    if frac > 0:
                        self._blob(win, t, spot[0], spot[1], amp * frac, sg)
        return win

    # ---- 取一个样本 --------------------------------------------------------
    def _pick(self, idx: int, rng: np.random.Generator) -> tuple[int, int]:
        """(seq_id, start)：时间窗随机起点（§2.6 允许项，在步距 8 的抖动范围内）。"""
        sid, base = self.index[idx]
        n = int(self.attrs[sid]["n_frames"])
        room = min(self.stride - 1, n - self.T - base)
        start = base + (int(rng.integers(0, room + 1)) if (self.start_jitter and room > 0) else 0)
        return sid, start

    def __getitem__(self, idx: int) -> dict:
        # 逐样本确定性 RNG：多 worker 不重复、训练可复现（规避 DataLoader 共享 rng 问题）
        rng = np.random.default_rng([self.seed, idx])
        item, ok = None, False
        for attempt in range(self.max_crop_attempts):
            if attempt:  # §4.7"否则重采"：裁剪不可行时换窗
                cur = int(rng.integers(0, len(self.index)))
            else:
                cur = idx
            sid, start = self._pick(cur, rng)
            item, ok = self._make(sid, start, rng)
            if ok:
                return item
        return item  # 全部尝试均不可行：交回最后一次样本（crop_tier=fallback 可被上游计数）

    def _make(self, sid: int, start: int, rng: np.random.Generator):
        seq = self._load_seq(sid)
        H, W = seq["frames"].shape[1:3]
        Ws = window_anchor_warps(seq["reg"], start, self.T) if self.align else None
        win = self._load_window(seq, start, Ws)
        mask = gt_masks_anchor(seq["boxes"], start, self.T, Ws, (H, W))
        sigma_raw = float(np.median(seq["stats"][start : start + self.T, 1]))
        quality = float(np.median(seq["quality"][start : start + self.T]))

        rects = _track_rects(seq["boxes"], seq["track_ids"], start, self.T, Ws)
        y0, x0, (ch, cw), tier = self._sample_crop(seq, start, Ws, rects, rng)
        win = win[:, y0 : y0 + ch, x0 : x0 + cw]
        mask = mask[:, y0 : y0 + ch, x0 : x0 + cw]
        clipped = self._gt_clipped(mask, tier)
        # 裁剪覆盖率（P0 报告口径）：本窗共几条轨迹、其中几条**整窗轨迹**完整含边距。
        # §4.7 只保证 ≥1 条，其余被裁半的条数必须可见，否则边界漏检偏差无从归因。
        n_tot, n_ok = len(rects), 0
        for _tid, rx1, ry1, rx2, ry2, _n in rects:
            if (y0 <= ry1 - self.crop_margin and y0 + ch >= ry2 + self.crop_margin
                    and x0 <= rx1 - self.crop_margin and x0 + cw >= rx2 + self.crop_margin):
                n_ok += 1

        if self.augment:
            win, mask = self._geom(win, mask, rng)
            win = self._per_frame_blur(win, rng)
            win, mask = self._frame_copy(win, mask, rng)
            win = self._photometric(win, sigma_raw, rng)
            win = self._inject(win, mask, rng)

        tgt = mask[:, None, :, :]
        target = downsample_max(tgt, self.fs) if self.fs > 1 else tgt
        if self.teacher_dilate > 0:
            dil = dilate_mask(tgt, self.teacher_dilate)
            teacher = downsample_max(dil, self.fs) if self.fs > 1 else dil
        else:
            teacher = target.copy()
        a = self.attrs[sid]
        tags = a["difficulty_tags"]
        return {
            "windows": np.ascontiguousarray(win[:, None, :, :], np.float32),
            "target": np.ascontiguousarray(target, np.float32),
            "teacher_mask": np.ascontiguousarray(teacher, np.float32),
            "quality": quality,
            "seq_id": int(sid),
            "start0": int(start),
            "crop_offset": (int(y0), int(x0)),
            "crop_tier": tier,
            "crop_cover": (int(n_ok), int(n_tot)),
            "gt_clipped": bool(clipped),
            "scene": str(a["scene"]["value"]),
            "daytime": str(a["daytime"]["value"]),
            "strong_clutter": bool(tags["strong_clutter"]["value"]),
            "distractor_present": bool(tags["distractor_present"]["value"]),
        }, tier != "fallback"

    @staticmethod
    def _gt_clipped(mask: np.ndarray, tier: str) -> bool:
        """裁剪后是否仍有 GT 贴到图像边界（贴边 = 该目标被裁半；负样本窗恒 False）。"""
        if tier in ("negative",) or mask.size == 0 or float(mask.max()) < 1:
            return False
        return bool(mask[:, 0].any() or mask[:, -1].any()
                    or mask[:, :, 0].any() or mask[:, :, -1].any())

    # ---- 供 QC / 快评追溯 --------------------------------------------------
    def window_info(self, idx: int) -> dict:
        sid, start = self.index[idx]
        seq = self._load_seq(sid)
        rects = _track_rects(seq["boxes"], seq["track_ids"], start, self.T, None)
        return {"seq_id": sid, "start": start, "n_tracks": len(rects),
                "n_gt_boxes": int(sum(r[5] for r in rects))}
