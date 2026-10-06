"""ITTD 真实窗口数据集（方案 2.5 时序采样 / 2.6 时间一致性增强 / 2.7 标签编码）。

M2 基线范围（8.1 框级评测口径下两基线共用）：
  - mode="frame"  单帧样本 [1,1,H,W]，供 MSHNet 单帧基线（整窗一致约束退化为逐帧）；
  - mode="window" 时序窗口 [T,1,H,W]，供多尺度帧差 + 3D conv 时序基线。

数据来源：M1 缓存 data/cache/ittd/seq_XXXX（2.8 布局）。原始帧 uint8 在线走
M1 同一套校正 + 归一化（normalize.correct_frame / normalize_frame，norm_stats 为
校正后域统计），保证与 QC 口径逐位一致。

标签目标：框填充掩码 M_box（方案 2.7-②：目标极小，填充优于分割轮廓）；
空标注帧为合法负样本（10.9 约定：全 0 掩码，不剔除）。

增强（2.6，M2 基线子集）：几何算子整窗一致（翻转/90°/小幅仿射/裁剪），
逐帧独立退化仅运动模糊/散焦；合成亮点注入属 DSLD 特色训练，M2 基线不启用
（configs 中 aug.highlight_inject: false，M6 消融再开）。
"""

from __future__ import annotations

import cv2
import numpy as np
from torch.utils.data import Dataset

from dsld.data.manifest import load_manifest
from dsld.data.preprocess.normalize import CLIP, correct_frame, normalize_frame


def _build_index(
    manifest: dict, split: str, cache_root, T: int, stride: int, mode: str
) -> list[tuple[int, int, int]]:
    """返回 [(seq_id, n_frames, start0)]，start0 为 0-based 窗口起点。

    frame 模式：start0 即帧索引，每帧一个样本；window 模式：起点按 stride 步进，
    不足 T 的尾部丢弃（推理期另行补尾窗，训练期靠随机起点覆盖）。
    """
    seqs = manifest["splits"][split]["seqs"]
    by_id = {s["seq_id"]: s for s in manifest["sequences"]}
    index = []
    for sid in seqs:
        n = int(by_id[sid]["n_frames"])
        if mode == "frame":
            index.extend((sid, n, f) for f in range(n))
        else:
            index.extend((sid, n, s) for s in range(0, n - T + 1, stride))
    return index


class IttdWindows(Dataset):
    """ITTD 缓存窗口数据集（训练用；推理走 dsld.eval.infer_seq 逐段滑窗）。"""

    def __init__(
        self,
        manifest_path: str,
        cache_root: str,
        split: str = "train-int",
        mode: str = "window",
        T: int = 32,
        stride: int = 8,
        crop: tuple[int, int] | None = None,
        augment: bool = True,
        seed: int = 0,
        limit_seqs: int = 0,
    ):
        assert mode in ("frame", "window")
        self.manifest = load_manifest(manifest_path)
        self.cache_root = cache_root
        self.split = split
        self.mode = mode
        self.T = 1 if mode == "frame" else T
        self.crop = tuple(crop) if crop else None
        self.augment = augment
        self.index = _build_index(self.manifest, split, cache_root, self.T, stride, mode)
        if limit_seqs:  # dry-run/冒烟用：只取前 N 段
            keep = set(self.manifest["splits"][split]["seqs"][:limit_seqs])
            self.index = [t for t in self.index if t[0] in keep]
        self.seed = seed
        self._seq_cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.index)

    # ---- 缓存载入 ----------------------------------------------------------
    def _load_seq(self, seq_id: int) -> dict:
        if seq_id in self._seq_cache:
            return self._seq_cache[seq_id]
        d = f"{self.cache_root}/seq_{seq_id:04d}"
        nuc = np.load(f"{d}/nuc_field.npy").astype(np.float32)
        dead = np.load(f"{d}/deadpix.npy")
        self._seq_cache[seq_id] = {
            "frames": np.load(f"{d}/frames.u8.npy", mmap_mode="r"),
            "stats": np.load(f"{d}/norm_stats.npy"),
            "nuc": nuc,
            "dead": dead,
            "labels": np.load(f"{d}/labels.npz"),
        }
        return self._seq_cache[seq_id]

    def _load_frames(self, seq: dict, start: int, T: int) -> np.ndarray:
        """[T,480,640] float32 归一化帧（M1 校正 + 逐帧 median/MAD，[0,1] 域）。"""
        raw = np.asarray(seq["frames"][start : start + T], dtype=np.uint8)
        out = np.empty((T, *raw.shape[1:]), dtype=np.float32)
        stats = seq["stats"][start : start + T]
        for t in range(T):
            x = correct_frame(raw[t], seq["nuc"], seq["dead"])
            out[t] = normalize_frame(x, float(stats[t, 0]), float(stats[t, 1]))
        return out

    def _load_mask(self, seq: dict, start: int, T: int, H: int, W: int) -> np.ndarray:
        """框填充掩码 [T,H,W] float32（0-based 帧索引 = 标注帧号−1）。"""
        boxes = seq["labels"]["boxes"]
        m = np.zeros((T, H, W), dtype=np.float32)
        lo, hi = start + 1, start + T + 1  # 标注帧号 1-based
        in_win = boxes[(boxes[:, 0] >= lo) & (boxes[:, 0] < hi)]
        for f, x1, y1, x2, y2 in in_win:
            m[f - 1 - start, y1 : y2 + 1, x1 : x2 + 1] = 1.0
        return m

    # ---- 增强（2.6；几何整窗一致，模糊逐帧独立） ---------------------------
    def _augment(self, win: np.ndarray, mask: np.ndarray, sigma_raw: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        T, H, W = win.shape

        # 几何：整窗一致
        if rng.random() < 0.5:
            win = win[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
        if rng.random() < 0.5:
            win = win[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        if rng.random() < 0.5:
            # 方形输入允许 90°/180°/270°；非方形（ITTD 480×640）退化为 180°，
            # 否则旋转后 H/W 互换无法成批（2.6 允许 90° 的约定按方形数据解释）
            k = 2 if H != W else int(rng.integers(1, 4))
            win = np.rot90(win, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        if rng.random() < 0.5:  # 小幅仿射：平移±2px / 旋转±0.5° / 缩放±2%
            c, a = (W / 2, H / 2), float(rng.uniform(-0.5, 0.5))
            s = float(rng.uniform(0.98, 1.02))
            M = cv2.getRotationMatrix2D(c, a, s)
            M[0, 2] += float(rng.uniform(-2, 2))
            M[1, 2] += float(rng.uniform(-2, 2))
            win = np.stack([cv2.warpAffine(f, M, (W, H)) for f in win])
            mask = np.stack(
                [cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_NEAREST) for f in mask]
            )

        # 逐帧独立退化：运动模糊核 3–9 px 或散焦 σ∈[0.5,1.5] px，p=0.3
        if rng.random() < 0.3:
            for t in range(T):
                if rng.random() < 0.5:  # 方向运动模糊：沿方向的均匀采样核（非负、和为 1）
                    L, ang = int(rng.integers(3, 10)), rng.uniform(0, np.pi)
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
            win = np.clip(win, 0.0, 1.0)  # 防御：保证后续 gamma 幂运算底数非负

        # 光度：整窗一致 gamma / gain / bias
        if rng.random() < 0.5:
            win = win ** float(rng.uniform(0.7, 1.4))
        win = win * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.05, 0.05))
        win = np.clip(win, 0.0, 1.0)

        # 加性高斯噪声 σ∈[0,3] 灰度级（8bit 语义 → 归一化域换算：x 域 1 灰度级 ≈ 1/(16·σ_raw)）
        sigma_gray = float(rng.uniform(0, 3))
        if sigma_gray > 0:
            win = win + rng.normal(0, sigma_gray / (16.0 * max(sigma_raw, 1.0)), win.shape).astype(np.float32)
            win = np.clip(win, 0.0, 1.0)

        # 帧复制（模拟短时停滞/遮挡），p=0.3，仅时序模式
        if T > 1 and rng.random() < 0.3:
            for _ in range(int(rng.integers(1, 3))):
                t = int(rng.integers(1, T))
                win[t] = win[t - 1]
                mask[t] = mask[t - 1]
        return win.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        # 逐样本确定性 RNG：多 worker 不重复、训练可复现（DataLoader worker 复制共享 rng 的问题规避）
        rng = np.random.default_rng([self.seed, idx])
        seq_id, n_frames, start = self.index[idx]
        seq = self._load_seq(seq_id)
        H, W = seq["frames"].shape[1:3]

        win = self._load_frames(seq, start, self.T)
        mask = self._load_mask(seq, start, self.T, H, W)
        sigma_raw = float(np.median(seq["stats"][start : start + self.T, 1]))

        if self.crop is not None and tuple(self.crop) != (H, W):
            ch, cw = self.crop
            y0 = int(rng.integers(0, H - ch + 1))
            x0 = int(rng.integers(0, W - cw + 1))
            win = win[:, y0 : y0 + ch, x0 : x0 + cw]
            mask = mask[:, y0 : y0 + ch, x0 : x0 + cw]

        if self.augment:
            win, mask = self._augment(win, mask, sigma_raw, rng)

        x = win[:, None, :, :]  # [T,1,H,W]
        return {
            "windows": x,
            "target": mask[:, None, :, :],
            "seq_id": seq_id,
            "start0": start,
        }


def noise_sigma_domain(sigma_gray: float, stats: np.ndarray) -> float:
    """灰度级噪声 σ → 归一化域：x 域 1 灰度级 ≈ 1/(16·σ_raw)，σ_raw 取帧间中位。"""
    sigma_raw = float(np.median(stats[:, 1]))
    return sigma_gray / (16.0 * max(sigma_raw, 1.0))
