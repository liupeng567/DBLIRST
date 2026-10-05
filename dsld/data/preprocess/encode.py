"""2.7 标签编码：中心热图 / 框+轨迹索引 / 实例掩码 RLE。

缓存（2.8）：
  labels.npz : boxes [N,5](frame,x1,y1,x2,y2 int32) + track_ids [N] int32 +
               track_index（{track_id: [帧号]} 的紧凑数组版）
  heatmaps.u8.npy : [N,H,W] uint8，中心 σ=1.5px 高斯核（量化 0-255，峰值 255）
  masks.npz : 逐实例掩码 RLE（instance_ids PNG == track_id 与框取交；
              suppressed 或空 → 框填充，box_fill 标记）

框回归目标 (l,t,r,b)+centerness 不落缓存——由 boxes 在线推导（M2 窗口采样器），省 4× 空间。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HEATMAP_SIGMA = 1.5
HEATMAP_RADIUS = 6  # 4σ 截断
RLE_DTYPE = np.uint32


def gaussian_peak(heatmap: np.ndarray, cx: float, cy: float) -> None:
    """在 heatmap 上叠加单中心高斯（局部 patch，峰值 255）。"""
    h, w = heatmap.shape
    x0, x1 = max(0, int(cx) - HEATMAP_RADIUS), min(w, int(cx) + HEATMAP_RADIUS + 1)
    y0, y1 = max(0, int(cy) - HEATMAP_RADIUS), min(h, int(cy) + HEATMAP_RADIUS + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    patch = np.exp(-d2 / (2 * HEATMAP_SIGMA ** 2))
    np.maximum(heatmap[y0:y1, x0:x1], (patch * 255).astype(np.uint8), out=heatmap[y0:y1, x0:x1])


def build_heatmap_frame(boxes: np.ndarray) -> np.ndarray:
    """单帧热图。boxes: [K,4] (x1,y1,x2,y2)。"""
    hm = np.zeros((480, 640), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        gaussian_peak(hm, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return hm


def build_track_index(boxes: np.ndarray, track_ids: np.ndarray) -> dict[int, list[int]]:
    """{track_id: [帧号升序]}（轨迹段索引，供时序一致性损失）。"""
    idx: dict[int, list[int]] = {}
    for f, tid in zip(boxes[:, 0], track_ids):
        idx.setdefault(int(tid), [])
        if not idx[int(tid)] or idx[int(tid)][-1] != int(f):
            idx[int(tid)].append(int(f))
    return idx


def track_index_to_arrays(idx: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """紧凑存储：ids [T]、offsets [T+1]、frames 拼接数组。"""
    ids = np.array(sorted(idx), dtype=np.int32)
    lens = [len(idx[int(t)]) for t in ids]
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    frames = np.concatenate([idx[int(t)] for t in ids]).astype(np.int32) if len(ids) else np.zeros(0, np.int32)
    return ids, offsets, frames


def rle_encode(mask: np.ndarray) -> np.ndarray:
    """行优先 RLE（交替游程，首游程记 0 值长度）。返回 uint32 [R]。"""
    flat = mask.ravel(order="C")
    if flat.size == 0 or flat.max() == 0:
        return np.zeros(1, dtype=RLE_DTYPE)
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds)
    if flat[0] == 1:
        runs = np.concatenate([[0], runs])
    return runs.astype(RLE_DTYPE)


def rle_decode(runs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    pos, val = 0, 0
    for r in runs:
        if val:
            flat[pos : pos + int(r)] = 1
        pos += int(r)
        val ^= 1
    return flat.reshape(shape, order="C")


def load_instance_masks(
    seg_root: str | Path,
    seq_id: int,
    frames: np.ndarray,
    track_ids: np.ndarray,
    boxes: np.ndarray,
    suppressed: np.ndarray,
) -> dict[str, np.ndarray]:
    """从 seg_dataset 提取逐实例掩码（与框取交），suppressed 回退框填充。

    frames/track_ids/boxes: labels 的实例级数组（每实例一行）。
    suppressed: [N] bool（来自 instances JSON；PNG 无像素的压制实例）。
    返回 masks.npz 的数组字典。
    """
    seg_root = Path(seg_root)
    h, w = 480, 640
    n = len(frames)
    rles: list[np.ndarray] = []
    box_fill = np.zeros(n, dtype=bool)
    # 按 frame 分组，PNG 每帧只读一次
    by_frame: dict[int, list[int]] = {}
    for i, f in enumerate(frames):
        by_frame.setdefault(int(f), []).append(i)
    for f, rows in by_frame.items():
        png_path = seg_root / "instance_ids" / str(seq_id) / f"{f:03d}.png"
        if png_path.exists():
            data = np.fromfile(str(png_path), dtype=np.uint8)
            png = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        else:
            png = None
        for i in rows:
            x1, y1, x2, y2 = boxes[i, 1:5]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
            box_mask = np.zeros((h, w), dtype=np.uint8)
            if x2 >= x1 and y2 >= y1:
                box_mask[y1 : y2 + 1, x1 : x2 + 1] = 1
            m = np.zeros((h, w), dtype=np.uint8)
            if png is not None and not suppressed[i]:
                m = ((png == track_ids[i]) & (box_mask > 0)).astype(np.uint8)
            if m.sum() == 0:  # 压制实例/无像素 → 框填充（方案 10.9 约定②）
                m = box_mask
                box_fill[i] = True
            rles.append(rle_encode(m))
    runs = np.concatenate(rles) if rles else np.zeros(0, dtype=RLE_DTYPE)
    lens = np.array([len(r) for r in rles], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return {
        "frames": frames.astype(np.int32),
        "track_ids": track_ids.astype(np.int32),
        "box_fill": box_fill,
        "rle_offsets": offsets,
        "rle_runs": runs.astype(RLE_DTYPE),
        "shape": np.array([h, w], dtype=np.int32),
    }
