"""序列级推理（M2 eval 全链：缓存 → 概率图 → 逐阈值预测框）。

推理约定（方案 2.5 / 3.4）：
  - 原生 640×480 分辨率，不缩放；
  - MSHNet 单帧：逐帧前向（批内多帧仅是加速），原始帧坐标（与 frame 模式训练一致）；
  - T-MSD3D 时序 / DSLD：滑窗 T=32、步距 24、窗尾补窗；每窗前 8 帧为预热，输出不
    参与**判决**；重叠覆盖帧取 sigmoid 概率平均（2.5"α 图平均"口径）；
  - 时序窗口与训练侧同口径对齐：每窗对齐到窗口首帧（锚点，align_to_anchor 复合
    reg.npz + WARP_INVERSE_MAP），模型输出（锚点坐标系）再按同一 W **回投原始帧
    坐标**后再做重叠平均与 GT 对比（输入侧 INVERSE_MAP 采样对齐 / 输出侧默认正向
    回投，同一矩阵的两次语义）。

归一化与训练侧同一实现（correct_frame + normalize_frame），无域偏移。
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from dsld.data.preprocess.normalize import correct_frame, normalize_frame
from dsld.data.preprocess.register import (
    is_identity_warp,
    warp_frame,
    window_anchor_warps,
)
from dsld.eval.mask_to_boxes import Box, mask_to_boxes


def load_seq_cache(cache_root: str, seq_id: int, with_reg: bool = False) -> dict:
    d = f"{cache_root}/seq_{seq_id:04d}"
    out = {
        "frames": np.load(f"{d}/frames.u8.npy", mmap_mode="r"),
        "stats": np.load(f"{d}/norm_stats.npy"),
        "nuc": np.load(f"{d}/nuc_field.npy").astype(np.float32),
        "dead": np.load(f"{d}/deadpix.npy"),
        "labels": np.load(f"{d}/labels.npz"),
    }
    if with_reg:  # 时序推理窗口对齐用（M1 口径：bridge_M 跨块复合）
        out["reg"] = dict(np.load(f"{d}/reg.npz"))
    return out


def normalize_seq(seq: dict) -> np.ndarray:
    """整段归一化 [N,H,W] float32（M1 同口径：坏点/NUC 校正 + 逐帧 median/MAD）。"""
    raw = np.asarray(seq["frames"])
    n = raw.shape[0]
    out = np.empty((n, *raw.shape[1:]), dtype=np.float32)
    for f in range(n):
        x = correct_frame(raw[f], seq["nuc"], seq["dead"])
        out[f] = normalize_frame(x, float(seq["stats"][f, 0]), float(seq["stats"][f, 1]))
    return out


@torch.no_grad()
def infer_mshnet(
    model, seq: dict, device: str, batch: int = 8
) -> np.ndarray:
    """单帧基线：返回逐帧 sigmoid 概率 [N,H,W]。"""
    frames = torch.from_numpy(normalize_seq(seq))
    model.eval()
    probs = []
    for i in range(0, len(frames), batch):
        x = frames[i : i + batch, None].to(device)  # [b,1,H,W]
        logits = model(x)
        probs.append(torch.sigmoid(logits.float())[:, 0].cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def infer_temporal(
    model, seq: dict, device: str, T: int = 32, stride: int = 24, warmup: int = 8,
    align: bool = True,
) -> np.ndarray:
    """时序基线：滑窗推理 + 预热排除 + 重叠平均。返回 [N,H,W]（前 warmup 帧为 0）。

    align=True（默认）：每窗输入先对齐到窗口首帧锚点（与训练侧 IttdWindows 同口径），
    输出概率图再回投原始帧坐标后参与重叠平均——回投 = warpAffine(同 W，默认正向)。
    """
    reg = seq.get("reg") if align else None
    frames = torch.from_numpy(normalize_seq(seq))
    n = len(frames)
    model.eval()
    prob_sum = np.zeros((n, *frames.shape[1:]), dtype=np.float32)
    prob_cnt = np.zeros((n,), dtype=np.int32)

    starts = list(range(0, n - T + 1, stride))
    if not starts or starts[-1] != n - T:
        starts.append(n - T)  # 尾窗补齐（覆盖末尾帧）
    for s in starts:
        Ws = window_anchor_warps(reg, s, T) if reg is not None else None
        win = frames[s : s + T].numpy()
        if Ws is not None:
            for j in range(1, T):
                if not is_identity_warp(Ws[j]):
                    win[j] = warp_frame(win[j], Ws[j])
        x = torch.from_numpy(win[:, None])[None]  # [1,T,1,H,W]
        out = model(x.to(device))
        logits = out["logits"] if isinstance(out, dict) else out
        prob = torch.sigmoid(logits.float()[0, :, 0]).cpu().numpy()  # [T,H,W]
        if Ws is not None:
            # 输出回投原始帧坐标：回投与输入对齐同一 W（默认正向语义）
            for j in range(warmup, T):
                if not is_identity_warp(Ws[j]):
                    h, w = prob[j].shape
                    prob[j] = cv2.warpAffine(
                        prob[j], np.asarray(Ws[j], np.float32), (w, h),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                    )
        valid = prob[warmup:]  # 前 warmup 帧预热不判决
        prob_sum[s + warmup : s + T] += valid
        prob_cnt[s + warmup : s + T] += 1
    out = np.zeros_like(prob_sum)
    m = prob_cnt > 0
    out[m] = prob_sum[m] / prob_cnt[m, None, None]
    return out


def boxes_at_thresholds(
    prob: np.ndarray, thr_list: list[float], min_area: int = 4, frame_offset: int = 1
) -> dict[float, list[Box]]:
    """整段概率图 → 各阈值下预测框（帧号 1-based，与标注同构）。"""
    out: dict[float, list[Box]] = {t: [] for t in thr_list}
    for i in range(prob.shape[0]):
        for t in thr_list:
            out[t].extend(mask_to_boxes(prob[i], frame=i + frame_offset, thr=t, min_area=min_area))
    return out


def gt_boxes_from_cache(seq: dict) -> list[Box]:
    """缓存 labels.npz → GT 框列表（score=1，保留 track_id）。"""
    lab = seq["labels"]
    out = []
    for (f, x1, y1, x2, y2), tid in zip(lab["boxes"], lab["track_ids"]):
        out.append(Box(frame=int(f), x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                       score=1.0, track_id=int(tid)))
    return out
