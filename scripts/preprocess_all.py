"""M1: ITTD 全量预处理与缓存构建（方案 2.2–2.8）。

单序列流水线：
  BMP 解码 → frames.u8.npy(memmap) → 时序中值 → 坏点表 + NUC 平场
  → 逐帧 median/MAD + 标准化质检量 → Harris+KLT+RANSAC 配准 reg.npz
  → labels.npz + heatmaps.u8.npy + masks.npz（SAM 实例掩码 RLE）
  → seq_meta.json（逐段统计：配准/非零占比/SCR/杂波密度/模糊占比，供 QC 与 manifest v3）

用法：
  python scripts/preprocess_all.py --seqs 1 2 3 4      # 冒烟
  python scripts/preprocess_all.py                     # 全量 87 段（默认 8 进程）
  python scripts/preprocess_all.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.ittd_parse import load_seq_annotation  # noqa: E402
from dsld.data.preprocess.encode import (  # noqa: E402
    build_heatmap_frame,
    build_track_index,
    load_instance_masks,
    track_index_to_arrays,
)
from dsld.data.preprocess.normalize import (  # noqa: E402
    correct_frame,
    detect_dead_pixels,
    frame_stats,
    nuc_field,
    temporal_median,
)
from dsld.data.preprocess.register import register_sequence  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# 路径可被环境变量覆盖（云端 Linux 重预处理用；默认保持本机 D:\ 路径）
ITTD_ROOT = Path(os.environ.get(
    "DSLD_ITTD_ROOT", r"D:\Datasets\面向空地应用的红外时敏目标检测跟踪数据集"))
SEG_ROOT = Path(os.environ.get(
    "DSLD_SEG_ROOT", r"D:\Datasets\kongdixiaomubiaodataset\seg_dataset"))
H, W, N_FRAMES = 480, 640, 250
FRAMES_PER_SEQ = N_FRAMES

# 统计口径（方案 1.4，M1 回填）
LONG_OCC_GAP = 20        # 连续消失 ≥ 20 帧
SMALL_DIAG_PX = 8.0      # 框对角线 < 8 px
SMALL_FRAC_GATE = 0.5    # 占比 > 50%
BLUR_RATIO_GATE = 0.10   # 模糊帧占比 > 10%（与官方晃动取并集）
SCR_RING_PAD = 8         # SCR 背景环宽（px）


def decode_bmp_gray(data: np.ndarray) -> np.ndarray | None:
    """8bit 灰度 BMP 直读（校验调色板恒等 + 无压缩 + 尺寸），失败返回 None 回退 imdecode。"""
    if len(data) < 1078 or data[0] != 0x42 or data[1] != 0x4D:  # "BM"
        return None
    off = int.from_bytes(data[10:14], "little")
    w = int.from_bytes(data[18:22], "little")
    h_s = int.from_bytes(data[22:26], "little")
    bpp = int.from_bytes(data[28:30], "little")
    comp = int.from_bytes(data[30:34], "little")
    if bpp != 8 or comp != 0 or off != 1078:
        return None
    pal = data[54:1078].reshape(256, 4)
    if not ((pal[:, 0] == np.arange(256)).all() and (pal[:, 1] == pal[:, 0]).all()
            and (pal[:, 2] == pal[:, 0]).all()):
        return None
    h = abs(h_s)
    px = data[1078:]
    if len(px) < w * h:
        return None
    img = np.frombuffer(px[: w * h], np.uint8).reshape(h, w)
    return np.ascontiguousarray(img[::-1]) if h_s > 0 else np.ascontiguousarray(img)


def read_frames(seq_id: int) -> np.ndarray:
    """解码一段序列全部 BMP → uint8 [250,480,640]。

    直读快路径 + 6 线程预取（fromfile 释放 GIL，重叠文件 I/O 与解码；
    并行运行时文件级 I/O 竞争是主要瓶颈）。
    """
    from concurrent.futures import ThreadPoolExecutor

    img_dir = ITTD_ROOT / "Images" / str(seq_id)

    def _one(f: int) -> np.ndarray:
        data = np.fromfile(str(img_dir / f"{f:03d}.bmp"), dtype=np.uint8)
        img = decode_bmp_gray(data)
        if img is None:
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"解码失败: {img_dir / f'{f:03d}.bmp'}")
        return img

    with ThreadPoolExecutor(max_workers=6) as ex:
        imgs = list(ex.map(_one, range(1, N_FRAMES + 1)))
    return np.stack(imgs)


def clutter_density(xf: np.ndarray, boxes_f: np.ndarray) -> float:
    """背景局部极大值密度（个/千像素）：3×3 邻域极大且 > μ+3σ，排除目标框。

    xf: float32 校正域帧（dilate 走 32F 快路径）。
    """
    bg = np.ones((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes_f:
        bg[max(0, y1 - 2):y2 + 3, max(0, x1 - 2):x2 + 3] = False
    if bg.sum() < 1000:
        return 0.0
    vals = xf[bg]
    mu, sig = float(vals.mean()), float(vals.std())
    local_max = xf == cv2.dilate(xf, np.ones((3, 3), np.uint8))
    n = int(((local_max & (xf > mu + 3 * sig)) & bg).sum())
    return n / (H * W / 1000.0)


def instance_scr(xc: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """单实例 SCR = |μ_T − μ_B| / σ_B（背景环 = 框外扩 8px）。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    mt = float(xc[mask > 0].mean()) if (mask > 0).any() else float(xc[y1:y2 + 1, x1:x2 + 1].mean())
    rx1, ry1 = max(0, x1 - SCR_RING_PAD), max(0, y1 - SCR_RING_PAD)
    rx2, ry2 = min(W - 1, x2 + SCR_RING_PAD), min(H - 1, y2 + SCR_RING_PAD)
    ring = np.ones((ry2 - ry1 + 1, rx2 - rx1 + 1), dtype=bool)
    ring[y1 - ry1:y2 - ry1 + 1, x1 - rx1:x2 - rx1 + 1] = False
    vals = xc[ry1:ry2 + 1, rx1:rx2 + 1][ring]
    mu_b, sig_b = float(vals.mean()), float(vals.std())
    return abs(mt - mu_b) / max(sig_b, 1e-3)


def process_sequence(seq_id: int) -> dict:
    """单序列全流水线，写缓存目录并返回 seq_meta 统计。"""
    t_all = time.time()
    t_stage: dict[str, float] = {}
    cache = REPO / "data" / "cache" / "ittd" / f"seq_{seq_id:04d}"
    cache.mkdir(parents=True, exist_ok=True)

    # ① 解码 + 原始帧 memmap
    t0 = time.time()
    frames = read_frames(seq_id)
    np.save(cache / "frames.u8.npy", frames)
    t_stage["decode"] = time.time() - t0

    # ② 时序中值 → 坏点 + NUC 平场
    t0 = time.time()
    med = temporal_median(frames)
    dead = detect_dead_pixels(med)
    nuc = nuc_field(med).astype(np.float32)  # 每段转换一次，供逐帧校正复用
    np.save(cache / "deadpix.npy", dead)
    np.save(cache / "nuc_field.npy", nuc_field(med))
    t_stage["deadpix_nuc"] = time.time() - t0

    # ③ 逐帧校正 + 稳健统计 + 质检量（配准输入 = 校正后 u8，中心化 +128）
    t0 = time.time()
    ann = load_seq_annotation(ITTD_ROOT / "Annotation" / str(seq_id))
    boxes = np.zeros((0, 5), dtype=np.int32)
    per_frame_boxes: dict[int, np.ndarray] = {}
    rows_f, rows_box = [], []
    for f in range(1, N_FRAMES + 1):
        insts = ann.frames.get(f, [])
        if insts:
            per_frame_boxes[f] = np.array(
                [[i.box[0], i.box[1], i.box[2], i.box[3]] for i in insts], np.int32
            )
            for i in insts:
                rows_f.append(f)
                rows_box.append([i.box[0], i.box[1], i.box[2], i.box[3]])
    boxes = np.array(rows_box, np.int32) if rows_box else np.zeros((0, 4), np.int32)
    frames_idx = np.array(rows_f, np.int32)

    stats = np.zeros((N_FRAMES, 2), np.float32)
    nonzero = np.zeros(N_FRAMES, np.float32)
    lapvar = np.zeros(N_FRAMES, np.float32)
    clutter = np.zeros(N_FRAMES, np.float32)
    reg_in = np.empty((N_FRAMES, H, W), np.uint8)
    for t in range(N_FRAMES):
        # xf: float32 校正域（几何/杂波运算）；xc16: int16 量化域（统计/配准输入/SCR）
        xf = correct_frame(frames[t], nuc, dead)
        xc16 = np.rint(xf).astype(np.int16)
        med_t, sig_t = frame_stats(xc16)
        stats[t] = (med_t, sig_t)
        nonzero[t] = float((xc16 > med_t).mean())      # x̂>0 占比（QC③，整型比较免建浮点数组）
        lapvar[t] = float(cv2.Laplacian(xf, cv2.CV_32F).var())
        clutter[t] = clutter_density(xf, per_frame_boxes.get(t + 1,
                                                         np.zeros((0, 4), np.int32)))
        reg_in[t] = np.clip(xc16 + 128, 0, 255).astype(np.uint8)
    np.save(cache / "norm_stats.npy", stats)
    np.save(cache / "quality.npy", lapvar.astype(np.float32))  # 逐帧清晰度（校正域拉普拉斯方差）
    t_stage["stats_qc"] = time.time() - t0

    # ④ 配准（滑动参考 + 质量门限 + FM 回退）
    t0 = time.time()
    reg = register_sequence(reg_in)
    reg.save(cache / "reg.npz")
    t_stage["register"] = time.time() - t0

    # ⑤ 标签 + 热图
    t0 = time.time()
    n_inst = len(frames_idx)
    if n_inst:
        labels_boxes = np.column_stack([frames_idx, boxes]).astype(np.int32)
        track_ids = np.array(
            [i.track_id for f in range(1, N_FRAMES + 1) for i in ann.frames.get(f, [])],
            np.int32,
        )
    else:
        labels_boxes = np.zeros((0, 5), np.int32)
        track_ids = np.zeros(0, np.int32)
    tindex = build_track_index(labels_boxes, track_ids)
    ids_a, offs_a, frames_a = track_index_to_arrays(tindex)
    np.savez_compressed(
        cache / "labels.npz",
        boxes=labels_boxes, track_ids=track_ids,
        track_ids_index=ids_a, track_offsets=offs_a, track_frames=frames_a,
    )
    hm = np.lib.format.open_memmap(
        cache / "heatmaps.u8.npy", mode="w+", dtype=np.uint8, shape=(N_FRAMES, H, W)
    )
    for f in range(1, N_FRAMES + 1):
        hm[f - 1] = build_heatmap_frame(per_frame_boxes.get(f, np.zeros((0, 4), np.int32)))
    hm.flush()
    t_stage["labels_heatmap"] = time.time() - t0

    # ⑥ 实例掩码 RLE（suppressed 来自 instances JSON）
    t0 = time.time()
    jd = json.loads((SEG_ROOT / "instances" / f"{seq_id}.json").read_text(encoding="utf-8"))
    supp_map: dict[tuple[int, int], bool] = {}
    for fr in jd["frames"]:
        for o in fr["objects"]:
            supp_map[(fr["frame"], o["id"])] = bool(
                o.get("suppressed", False) or o.get("won_px", 1) == 0
            )
    suppressed = np.array(
        [supp_map.get((int(f), int(i)), False) for f, i in zip(frames_idx, track_ids)],
        bool,
    ) if n_inst else np.zeros(0, bool)
    masks = load_instance_masks(SEG_ROOT, seq_id, frames_idx, track_ids,
                                labels_boxes, suppressed)
    np.savez_compressed(cache / "masks.npz", **masks)
    t_stage["masks"] = time.time() - t0

    # ⑦ SCR 统计（复用掩码；同帧实例共享一次校正）+ 序列级标签
    t0 = time.time()
    rle_off = masks["rle_offsets"]
    rle_runs = masks["rle_runs"]
    shape = tuple(masks["shape"])
    scrs: dict[int, list[float]] = {}
    last_f, xc_cache = -1, None
    for i in range(n_inst):
        runs = rle_runs[rle_off[i]:rle_off[i + 1]]
        m = np.zeros(H * W, np.uint8)
        pos, val = 0, 0
        for r in runs:
            if val:
                m[pos:pos + int(r)] = 1
            pos += int(r)
            val ^= 1
        m = m.reshape(shape)
        t = int(track_ids[i])
        f = int(frames_idx[i])
        if f != last_f:
            xc_cache = np.rint(correct_frame(frames[f - 1], nuc, dead)).astype(np.int16)
            last_f = f
        scrs.setdefault(t, []).append(instance_scr(xc_cache, m, labels_boxes[i, 1:5]))
    per_track_scr = {t: float(np.mean(v)) for t, v in scrs.items()}
    mean_scr = float(np.mean(list(per_track_scr.values()))) if per_track_scr else None
    min_scr = float(np.min(list(per_track_scr.values()))) if per_track_scr else None
    t_stage["scr"] = time.time() - t0

    # 轨迹级统计标签
    max_gap, long_occ = 0, False
    for t, fl in tindex.items():
        fl = sorted(fl)
        if len(fl) > 1:
            g = int(np.max(np.diff(fl))) - 1
            max_gap = max(max_gap, g)
            long_occ |= g >= LONG_OCC_GAP
    if n_inst:
        wh = labels_boxes[:, 4] - labels_boxes[:, 2] + 1
        hh = labels_boxes[:, 3] - labels_boxes[:, 1] + 1
        diag = np.hypot(wh, hh)
        small_frac = float((diag < SMALL_DIAG_PX).mean())
    else:
        small_frac = 0.0
    blur_med = float(np.median(lapvar)) if lapvar.size else 0.0
    blur_ratio = float((lapvar < 0.3 * blur_med).mean()) if blur_med > 0 else 0.0

    reg_ok = ~(reg.failed | (reg.method == "flat"))
    klt_ok = reg_ok & np.isin(reg.method, ["klt", "klt2"])
    fm_ok = reg_ok & np.isin(reg.method, ["fm", "fm2"])
    n_flat = int((reg.method == "flat").sum())
    meta = {
        "seq_id": seq_id,
        "n_frames": N_FRAMES,
        "n_instances": int(n_inst),
        "n_tracks": len(tindex),
        "n_empty_frames": int(sum(1 for v in ann.frames.values() if len(v) == 0)),
        "deadpix": int(len(dead)),
        "reg": {
            "n_failed": int(reg.failed.sum()),
            "n_flat": n_flat,
            "n_klt": int((reg.method == "klt").sum()),
            "n_klt2": int((reg.method == "klt2").sum()),
            "n_fm": int((reg.method == "fm").sum()),
            "n_fm2": int((reg.method == "fm2").sum()),
            "n_chained": int(reg.extras.get("n_chained", 0)),
            "success_rate": float(reg_ok.mean()),
            # rmse 语义拆分（评审 2.2a）：KLT 帧的 rmse 是像素域内点 RMSE、且仅当
            # ≤0.5px 才被接受（受门限截断，P95 是"门限满足性"而非独立精度证据）；
            # FM 帧的 rmse 字段存 1−corr（无量纲），绝不可与像素值混入同一分位。
            "rmse_p50": float(np.percentile(reg.rmse[klt_ok], 50)) if klt_ok.any() else None,
            "rmse_p95": float(np.percentile(reg.rmse[klt_ok], 95)) if klt_ok.any() else None,
            "fm_corr_med": (float(np.median(1.0 - reg.rmse[fm_ok])) if fm_ok.any() else None),
            "fm_corr_min": (float((1.0 - reg.rmse[fm_ok]).min()) if fm_ok.any() else None),
        },
        "nonzero_frac": {"min": float(nonzero.min()), "mean": float(nonzero.mean()),
                         "max": float(nonzero.max())},
        "lapvar_median": blur_med,
        "blur_ratio": blur_ratio,
        "clutter_density_mean": float(clutter.mean()),
        "clutter_density_p75": float(np.percentile(clutter, 75)),
        "mean_scr": mean_scr,
        "min_scr": min_scr,
        "max_track_gap": max_gap,
        "long_occlusion": bool(long_occ),
        "small_target_frac": small_frac,
        "small_target": bool(small_frac > SMALL_FRAC_GATE),
        "blur_tag": bool(blur_ratio > BLUR_RATIO_GATE),
        "timings": {**{k: round(v, 2) for k, v in t_stage.items()},
                    "total": round(time.time() - t_all, 2)},
        "fps": round(N_FRAMES / (time.time() - t_all), 1),
    }
    (cache / "seq_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return meta


def _worker(seq_id: int) -> dict:
    cv2.setNumThreads(1)  # 物理核数 = 进程数时禁内部线程，避免超订争抢
    return process_sequence(seq_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="*", default=None, help="序列号（默认全量 1-87）")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    seqs = args.seqs or list(range(1, 88))

    t0 = time.time()
    if args.workers <= 1 or len(seqs) == 1:
        metas = [_worker(s) for s in seqs]
    else:
        with Pool(args.workers) as pool:
            metas = []
            for i, m in enumerate(pool.imap_unordered(_worker, seqs), 1):
                metas.append(m)
                print(f"  [{i}/{len(seqs)}] seq{m['seq_id']:3d} "
                      f"{m['timings']['total']:6.1f}s ({m['fps']:5.1f} fps/seq) "
                      f"reg_ok={m['reg']['success_rate']:.3f}")

    wall = time.time() - t0
    total_frames = sum(m["n_frames"] for m in metas)
    summary = {
        "seqs": len(metas),
        "wall_sec": round(wall, 1),
        "total_frames": total_frames,
        "throughput_fps_wall": round(total_frames / wall, 1),
        "workers": args.workers,
        "gate_throughput>=200": total_frames / wall >= 200,
    }
    out = REPO / "reports" / "m1"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"preprocess_{'smoke' if len(seqs) < 87 else 'full'}.json").write_text(
        json.dumps({"summary": summary, "per_seq": sorted(metas, key=lambda m: m["seq_id"])},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
