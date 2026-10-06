"""M1 补填：逐帧质量指标 quality.npy（方案 10.3 质量感知推理的数据基础）。

对已建缓存补算每帧拉普拉斯方差（校正域，与 preprocess_all 同一口径），
供 M2 退化增强标定 / 质量感知 FiLM / 关联门放宽使用。
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.preprocess.normalize import correct_frame  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "cache" / "ittd"


def backfill_seq(seq_id: int) -> int:
    cache = CACHE / f"seq_{seq_id:04d}"
    if not (cache / "seq_meta.json").exists():
        return 0
    frames = np.load(cache / "frames.u8.npy", mmap_mode="r")
    nuc = np.load(cache / "nuc_field.npy")
    dead = np.load(cache / "deadpix.npy")
    lap = np.empty(len(frames), np.float32)
    for t in range(len(frames)):
        xf = correct_frame(np.asarray(frames[t]), nuc, dead)
        lap[t] = float(cv2.Laplacian(xf, cv2.CV_32F).var())
    np.save(cache / "quality.npy", lap)
    return 1


def _w(sid: int) -> int:
    cv2.setNumThreads(1)
    return backfill_seq(sid)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(16) as pool:
        n = sum(pool.map(_w, range(1, 88)))
    print(f"quality.npy 补填完成: {n}/87 段, {time.time() - t0:.1f}s")
