"""最小化贪心跟踪器（M2 基线用；DSLD 正式跟踪状态机在 M5 按 6.2 实现）。

用途：官方评分的航迹连续性项需要预测航迹 ID（6.6）。基线输出为逐帧独立检测框，
此模块以常数速度预测 + 匈牙利关联 + 速度门（表 I：vmax 8 px/帧，遮挡滑行门宽 ×2，
coast ≤ 10 帧）形成最简航迹，保证基线官方分可计算、可复现。

设计保持"检测器无偏"：不做 M-of-N 确认过滤、不按航迹长度删输出（min_len=1）——
航迹质量过滤是 DSLD 的贡献点之一（6.4），基线若提前使用会虚增其官方分。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from dsld.eval.mask_to_boxes import Box


class GreedyTracker:
    def __init__(self, vmax: float = 8.0, coast_max: int = 10):
        self.vmax = vmax
        self.coast_max = coast_max
        self.next_id = 1
        self.tracks: dict[int, dict] = {}  # tid → {cx, cy, vx, vy, coast, last_frame}

    def _predict(self, tid: int) -> tuple[float, float]:
        t = self.tracks[tid]
        return t["cx"] + t["vx"], t["cy"] + t["vy"]

    def update(self, dets: list[Box], frame: int) -> list[Box]:
        """逐帧更新：关联 → 更新状态 → 为检测分配 track_id（原地写入返回）。"""
        if not self.tracks and not dets:
            return dets
        tids = list(self.tracks.keys())
        if tids and dets:
            cost = np.full((len(tids), len(dets)), 1e6, dtype=np.float64)
            for i, tid in enumerate(tids):
                t = self.tracks[tid]
                px, py = self._predict(tid)
                gate = self.vmax * (2.0 if t["coast"] > 0 else 1.0)
                for j, d in enumerate(dets):
                    dist = float(np.hypot(d.cx - px, d.cy - py))
                    if dist <= gate:
                        cost[i, j] = dist
            rows, cols = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                tid = tids[i]
                t = self.tracks[tid]
                d = dets[j]
                nvx = 0.5 * t["vx"] + 0.5 * (d.cx - t["cx"])
                nvy = 0.5 * t["vy"] + 0.5 * (d.cy - t["cy"])
                t.update(cx=d.cx, cy=d.cy, vx=nvx, vy=nvy,
                         coast=0, last_frame=frame)
                d.track_id = tid
                matched_t.add(tid)
                matched_d.add(j)
        else:
            matched_t, matched_d = set(), set()

        for j, d in enumerate(dets):
            if j not in matched_d:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "cx": d.cx, "cy": d.cy, "vx": 0.0, "vy": 0.0,
                    "coast": 0, "last_frame": frame,
                }
                d.track_id = tid

        for tid in list(self.tracks.keys()):
            if tid in matched_t:
                continue
            self.tracks[tid]["coast"] += 1
            if self.tracks[tid]["coast"] > self.coast_max:
                del self.tracks[tid]
        return dets
