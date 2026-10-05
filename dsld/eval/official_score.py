"""官方评分复刻器（方案 6.6 / 8.1 / 8.2；风险 R6 的自研兜底）。

口径严格按 ITTD 数据论文（《中国科学数据》2022,7(2)，正文第 3-4 步）：

① 检测准确性得分（逐帧、逐目标）：
   - 正确检测：真值标注框内（含边界）有检测结果 → 该真值框 +1 分；
   - 漏检：真值标注框内（含）无检测结果 → −1 分；
   - 虚警：真值标注框外出现检测结果 → 每个检测点 −2 分。

② 航迹连续性得分：
   - 重合度 overlap(i,j) = 真值航迹 i 与预测航迹 j 之间"正确检测的数量"
     （预测航迹 j 的点落在真值航迹 i 当帧框内的帧数）；
   - 匈牙利算法求 GT×Pred 最优匹配（最大化）；
   - 得分 = 匹配后所有真值航迹获得的重合度之和 ×1。

自洽性锚点：预测=真值中心点时，检测分 = 航迹分 = GT 实例数，
验证集（77-87）GT 实例数 11,209 → 满分 22,418，与论文一致（check_ittd.py 已实证）。

歧义点决策（若获得官方程序须回溯对齐，R6）：
   - 多个检测点落在同一真值框内：默认 mode="literal"（论文逐字口径）——该框只记一次
     正确 +1，多余点不奖不罚（它们不在"框外"，不构成虚警）；
     mode="strict" 提供逐帧一一对应口径（每框至多配一点、每点至多配一框，
     未配对点按虚警 −2），用于敏感性分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from dsld.data.ittd_parse import Instance, SeqAnnotation, parse_official_txt

FULL_SCORE_VAL_OFFICIAL = 22_418  # 论文口径满分，对齐检查用


def point_inside(pt: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    """检测点是否在框内（含边界，论文"框内（含）"）。"""
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def det_center(ins: Instance) -> tuple[float, float]:
    """检测输出的判决点：坐标点退化框即其自身；回归框取中心（方案 6.6）。"""
    return ins.cx, ins.cy


@dataclass
class ScoreResult:
    seq_id: int
    detection: int
    continuity: int
    total: int
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seq_id": self.seq_id,
            "detection": self.detection,
            "continuity": self.continuity,
            "total": self.total,
            **self.details,
        }


def _score_detection_literal(
    gt: SeqAnnotation, pred: dict[int, list[Instance]]
) -> tuple[int, dict]:
    hit, miss, false_alarm = 0, 0, 0
    for f, gt_insts in gt.frames.items():
        pts = [det_center(p) for p in pred.get(f, [])]
        for g in gt_insts:
            if any(point_inside(pt, g.box) for pt in pts):
                hit += 1
            else:
                miss += 1
        for pt in pts:
            if not any(point_inside(pt, g.box) for g in gt_insts):
                false_alarm += 1
    score = hit - miss - 2 * false_alarm
    return score, {"hit": hit, "miss": miss, "false_alarm": false_alarm}


def _score_detection_strict(
    gt: SeqAnnotation, pred: dict[int, list[Instance]]
) -> tuple[int, dict]:
    hit, miss, false_alarm = 0, 0, 0
    for f, gt_insts in gt.frames.items():
        pts = [det_center(p) for p in pred.get(f, [])]
        assigned = [False] * len(gt_insts)
        for pt in pts:
            best = -1
            best_area = None
            for gi, g in enumerate(gt_insts):
                if assigned[gi] or not point_inside(pt, g.box):
                    continue
                area = (g.box[2] - g.box[0]) * (g.box[3] - g.box[1])
                if best_area is None or area < best_area:  # 最小包含框优先
                    best, best_area = gi, area
            if best >= 0:
                assigned[best] = True
                hit += 1
            else:
                false_alarm += 1
        miss = sum(1 for a in assigned if not a) + miss
    score = hit - miss - 2 * false_alarm
    return score, {"hit": hit, "miss": miss, "false_alarm": false_alarm}


def _score_continuity(
    gt: SeqAnnotation, pred: dict[int, list[Instance]]
) -> tuple[int, dict]:
    gt_tracks = sorted(gt.track_ids)
    pred_tracks = sorted({p.track_id for v in pred.values() for p in v})
    if not gt_tracks or not pred_tracks:
        return 0, {"overlap_matrix_shape": (len(gt_tracks), len(pred_tracks)),
                   "matched": []}
    idx_gt = {t: i for i, t in enumerate(gt_tracks)}
    idx_pr = {t: i for i, t in enumerate(pred_tracks)}
    overlap = np.zeros((len(gt_tracks), len(pred_tracks)), dtype=np.int64)
    for f, gt_insts in gt.frames.items():
        for g in gt_insts:
            for p in pred.get(f, []):
                if point_inside(det_center(p), g.box):
                    overlap[idx_gt[g.track_id], idx_pr[p.track_id]] += 1
    rows, cols = linear_sum_assignment(overlap, maximize=True)
    matched = [
        {"gt_track": gt_tracks[r], "pred_track": pred_tracks[c],
         "overlap": int(overlap[r, c])}
        for r, c in zip(rows, cols)
    ]
    score = int(sum(m["overlap"] for m in matched))
    return score, {"matched": matched,
                   "n_gt_tracks": len(gt_tracks), "n_pred_tracks": len(pred_tracks)}


def score_sequence(
    gt: SeqAnnotation,
    pred: dict[int, list[Instance]],
    mode: str = "literal",
) -> ScoreResult:
    """单序列官方评分。gt 来自 parse_official_txt / XML，pred = {frame: [Instance]}。"""
    if mode == "literal":
        det, det_details = _score_detection_literal(gt, pred)
    elif mode == "strict":
        det, det_details = _score_detection_strict(gt, pred)
    else:
        raise ValueError(f"未知 mode: {mode}")
    cont, cont_details = _score_continuity(gt, pred)
    return ScoreResult(
        seq_id=gt.seq_id,
        detection=det,
        continuity=cont,
        total=det + cont,
        details={"detection_details": det_details, "continuity_details": cont_details,
                 "mode": mode},
    )


def score_val_set(
    gt_dir: str | Path,
    pred_dir: str | Path,
    seqs: list[int] | None = None,
    mode: str = "literal",
) -> dict:
    """验证集批量评分。pred_dir 内 {seq}.txt 为官方输出格式（中心点版本）。"""
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    if seqs is None:
        seqs = sorted(int(p.stem) for p in pred_dir.glob("*.txt"))
    results: dict[int, ScoreResult] = {}
    for sid in seqs:
        gt = parse_official_txt(gt_dir / f"{sid}.txt")
        pred_ann = parse_official_txt(pred_dir / f"{sid}.txt")
        results[sid] = score_sequence(gt, pred_ann.frames, mode=mode)
    total = sum(r.total for r in results.values())
    det = sum(r.detection for r in results.values())
    cont = sum(r.continuity for r in results.values())
    return {
        "per_seq": {sid: r.as_dict() for sid, r in results.items()},
        "detection_total": det,
        "continuity_total": cont,
        "grand_total": total,
        "full_score": FULL_SCORE_VAL_OFFICIAL,
        "ratio": total / FULL_SCORE_VAL_OFFICIAL,
    }
