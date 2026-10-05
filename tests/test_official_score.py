"""官方评分复刻器三组合成用例（方案 8.2，M0 验收 Gate）。

① 全对场景：预测=真值中心 → 检测分=航迹分=GT 实例数（总分的满分假设 2×N_inst）。
② 虚警/漏检注入：虚警 −2/个、漏检 −1/个、含边界判"框内（含）"。
③ 交叉航迹：构造最优匹配需匈牙利算法的场景（贪心/按 ID 直配会得次优），
   验证重合度累计与匹配的边界行为（含零重叠预测航迹）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.ittd_parse import Instance, SeqAnnotation, parse_official_txt, write_official_txt
from dsld.eval.official_score import score_sequence


def box_at(cx: float, cy: float, half: int = 2) -> tuple[int, int, int, int]:
    return (int(cx) - half, int(cy) - half, int(cx) + half, int(cy) + half)


def pt(track_id: int, x: float, y: float) -> Instance:
    """中心点检测（官方输出坐标点 → 退化框）。"""
    return Instance(track_id=track_id, box=(int(x), int(y), int(x), int(y)))


# ---------- 用例①：全对场景 ----------

class TestAllCorrect:
    def make_gt(self) -> SeqAnnotation:
        gt = SeqAnnotation(seq_id=1, n_target=2)
        for f in range(1, 11):  # 两条对穿航迹，10 帧
            gt.frames[f] = [
                Instance(track_id=1, box=box_at(f, 10.0)),
                Instance(track_id=2, box=box_at(21.0 - f, 30.0)),
            ]
        return gt

    def test_full_score_equals_2x_instances(self):
        gt = self.make_gt()
        pred = {
            f: [pt(1, *_center(gt.frames[f][0])),
                pt(2, *_center(gt.frames[f][1]))]
            for f in gt.frames
        }
        res = score_sequence(gt, pred)
        n_inst = 20
        assert res.detection == n_inst, f"检测分应为 {n_inst}，实得 {res.detection}"
        assert res.continuity == n_inst, f"航迹分应为 {n_inst}，实得 {res.continuity}"
        assert res.total == 2 * n_inst

    def test_track_ids_shuffled_still_full(self):
        """预测航迹 ID 与 GT 无对应关系也不影响得分（匈牙利重配）。"""
        gt = self.make_gt()
        pred = {f: [pt(7, *_center(gt.frames[f][0])), pt(3, *_center(gt.frames[f][1]))]
                for f in gt.frames}
        res = score_sequence(gt, pred)
        assert res.total == 40

    def test_end_to_end_txt_roundtrip(self, tmp_path: Path):
        """write_official_txt → parse → score 与内存直评一致。"""
        gt = self.make_gt()
        pred = {f: [pt(1, *_center(gt.frames[f][0])), pt(2, *_center(gt.frames[f][1]))]
                for f in gt.frames}
        out = tmp_path / "1.txt"
        write_official_txt(out, seq_id=1, detections=pred)
        reparsed = parse_official_txt(out)
        res = score_sequence(gt, reparsed.frames)
        assert res.total == 40


def _center(ins: Instance) -> tuple[float, float]:
    return ins.cx, ins.cy


# ---------- 用例②：虚警 / 漏检 / 边界 ----------

class TestFalseAlarmAndMiss:
    def make_gt(self) -> SeqAnnotation:
        gt = SeqAnnotation(seq_id=2, n_target=1)
        for f in range(1, 6):
            gt.frames[f] = [Instance(track_id=1, box=(0, 0, 10, 10))]
        return gt

    def test_each_false_alarm_minus2(self):
        gt = self.make_gt()
        pred = {f: [pt(1, 5, 5)] for f in gt.frames}
        pred[3].append(pt(9, 100, 100))
        pred[4].append(pt(9, 100, 100))
        res = score_sequence(gt, pred)
        assert res.detection == 5 - 2 * 2  # 5 命中 − 2×2 虚警
        assert res.details["detection_details"]["false_alarm"] == 2

    def test_each_miss_minus1(self):
        gt = self.make_gt()
        pred = {f: ([pt(1, 5, 5)] if f != 2 else []) for f in gt.frames}
        res = score_sequence(gt, pred)
        assert res.detection == 4 - 1  # 4 命中 − 1 漏检

    def test_boundary_inclusive(self):
        """点恰在框边界上 → 判"框内（含）"。"""
        gt = self.make_gt()
        pred = {f: [pt(1, 10, 10)] for f in gt.frames}  # 右下角点
        res = score_sequence(gt, pred)
        assert res.detection == 5

    def test_duplicate_points_in_box_neutral_literal(self):
        """同框多点：literal 口径下多余点不在框外 → 不罚（论文逐字口径）。"""
        gt = self.make_gt()
        pred = {f: [pt(1, 5, 5), pt(1, 6, 6)] for f in gt.frames}
        res = score_sequence(gt, pred)
        assert res.detection == 5

    def test_strict_mode_counts_duplicate_as_fa(self):
        """strict 口径：一一对应，同框第二点按虚警 −2。"""
        gt = self.make_gt()
        pred = {f: [pt(1, 5, 5), pt(1, 6, 6)] for f in gt.frames}
        res = score_sequence(gt, pred, mode="strict")
        assert res.detection == 5 - 2 * 5

    def test_continuity_immune_to_fa(self):
        gt = self.make_gt()
        pred = {f: [pt(1, 5, 5), pt(9, 100, 100)] for f in gt.frames}
        res = score_sequence(gt, pred)
        assert res.continuity == 5  # 虚警航迹 9 重叠为 0


# ---------- 用例③：交叉航迹与匈牙利匹配 ----------

class TestCrossingTracks:
    def make_case(self):
        """GT A(1-10 帧) 与 GT B(1-10 帧)，三条预测航迹：

        P1 = A 全程（overlap(A,P1)=10）；
        P2 = B 前 6 帧 + 后 4 帧漂入 A 的框（overlap(A,P2)=4, overlap(B,P2)=6）；
        P3 = B 后 4 帧（overlap(B,P3)=4）。
        全部 GT 框均有检测 → 检测分 = 20；匈牙利最优 = A→P1(10) + B→P2(6) = 16
        （次优组合 A→P1 + B→P3 = 14、A→P2 + B→P3 = 8）。
        """
        gt = SeqAnnotation(seq_id=3, n_target=2)
        a_pos = {f: (10.0 + f, 20.0) for f in range(1, 11)}
        b_pos = {f: (50.0 - f, 20.0) for f in range(1, 11)}
        for f in range(1, 11):
            gt.frames[f] = [
                Instance(track_id=1, box=box_at(*a_pos[f])),
                Instance(track_id=2, box=box_at(*b_pos[f])),
            ]
        pred = {}
        for f in range(1, 11):
            pred[f] = [pt(1, *a_pos[f])]
        for f in range(1, 7):
            pred[f].append(pt(2, *b_pos[f]))
        for f in range(7, 11):
            pred[f].append(pt(2, *a_pos[f]))  # ID 漂移进 A 的框
            pred[f].append(pt(3, *b_pos[f]))  # P3 补上 B 后半程
        return gt, pred, {"expect_cont": 16, "expect_det": 20}

    def test_hungarian_optimal_not_greedy(self):
        gt, pred, exp = self.make_case()
        res = score_sequence(gt, pred)
        assert res.detection == exp["expect_det"]
        assert res.continuity == exp["expect_cont"]
        matched = {m["gt_track"]: m["overlap"]
                   for m in res.details["continuity_details"]["matched"]}
        # B 应匹配到重叠更高的 P2(6)，而非 P3(4)——匈牙利全局最优的直接证据
        assert matched[1] == 10 and matched[2] == 6 and matched.get(3) is None

    def test_pure_fp_track_zero_overlap(self):
        """纯虚警航迹参与匹配但贡献 0，总分不变。"""
        gt, pred, exp = self.make_case()
        for f in range(1, 11):
            pred[f].append(pt(9, 200.0 + f, 200.0))
        res = score_sequence(gt, pred)
        assert res.continuity == exp["expect_cont"]
        assert res.detection == exp["expect_det"] - 2 * 10  # 每帧 1 虚警

    def test_empty_pred_scores_negatively(self):
        """全空预测：检测分 = −N_inst（全漏检），航迹分 = 0。"""
        gt, _, _ = self.make_case()
        res = score_sequence(gt, {})
        assert res.detection == -20
        assert res.continuity == 0

    def test_strict_mode_drift_points_counted_fa(self):
        """strict 一一对应口径：P2 漂入已被 P1 占据的 A 框（4 帧）→ 各记虚警 −2。

        与 literal 的差异是模式语义本身（重复点处理），用在此处固化两种口径。
        """
        gt, pred, exp = self.make_case()
        res = score_sequence(gt, pred, mode="strict")
        assert res.detection == exp["expect_det"] - 2 * 4
        assert res.continuity == exp["expect_cont"]  # 航迹分与模式无关

    def test_modes_agree_on_one_point_per_box(self):
        """真干净场景（每框恰一点、无 ID 漂移）下两种口径必然一致。"""
        gt = SeqAnnotation(seq_id=9, n_target=1)
        for f in range(1, 6):
            gt.frames[f] = [Instance(track_id=1, box=box_at(f * 3.0, 50.0))]
        pred = {f: [pt(1, f * 3.0, 50.0)] for f in range(1, 6)}
        lit = score_sequence(gt, pred, mode="literal")
        stc = score_sequence(gt, pred, mode="strict")
        assert lit.total == stc.total == 10
