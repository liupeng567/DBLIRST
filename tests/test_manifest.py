"""manifest 单元测试（方案 9.1 tests 四件套之一）。

覆盖：v1/v2/v3 校验和复核、划分断言（64/12/11 段、16,000/3,000/2,750 帧、393 轨迹、
无重复、val-official 只读）、v2/v3 划分一致（统计回填不动 splits）、v2 重平衡构成、
provenance 字段完整性、split_of 查询。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.manifest import ManifestError, load_manifest, split_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
M_DIR = REPO / "data" / "manifests"


def _load(v: str) -> dict:
    return load_manifest(M_DIR / f"ittd_split_{v}.json")


@pytest.mark.parametrize("v", ["v1", "v2", "v3"])
class TestManifestCommon:
    def test_checksum_and_splits(self, v):
        m = _load(v)
        assert {sp: len(i["seqs"]) for sp, i in m["splits"].items()} == {
            "train-int": 64, "val-int": 12, "val-official": 11,
        }
        assert m["summary"]["n_frames"] == {
            "train-int": 16_000, "val-int": 3_000, "val-official": 2_750,
        }
        assert m["summary"]["n_tracks_total"] == 393
        covered = sum((i["seqs"] for i in m["splits"].values()), [])
        assert sorted(covered) == list(range(1, 88))  # 无重复、全覆盖

    def test_val_official_read_only(self, v):
        assert _load(v)["splits"]["val-official"]["read_only"] is True

    def test_provenance_complete(self, v):
        for s in _load(v)["sequences"]:
            for field in ("scene", "daytime"):
                assert s[field]["source"] in ("official", "auto", "manual")
                assert s[field]["confidence"] in ("high", "medium", "low", None)
            assert set(s["difficulty_tags"]) >= {
                "crossing", "occlusion", "static_target", "distractor_present",
                "motion_blur", "strong_clutter", "long_occlusion", "small_target",
            }


class TestV2Rebalance:
    def test_composition(self):
        m = _load("v2")
        seqs = {s["seq_id"]: s for s in m["sequences"]}
        val = m["splits"]["val-int"]["seqs"]
        day = sum(1 for s in val if seqs[s]["daytime"]["value"] == "day")
        dusk = sum(1 for s in val if seqs[s]["daytime"]["value"] == "dusk")
        assert (day, dusk) == (6, 6), f"v2 val-int 应 day6/dusk6，实得 {day}/{dusk}"
        train = m["splits"]["train-int"]["seqs"]
        n_outdoor_dusk = sum(
            1 for s in train
            if seqs[s]["scene"]["value"] == "outdoor"
            and seqs[s]["daytime"]["value"] == "dusk"
        )
        assert n_outdoor_dusk >= 6
        # 官方验证集不动
        assert m["splits"]["val-official"]["seqs"] == list(range(77, 88))


class TestV3Backfill:
    def test_splits_unchanged_from_v2(self):
        assert _load("v2")["splits"] == _load("v3")["splits"]

    def test_statistics_backfilled(self):
        m = _load("v3")
        for s in m["sequences"]:
            assert s["mean_scr"] is not None and s["mean_scr"] > 0
            assert s["clutter_density"] is not None
            assert s["reg_difficulty"] is not None and 0 <= s["reg_difficulty"] <= 0.5
            sc = s["difficulty_tags"]["strong_clutter"]
            assert sc["value"] == (sc["raw"] >= sc["threshold"])  # 标签与原始值自洽
        n_sc = sum(1 for s in m["sequences"]
                   if s["difficulty_tags"]["strong_clutter"]["value"])
        assert n_sc == 21  # 前 25% of 87

    def test_changelog(self):
        m = _load("v3")
        assert any(c["version"] == "v3" for c in m["changelog"])


class TestSplitOf:
    def test_query(self):
        m = _load("v2")
        assert split_of(m, 1) == "train-int"
        assert split_of(m, 71) == "train-int"   # v2 重平衡：外场傍晚 71-76 调入
        assert split_of(m, 70) == "val-int"     # v2 保留 67-70 于 val-int
        assert split_of(m, 22) == "val-int"     # v2 重平衡：白天 21-23/38-40 调入
        assert split_of(m, 87) == "val-official"
        with pytest.raises(ManifestError):
            split_of(m, 999)


class TestTamperDetection:
    def test_checksum_breaks_on_edit(self, tmp_path: Path):
        import json

        src = M_DIR / "ittd_split_v2.json"
        data = json.loads(src.read_text(encoding="utf-8"))
        data["sequences"][0]["mean_scr"] = 12345.0  # 篡改
        p = tmp_path / "tampered.json"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ManifestError):
            load_manifest(p)
