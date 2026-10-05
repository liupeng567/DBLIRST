"""Manifest 数据结构与校验（方案 1.5 / 1.6）。

- 属性字段三分类：确定性（断言）/ 官方权威源（scene、daytime、五个难点标签）/ 统计（M1 回填）。
- 每个属性强制携带 source: official / auto / manual 与 confidence: high / medium / low。
- 校验断言：64/12/11 段、16,000/3,000/2,750 帧、393 轨迹、无重复、val-official 只读、
  官方属性表 87 行且与 split 交叉一致。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

N_TRAIN, N_VALINT, N_VALOFF = 64, 12, 11
SPLITS = ("train-int", "val-int", "val-official")

_DAYTIME_MAP = {"白天": "day", "傍晚": "dusk"}
_SCENE_MAP = {"内场": "indoor", "外场": "outdoor"}
# 官方 CSV 列 → difficulty tag（1.4 v1.6 映射，provenance=official）
_TAG_COLUMNS = {
    "crossing": "crossing",
    "occlusion": "occlusion",
    "static_target": "static_target",
    "distractor": "distractor_present",
    "shake": "motion_blur",  # 官方"平台大幅晃动"列；M1 与自动模糊帧占比取并集
}


class ManifestError(AssertionError):
    """manifest 校验失败（方案 1.5：任何不符即拒绝生成）。"""


def load_official_attributes(csv_path: str | Path) -> dict[int, dict[str, str]]:
    """解析官方属性表 ittd_official_attributes.csv（UTF-8 BOM）。"""
    rows: dict[int, dict[str, str]] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            sid = int(row["seq_id"])
            if sid != len(rows) + 1:
                raise ManifestError(f"属性表序号不连续: 期待 {len(rows)+1} 得 {sid}")
            if row["daytime"] not in _DAYTIME_MAP:
                raise ManifestError(f"seq{sid} 天候非法: {row['daytime']}")
            if row["scene"] not in _SCENE_MAP:
                raise ManifestError(f"seq{sid} 场地非法: {row['scene']}")
            rows[sid] = row
    if len(rows) != 87:
        raise ManifestError(f"属性表应 87 行，实得 {len(rows)}")
    return rows


def make_sequence_entry(
    seq_id: int, stats: dict[str, Any], attr: dict[str, str]
) -> dict[str, Any]:
    """由解析统计 + 官方属性行构造单序列 manifest 条目（含 provenance）。"""
    difficulty: dict[str, Any] = {
        tag: {"value": attr[col] == "有", "source": "official", "confidence": "high"}
        for col, tag in _TAG_COLUMNS.items()
    }
    # M1 待回填的统计型标签与指标：占位但显式声明 provenance
    for pending in ("strong_clutter", "long_occlusion", "small_target"):
        difficulty[pending] = {
            "value": None,
            "source": "auto",
            "confidence": None,
            "pending": "M1",
        }
    return {
        "seq_id": seq_id,
        "scene": {"value": _SCENE_MAP[attr["scene"]], "source": "official",
                  "confidence": "high"},
        "daytime": {"value": _DAYTIME_MAP[attr["daytime"]], "source": "official",
                    "confidence": "high"},
        "n_frames": stats["n_frames_img"],
        "n_tracks": stats["n_tracks"],
        "n_instances": stats["n_instances"],
        "track_ids": stats["track_ids"],
        "n_empty_frames": stats["n_empty_frames"],
        "difficulty_tags": difficulty,
        "official_raw": {
            k: attr[k]
            for k in ("daytime", "scene", "shake", "static_target", "moving_target",
                      "occlusion", "crossing", "distractor")
        },
        # 统计字段（1.6 第三类）：M1 回填，留原始值避免只留分档结论
        "mean_scr": None,
        "min_scr": None,
        "clutter_density": None,
        "reg_difficulty": None,
    }


def build_manifest_object(
    version: str,
    splits: dict[str, list[int]],
    sequences: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装 manifest 并执行 1.5 全部断言，任何不符抛 ManifestError。"""
    seq_ids = [s["seq_id"] for s in sequences]
    if sorted(seq_ids) != list(range(1, 88)):
        raise ManifestError("sequences 必须恰好覆盖 1..87")

    covered: list[int] = []
    for sp in SPLITS:
        covered.extend(splits[sp])
    if sorted(covered) != list(range(1, 88)):
        raise ManifestError("三个子集并集必须为 1..87 且无重复")

    n_by_split = {sp: len(v) for sp, v in splits.items()}
    if (n_by_split["train-int"], n_by_split["val-int"], n_by_split["val-official"]) != (
        N_TRAIN, N_VALINT, N_VALOFF
    ):
        raise ManifestError(f"段数应为 64/12/11，实得 {n_by_split}")

    inst_by_split = {
        sp: sum(s["n_instances"] for s in sequences if s["seq_id"] in set(v))
        for sp, v in splits.items()
    }
    tracks_total = sum(s["n_tracks"] for s in sequences)
    frames_by_split = {
        sp: sum(s["n_frames"] for s in sequences if s["seq_id"] in set(v))
        for sp, v in splits.items()
    }
    if frames_by_split != {
        "train-int": 16_000, "val-int": 3_000, "val-official": 2_750
    }:
        raise ManifestError(f"帧数应为 16000/3000/2750，实得 {frames_by_split}")
    if tracks_total != 393:
        raise ManifestError(f"轨迹总数应为 393，实得 {tracks_total}")

    payload = {
        "version": version,
        "splits": {
            "train-int": {"seqs": sorted(splits["train-int"])},
            "val-int": {"seqs": sorted(splits["val-int"])},
            "val-official": {"seqs": sorted(splits["val-official"]),
                             "read_only": True},
        },
        "sequences": sorted(sequences, key=lambda s: s["seq_id"]),
        "summary": {
            "n_seqs": n_by_split,
            "n_frames": frames_by_split,
            "n_instances": inst_by_split,
            "n_tracks_total": tracks_total,
        },
    }
    if extra:
        payload.update(extra)
    payload["checksum"] = "md5:" + hashlib.md5(
        json.dumps(
            {"splits": payload["splits"], "sequences": payload["sequences"]},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def load_manifest(path: str | Path) -> dict[str, Any]:
    """读取 manifest 并复核 checksum。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = data.pop("checksum", None)
    expect = "md5:" + hashlib.md5(
        json.dumps(
            {"splits": data["splits"], "sequences": data["sequences"]},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if checksum != expect:
        raise ManifestError(f"checksum 不符: {checksum} != {expect}")
    data["checksum"] = checksum
    return data


def split_of(manifest: dict[str, Any], seq_id: int) -> str:
    for sp, info in manifest["splits"].items():
        if seq_id in info["seqs"]:
            return sp
    raise ManifestError(f"seq{seq_id} 不在任何 split")
