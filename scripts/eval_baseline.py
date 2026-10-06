"""M2 基线 eval 全链驱动（方案 8.1/8.3/6.6/9.2-M2）。

用法：
  python scripts/eval_baseline.py --ckpt experiments/mshnet_B/ckpt/B_0_25.pt \
      --splits val-int val-official --out experiments/mshnet_B/eval_report.json

链路：ckpt → 重建模型 → 逐段推理（infer_seq）→ 逐阈值预测框 →
  val-int：IoU≥0.5 框级 P/R/F1 + AP50/mAP + 中心命中 + F_a 双口径 + 阈值扫描
           （F_a@P_d=0.90 工作点）+ 分子集报告（8.3）；
  val-official：官方评分（复刻器，只读）+ 官方 txt/框版本 txt 落盘（6.6）；
  效率层：Params / FLOPs（thop）/ FPS（4060 batch=1 全分辨率）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.ittd_parse import Instance, SeqAnnotation, write_official_txt
from dsld.eval import official_score
from dsld.eval.infer_seq import (
    boxes_at_thresholds,
    gt_boxes_from_cache,
    infer_mshnet,
    infer_temporal,
    load_seq_cache,
)
from dsld.eval.map_iou import evaluate_ap, evaluate_boxes
from dsld.eval.tracker import GreedyTracker
from dsld.train.trainer import REPO, build_model, report_model_params

THR_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PRIMARY_THR = 0.5
DIFFICULTY_TAGS = [
    "strong_clutter", "crossing", "occlusion", "long_occlusion",
    "small_target", "static_target", "distractor_present", "motion_blur",
]


def load_model_from_ckpt(ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    model = build_model(cfg)
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg


def gt_seq_annotation(cache: dict) -> SeqAnnotation:
    ann = SeqAnnotation(seq_id=0)
    lab = cache["labels"]
    for (f, x1, y1, x2, y2), tid in zip(lab["boxes"], lab["track_ids"]):
        ann.frames.setdefault(int(f), []).append(
            Instance(track_id=int(tid), box=(int(x1), int(y1), int(x2), int(y2)))
        )
    return ann


def fa_at_pd90(sweep: list[dict]) -> dict:
    """工作点：recall ≥ 0.90 的阈值中 F_a_frm 最小者（G-虚警口径）。"""
    ok = [s for s in sweep if s["recall"] >= 0.90]
    if not ok:
        best = max(sweep, key=lambda s: s["recall"])
        return {"available": False, "note": f"P_d=0.90 未达（最高 R={best['recall']:.3f}）",
                "recall": best["recall"], "fa_frm": best["fa_frm"],
                "fa_pix_e6": best["fa_pix_e6"],
                "thr": best.get("conf_thr", best["iou_thr"])}
    best = min(ok, key=lambda s: s["fa_frm"])
    return {"available": True, "thr": best.get("conf_thr", best["iou_thr"]),
            "recall": best["recall"],
            "fa_frm": best["fa_frm"], "fa_pix_e6": best["fa_pix_e6"],
            "precision": best["precision"], "f1": best["f1"]}


def subset_of(seq_meta: dict) -> list[str]:
    subs = [f"{seq_meta['daytime']['value']}", f"{seq_meta['scene']['value']}",
            f"{seq_meta['daytime']['value']}×{seq_meta['scene']['value']}"]
    for tag in DIFFICULTY_TAGS:
        if seq_meta["difficulty_tags"].get(tag, {}).get("value"):
            subs.append(tag)
    return subs


def measure_flops_fps(model, model_type: str, device: str) -> dict:
    from thop import profile

    if model_type == "mshnet":
        x = torch.rand(1, 1, 480, 640, device=device)
        per_frame_div = 1
    else:
        x = torch.rand(1, 32, 1, 480, 640, device=device)
        per_frame_div = 32
    flops, params = profile(model.eval(), inputs=(x,), verbose=False)
    torch.cuda.synchronize() if device == "cuda" else None
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(30):
            model(x)
        torch.cuda.synchronize() if device == "cuda" else None
        dt = (time.perf_counter() - t0) / 30
    return {
        "params_m": round(params / 1e6, 3),
        "flops_g_per_frame": round(flops / 1e9 / per_frame_div, 2),
        "fps_fullres": round(1.0 / dt, 1),
    }


def eval_split(
    model, model_type: str, split: str, manifest: dict, cache_root: str,
    device: str, out_dir: Path, write_txt: bool, limit_seqs: int = 0,
) -> dict:
    seqs = manifest["splits"][split]["seqs"]
    if limit_seqs:
        seqs = seqs[:limit_seqs]  # dry-run/冒烟用
    meta_by_id = {s["seq_id"]: s for s in manifest["sequences"]}
    all_boxes = {t: [] for t in THR_SWEEP}
    all_gts: list = []
    n_frames_total = 0
    official_results = {}
    per_seq_rows = []
    subset_pred: dict[str, list] = defaultdict(list)
    subset_gt: dict[str, list] = defaultdict(list)
    subset_frames: dict[str, int] = defaultdict(int)
    for sid in seqs:
        cache = load_seq_cache(cache_root, sid, with_reg=model_type != "mshnet")
        n_frames = int(meta_by_id[sid]["n_frames"])
        n_frames_total += n_frames
        if model_type == "mshnet":
            prob = infer_mshnet(model, cache, device)
        else:
            # 方案 2.5 推理约定；时序窗口与训练侧同口径对齐（reg.npz bridge_M 复合）
            prob = infer_temporal(model, cache, device, T=32, stride=24, warmup=8)
        boxes_by_thr = boxes_at_thresholds(prob, THR_SWEEP)
        for t, bl in boxes_by_thr.items():
            all_boxes[t].extend(bl)

        gts = gt_boxes_from_cache(cache)
        all_gts.extend(gts)

        # 官方评分口径：主阈值 0.5 框 → 逐帧贪心跟踪 → 中心点 Instance
        tracker = GreedyTracker(vmax=8.0, coast_max=10)
        frame_boxes: dict[int, list] = defaultdict(list)
        for d in boxes_by_thr[PRIMARY_THR]:
            frame_boxes[d.frame].append(d)
        tracked = []
        for f in range(1, n_frames + 1):
            tracked.extend(tracker.update(frame_boxes.get(f, []), f))
        pred: dict[int, list[Instance]] = defaultdict(list)
        for d in tracked:
            pred[d.frame].append(
                Instance(track_id=d.track_id, box=(d.cx, d.cy, d.cx, d.cy))
            )
        gt_ann = gt_seq_annotation(cache)
        res = official_score.score_sequence(gt_ann, pred, mode="literal")
        official_results[sid] = res.as_dict()

        if write_txt:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_official_txt(out_dir / f"{sid}.txt", sid, pred, frames_per_seq=n_frames)
            with open(out_dir / f"{sid}_boxes.txt", "w", encoding="utf-8") as fp:
                for d in tracked:
                    fp.write(f"{d.frame} {d.track_id} {d.score:.4f} "
                             f"{d.x1} {d.y1} {d.x2} {d.y2}\n")

        per_seq_rows.append({
            "seq_id": sid, "n_gt": len(gts),
            "n_pred@0.5": len(boxes_by_thr[PRIMARY_THR]),
            "official_total": res.total,
        })

        # 分子集（8.3）：主阈值框 + GT 按序列属性归组
        subs = subset_of(meta_by_id[sid])
        for s in subs:
            subset_gt[s].extend(gts)
            subset_frames[s] += n_frames
            subset_pred[s].extend(boxes_by_thr[PRIMARY_THR])

    # 主阈值指标 + AP + 阈值扫描
    primary = evaluate_boxes(all_boxes[PRIMARY_THR], all_gts, n_frames_total)
    ap = evaluate_ap(all_boxes[PRIMARY_THR], all_gts)
    sweep = [{**evaluate_boxes(all_boxes[t], all_gts, n_frames_total), "conf_thr": t}
             for t in THR_SWEEP]
    result = {
        "split": split,
        "n_seqs": len(seqs), "n_frames": n_frames_total,
        "n_gt": len(all_gts),
        "primary_thr": PRIMARY_THR,
        "primary": primary,
        "ap": ap,
        "thr_sweep": sweep,
        "fa_at_pd90": fa_at_pd90(sweep),
        "official": {
            "detection_total": sum(r["detection"] for r in official_results.values()),
            "continuity_total": sum(r["continuity"] for r in official_results.values()),
            "grand_total": sum(r["total"] for r in official_results.values()),
            "full_score": official_score.FULL_SCORE_VAL_OFFICIAL if split == "val-official" else None,
            "ratio": (sum(r["total"] for r in official_results.values())
                      / official_score.FULL_SCORE_VAL_OFFICIAL
                      ) if split == "val-official" else None,
            "per_seq": official_results,
        },
        "per_seq": per_seq_rows,
    }

    # 分子集指标（8.3）
    result["subsets"] = {}
    for s in sorted(set(subset_gt) | set(subset_pred)):
        result["subsets"][s] = {
            **evaluate_boxes(subset_pred[s], subset_gt[s], subset_frames[s]),
            "ap50": evaluate_ap(subset_pred[s], subset_gt[s])["ap50"],
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["val-int", "val-official"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-efficiency", action="store_true")
    ap.add_argument("--limit-seqs", type=int, default=0, help="每 split 只评前 N 段（冒烟）")
    args = ap.parse_args()

    model, cfg = load_model_from_ckpt(args.ckpt, args.device)
    model_type = cfg.model.type
    manifest = json.load(open(REPO / "data" / "manifests" / str(cfg.data.manifest), encoding="utf-8"))
    cache_root = str(cfg.data.get("cache_root", REPO / "data" / "cache" / "ittd"))
    exp_dir = Path(args.ckpt).parent.parent
    out_path = Path(args.out) if args.out else exp_dir / "eval_report.json"
    txt_dir = exp_dir / "eval_preds"

    print(f"eval_baseline: model={model_type} ckpt={args.ckpt}")
    report_model_params(model)
    report = {
        "ckpt": args.ckpt, "model_type": model_type,
        "manifest": str(cfg.data.manifest),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for split in args.splits:
        print(f"=== 评测 {split} ===")
        write_txt = split == "val-official"
        report[split] = eval_split(model, model_type, split, manifest, cache_root,
                                   args.device, txt_dir, write_txt, args.limit_seqs)
        r = report[split]["primary"]
        print(f"  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"AP50={report[split]['ap']['ap50']:.4f} "
              f"mAP={report[split]['ap']['map_5095']:.4f} "
              f"Fa_frm={r['fa_frm']:.4f} Fa_pix={r['fa_pix_e6']:.2f}e-6 "
              f"官方分={report[split]['official']['grand_total']}")
    if not args.skip_efficiency:
        print("=== 效率测量（thop/FPS）===")
        report["efficiency"] = measure_flops_fps(model, model_type, args.device)
        print(f"  {report['efficiency']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=1)
    print(f"报告 → {out_path}")


if __name__ == "__main__":
    main()
