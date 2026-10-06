"""IRSTD-1k 评测（M2-L1 文献对拍）：官方 metric.py 口径的 mIoU + P_d/F_a。

官方语义（github.com/ying-fu/MSHNet utils/metric.py，README 报告值即此口径）：
  - mIoU：logits>0（≡ sigmoid>0.5）二值化，逐像素 inter/union 累计（单类）；
  - P_d/F_a：同一阈值下 8 连通域分析；预测域质心与 GT 域质心距离 <3 px 判命中，
    P_d = 命中数/GT 域数；F_a = 未命中预测域面积和/(256²·图数)；
  - 输出 bin0（score_thresh=0）即 README 数字口径；
  - 对拍目标：IRSTD-1k mIoU 67.87 / P_d 92.86 / F_a 8.88×10⁻⁶（官方重检版）。

skimage 依赖以 cv2 连通域等价实现（connectivity=2 ≡ cv2 8 连通）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.data.irstd1k_dataset import Irstd1k
from dsld.models.mshnet import MSHNetBaseline

PUBLISHED = {"miou": 0.6787, "pd": 0.9286, "fa_e6": 8.88}  # 官方 README（重检版）


@torch.no_grad()
def evaluate(model, loader, device: str, size: int = 256) -> dict:
    model.eval()
    inter = union = 0
    fa_sum = 0.0
    pd_hit = 0
    pd_target = 0
    n_img = 0
    for batch in loader:
        img = batch["windows"].to(device)
        mask = batch["target"]
        logits = model(img)  # [B,1,256,256]
        pred = (logits[:, 0] > 0).cpu().numpy().astype(np.uint8)
        gt = (mask[:, 0].numpy() > 0.5).astype(np.uint8)
        for p, g in zip(pred, gt):
            # mIoU（单类累计）
            inter += int(np.logical_and(p, g).sum())
            union += int(np.logical_or(p, g).sum())
            # P_d / F_a（官方 PD_FA bin0）
            n_pred, _, stats, cents = cv2.connectedComponentsWithStats(p, connectivity=8)
            n_gt, _, _, cents_g = cv2.connectedComponentsWithStats(g, connectivity=8)
            pd_target += n_gt - 1
            matched = []
            hit = 0
            pred_areas = [int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n_pred)]
            for gi in range(1, n_gt):
                cy, cx = cents_g[gi]
                for k in range(1, n_pred):
                    if k in matched:
                        continue
                    y, x = cents[k]
                    if float(np.hypot(x - cx, y - cy)) < 3:
                        matched.append(k)
                        hit += 1
                        break
            pd_hit += hit
            unmatched = [a for k, a in enumerate(pred_areas) if (k + 1) not in matched]
            fa_sum += float(np.sum(unmatched))
            n_img += 1
    return {
        "miou": round(inter / max(union, 1), 4),
        "pd": round(pd_hit / max(pd_target, 1), 4),
        "fa_e6": round(fa_sum / (size * size * n_img) * 1e6, 2),
        "n_images": n_img,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=r"D:\Datasets\IRSTD-1k\IRSTD-1k")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MSHNetBaseline(in_ch=state["cfg"]["model"].get("in_ch", 3))
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device)

    ds = Irstd1k(args.root, mode="val")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    t0 = time.perf_counter()
    res = evaluate(model, loader, args.device)
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["published"] = PUBLISHED
    res["delta_vs_published"] = {
        "miou_pt": round((res["miou"] - PUBLISHED["miou"]) * 100, 2),
        "pd_pt": round((res["pd"] - PUBLISHED["pd"]) * 100, 2),
        "fa_ratio": round(res["fa_e6"] / PUBLISHED["fa_e6"], 2) if PUBLISHED["fa_e6"] else None,
    }
    out = Path(args.ckpt).parent.parent / "irstd1k_eval.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
