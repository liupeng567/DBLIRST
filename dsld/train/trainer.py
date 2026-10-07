"""DSLD 训练循环（方案 9.1 main.py 规范，main.py 与 scripts/train.py 共用）。

M0 范围：训练链路打通——启动四步（配置树 / 环境 / 参数量 / 权重加载报告）+
按轮训练（tqdm 双进度条）+ 轮摘要 + metrics.jsonl + 按轮断点。
M2 范围：接入 ITTD 真实窗口数据集与两基线模型（mshnet 单帧 / msd3d 时序），
SLSIoU 多级深监督（官方 MSHNet 口径），Adagrad/AdamW 可选，bf16 AMP，梯度累积。
真实 DSLD 模型/损失在 M3–M5 逐阶段替换，接口保持不变。
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsld.train.losses import SLSIoULoss

REPO = Path(__file__).resolve().parents[2]

# 训练全程输入尺寸恒定（全幅 480×640 / crop 档 240×320），启用 cuDNN 自动算法调优
torch.backends.cudnn.benchmark = True


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "(no git)"


def print_config_tree(cfg) -> None:
    print("=" * 72)
    print("① 生效参数配置（OmegaConf 树）")
    print("=" * 72)
    print(OmegaConf.to_yaml(cfg))


def print_environment(cfg) -> None:
    print("=" * 72)
    print("② 运行环境")
    print("=" * 72)
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    mem = [f"{torch.cuda.get_device_properties(i).total_memory / 2**30:.0f}GB"
           for i in range(torch.cuda.device_count())]
    print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"  GPU: {list(zip(gpus, mem)) or '无（CPU）'}")
    print(f"  seed={cfg.train.seed}  git={git_commit()}")
    manifest = REPO / "data" / "manifests" / f"{cfg.data.manifest}"
    if manifest.exists():
        print(f"  manifest={cfg.data.manifest}  md5={md5_of(manifest)}")
    else:
        print(f"  manifest={cfg.data.manifest}  [未找到]")
    print(f"  val_official_read_only={cfg.data.val_official_read_only}"
          "（77-87 只读，训练入口禁用）")


def report_model_params(model: nn.Module) -> dict[str, int]:
    print("=" * 72)
    print("③ 逐模块参数量（对齐方案 4.7 预算口径）")
    print("=" * 72)
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    total = 0
    for g, n in groups.items():
        print(f"  {g:12s} {n/1e6:8.3f} M")
        total += n
    print(f"  {'TOTAL':12s} {total/1e6:8.3f} M   (方案闸门: ≤ 5M)")
    return groups


def load_checkpoint(model: nn.Module, ckpt: str | None, optimizer=None) -> tuple[int, dict]:
    """加载权重；返回 (start_epoch, 附带状态)。optimizer 传入时一并恢复（断点续训）。"""
    print("=" * 72)
    print("④ 权重加载情况")
    print("=" * 72)
    if not ckpt:
        print("  No checkpoint → training from scratch（官方初始化，τ 取范围中点）")
        return 0, {}
    path = Path(ckpt)
    if not path.exists():
        raise FileNotFoundError(f"ckpt 不存在: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    own = model.state_dict()
    matched, missing, unexpected, shape_bad = [], [], [], []
    for k, v in sd.items():
        if k not in own:
            unexpected.append(k)
        elif own[k].shape != v.shape:
            shape_bad.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(own[k].shape)}")
        else:
            matched.append(k)
    missing = [k for k in own if k not in sd]
    print(f"  来源: {path}")
    print(f"  matched={len(matched)}  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  [WARN] missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  [WARN] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if shape_bad:
        for s in shape_bad:
            print(f"  [FATAL] 形状不匹配: {s}")
        raise RuntimeError("存在形状不匹配的键，拒绝启动（方案 9.1 规范②）")
    model.load_state_dict(sd, strict=True)
    start_epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
    if optimizer is not None and isinstance(state, dict) and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        print(f"  优化器状态已恢复（Adagrad 累积量/AdamW 动量随 ckpt 续用）")
    for extra in ("best_metric", "ema", "history"):
        if isinstance(state, dict) and extra in state:
            print(f"  附带状态: {extra}")
    print(f"  start_epoch = {start_epoch + 1}（断点续训）")
    return start_epoch, state if isinstance(state, dict) else {}


def build_model(cfg) -> nn.Module:
    """按 cfg.model.type 分发：dryrun（M0）/ mshnet·msd3d（M2 基线）/ dsld_core（M3）。"""
    mtype = cfg.model.type
    if mtype == "dryrun":
        from dsld.models.dryrun_net import DryRunNet

        return DryRunNet(in_ch=cfg.model.in_ch, width=cfg.model.width)
    if mtype == "mshnet":
        from dsld.models.mshnet import MSHNetBaseline

        return MSHNetBaseline(in_ch=cfg.model.get("in_ch", 1))
    if mtype == "msd3d":
        from dsld.models.msd3d import TemporalBaseline

        ch = cfg.model.get("channels", None)
        return TemporalBaseline(in_ch=cfg.model.get("in_ch", 1),
                                channels=tuple(ch) if ch else None)
    if mtype == "dsld_core":
        from dsld.models.dsld_core import DsldCore

        liq = cfg.model.liquid
        fb = cfg.model.feedback
        return DsldCore(
            in_ch=cfg.model.get("in_ch", 1),
            width=cfg.model.encoder.get("width", 1.0),
            c_main=cfg.model.neck.get("c_main", 32),
            c_h=liq.get("h", 32),
            liquid_mode=liq.get("mode", "dual"),
            tau_b=tuple(liq.tau_b.get(k) for k in ("min", "max", "init")),
            tau_t=tuple(liq.tau_t.get(k) for k in ("min", "max", "init")),
            mask_radius=fb.get("mask_radius", 5),
            mask_decay=fb.get("mask_decay", 0.9),
            alpha_th=fb.get("alpha_th", 0.6),
            mask_m_max=fb.get("m_max", 0.8),
            mask_softness=fb.get("softness", 0.0),
            detach_every=liq.get("detach_every", 0),
            use_checkpoint=liq.get("use_checkpoint", False),
            s_max=liq.get("s_max", 0.7),
            kappa_max=liq.get("kappa_max", 0.5),
            m_scale=liq.get("m_scale", 0.5),
            state_dependent=liq.get("state_dependent", False),
            bound_f=liq.get("bound_f", 0.0),  # 已废弃，>0 时模型内告警
        )
    raise RuntimeError(f"model.type={mtype} 未实现")


def build_dataloader(cfg) -> DataLoader:
    """M0: 合成窗口；M2: ITTD 真实窗口/单帧数据集；L1: IRSTD-1k（source 切换）。"""
    if cfg.data.get("mode", "dryrun") == "dryrun":
        from dsld.data.dryrun_dataset import DryRunWindows

        return DataLoader(
            DryRunWindows(
                n_windows=cfg.data.dryrun.n_windows,
                T=cfg.train.window.T,
                H=cfg.data.dryrun.height,
                W=cfg.data.dryrun.width,
                seed=cfg.train.seed,
            ),
            batch_size=cfg.train.batch.windows_per_gpu,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
    if cfg.data.get("source", "ittd") == "irstd1k":
        from dsld.data.irstd1k_dataset import Irstd1k

        ds = Irstd1k(root=str(cfg.data.irstd1k_root), mode="train")
        print(f"  数据集: IRSTD-1k trainval n={len(ds)} 256×256（官方管线）")
        return DataLoader(
            ds,
            batch_size=cfg.train.batch.windows_per_gpu,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
    from dsld.data.ittd_window_dataset import IttdWindows

    crop = cfg.data.get("crop", None)
    ds = IttdWindows(
        manifest_path=str(REPO / "data" / "manifests" / str(cfg.data.manifest)),
        cache_root=str(cfg.data.get("cache_root", REPO / "data" / "cache" / "ittd")),
        split=cfg.data.get("split", "train-int"),
        mode=cfg.data.mode,
        T=cfg.train.window.T,
        stride=cfg.train.window.get("stride", 8),
        crop=tuple(crop) if crop else None,
        augment=cfg.data.get("augment", True),
        seed=cfg.train.seed,
        limit_seqs=cfg.data.get("limit_seqs", 0),
    )
    print(f"  数据集: mode={cfg.data.mode} split={cfg.data.get('split', 'train-int')}"
          f" n_samples={len(ds)} T={cfg.train.window.T}"
          f" crop={tuple(crop) if crop else '全幅'}")
    return DataLoader(
        ds,
        batch_size=cfg.train.batch.windows_per_gpu,
        shuffle=True,
        num_workers=cfg.data.get("num_workers", 2),
        persistent_workers=cfg.data.get("num_workers", 2) > 0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_dl_worker_init,
    )


def _dl_worker_init(_wid: int) -> None:
    # 多 worker 下禁用 cv2 内部线程池：每窗 32 次 warpAffine 若各开线程池，
    # 与 num_workers 进程叠加造成 CPU 超订（M1 预处理同款教训）
    import cv2
    cv2.setNumThreads(0)


def _make_optimizer(cfg, model: nn.Module):
    name = cfg.train.optim.get("name", "adamw")
    if name == "adagrad":
        # MSHNet 官方口径：Adagrad，恒定 lr（自适应累积自带衰减）
        return torch.optim.Adagrad(model.parameters(), lr=cfg.train.optim.lr)
    # AdamW（表 T）：wd 不作用于 τ/门控偏置/GN——1 维参数（偏置、归一化、θ_τ/θ_β）全免 wd
    if cfg.model.type == "dsld_core":
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.train.optim.wd},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.train.optim.lr,
        )
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.train.optim.lr, weight_decay=cfg.train.optim.wd
    )


def _make_scheduler(cfg, optim, iters_total: int):
    """表 T：warmup + cosine → cosine_min。Adagrad（官方恒定 lr）返回 None。"""
    if cfg.train.optim.get("name", "adamw") == "adagrad":
        return None
    warmup = cfg.train.optim.get("warmup", 1000)
    min_lr = cfg.train.optim.get("cosine_min", 3.0e-5)
    base_lr = cfg.train.optim.lr

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, iters_total - warmup))
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def _build_optimizer(cfg, model: nn.Module, iters_total: int):
    optim = _make_optimizer(cfg, model)
    return optim, _make_scheduler(cfg, optim, iters_total)


def _flatten_frames(t: torch.Tensor) -> torch.Tensor:
    """[B,T,1,H,W] → [B·T,1,H,W]（SLSIoU 按帧计算）。"""
    return t.reshape(-1, *t.shape[2:])


def _baseline_loss(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    loss_fn: SLSIoULoss,
    warm_epoch: int,
    epoch: int,
    temporal: bool,
) -> torch.Tensor:
    """官方 MSHNet 多级深监督口径：final + Σ aux（标签逐级空间池化），除以级数。"""
    aux, final = model.forward_train(x, warm_flag=epoch > warm_epoch)
    labels = target
    if temporal:  # [B,T,1,H,W] → [B·T,1,H,W]
        final = _flatten_frames(final)
        labels = _flatten_frames(labels)
        aux = [_flatten_frames(a) for a in aux]
    else:  # MSHNet [B,1,H,W]；ITTD 目标 [B,1,1,H,W]，IRSTD-1k 目标 [B,1,256,256]
        if labels.dim() == 5:
            labels = labels[:, 0]
        elif labels.dim() == 3:
            labels = labels.unsqueeze(1)
    pool_t = torch.nn.MaxPool3d((1, 2, 2)) if temporal else torch.nn.MaxPool2d(2, 2)
    loss = loss_fn(final, labels, warm_epoch, epoch)
    for j, m in enumerate(aux):
        if j > 0:
            labels = pool_t(labels)
        loss = loss + loss_fn(m, labels, warm_epoch, epoch)
    return loss / (len(aux) + 1)


def _dsld_core_loss(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    quality: torch.Tensor | None,
    weights: dict,
) -> tuple[torch.Tensor, dict]:
    """M3 DSLD 损失（7.1 ①②③）：L = λ₁·L_seg + λ₂·L_recon + λ₃·L_dec。

    核心输出 fp32（4.8-①），损失在 autocast 外按 fp32 计算。
    返回 (loss, 分量表)。
    """
    from dsld.train.losses import decouple_loss, focal_dice_loss, recon_loss

    out = model(x, quality=quality)
    seg = focal_dice_loss(out["logits"], target)
    rec = recon_loss(out["y_b"], out["x_main"], out["m_tgt"], target)
    parts = {"seg": float(seg), "recon": float(rec),
             "recon_valid_frac": round(getattr(recon_loss, "last_valid_frac", 1.0), 6)}
    loss = weights.get("seg", 1.0) * seg + weights.get("recon", 0.5) * rec
    if model.liquid_mode == "dual":  # 单状态无解耦对象（消融 a 对照）
        dec = decouple_loss(out["h_t"], out["h_b"])
        loss = loss + weights.get("decouple", 0.1) * dec
        parts["decouple"] = float(dec)
    parts["total"] = float(loss)
    return loss, parts


QUICK_THR = [0.3, 0.5, 0.7, 0.9]  # 快评精简阈值扫描（全链 7 点）


def _quick_eval(
    model: nn.Module,
    cfg,
    device: str,
    manifest_path: str | None = None,
    cache_root: str | None = None,
) -> dict:
    """方案 7.5 快评：val-int 固定子集（前 n_seqs 段 × n_windows 个固定起点窗）。

    口径与 eval 全链一致：窗口对齐锚点 → 前向 → 概率回投原始帧坐标（预热帧不计）
    → IoU≥0.5 框级 P/R/F1 + F_a 双口径 + F_a@P_d=0.90 工作点（复用 fa_at_pd90 同一
    实现）。另记录：背景吸收监控（背景残差 RMS、残差 SCR=目标框/背景残差——M3 无
    抑制图的过渡口径，方案 8.1 的 BSF/SCRG 待 M4 以 5.3 抑制图重算）、α 分布
    （临时门控图统计）、τ 分位数。子集与起点确定性（np.linspace 固定），跨期可比；
    子集描述（seq_id+starts）随记录落盘。
    """
    import cv2
    import numpy as np

    from dsld.data.manifest import load_manifest
    from dsld.data.preprocess.normalize import correct_frame, normalize_frame
    from dsld.data.preprocess.register import (
        is_identity_warp,
        warp_frame,
        window_anchor_warps,
    )
    from dsld.eval.infer_seq import gt_boxes_from_cache, load_seq_cache
    from dsld.eval.map_iou import evaluate_boxes
    from dsld.eval.mask_to_boxes import Box, mask_to_boxes

    T = int(cfg.train.window.T)
    warmup = min(8, max(1, T // 4))  # T=32 → 8（2.5 推理口径）；小 T 冒烟按比例
    qe = cfg.train.get("quick_eval") or {}
    n_seqs = int(qe.get("n_seqs", 4))
    n_windows = int(qe.get("n_windows", 5))
    manifest_path = manifest_path or str(REPO / "data" / "manifests" / str(cfg.data.manifest))
    cache_root = cache_root or str(cfg.data.get("cache_root", REPO / "data" / "cache" / "ittd"))
    manifest = load_manifest(manifest_path)
    val_seqs = manifest["splits"]["val-int"]["seqs"]
    # 返工 A5：显式子集优先（须覆盖两种配准域——KLT 主导段 + FM 主导段 67/68，
    # 评审 1.4：前 n 段子集 [21,22,23,38] 对 FM 配准域完全不可见）
    seq_ids = qe.get("seq_ids") or None
    if seq_ids:
        missing = [s for s in seq_ids if s not in val_seqs]
        if missing:
            print(f"[quick_eval][WARN] seq_ids 不在 val-int，已跳过: {missing}")
        seqs = [s for s in seq_ids if s in val_seqs]
        if not seqs:
            raise RuntimeError("quick_eval.seq_ids 过滤后为空（均不在 val-int）")
    else:
        seqs = val_seqs[:n_seqs]

    all_boxes: dict[float, list] = {t: [] for t in QUICK_THR}
    all_gts: list[Box] = []
    n_frames_eval = 0
    alpha_all: list[float] = []
    bg_resid_all: list[float] = []
    bg_frac_all: list[float] = []
    resid_scr_all: list[float] = []
    subset_desc: list[dict] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sid in seqs:
                cache = load_seq_cache(cache_root, int(sid), with_reg=True)
                frames_u8 = cache["frames"]
                n, H_img, W_img = frames_u8.shape
                starts = np.linspace(0, n - T, n_windows).astype(int)  # 固定起点
                subset_desc.append({"seq_id": int(sid), "starts": starts.tolist()})
                for s in starts:
                    Ws = window_anchor_warps(cache["reg"], int(s), T)
                    win = np.empty((T, H_img, W_img), np.float32)
                    for j in range(T):
                        f = s + j
                        xc = correct_frame(np.asarray(frames_u8[f]), cache["nuc"], cache["dead"])
                        win[j] = normalize_frame(xc, float(cache["stats"][f, 0]),
                                                 float(cache["stats"][f, 1]))
                    for j in range(1, T):
                        if not is_identity_warp(Ws[j]):
                            win[j] = warp_frame(win[j], Ws[j])
                    x = torch.from_numpy(win[:, None])[None].to(device)
                    out = model(x, quality=None)
                    logits = out["logits"] if isinstance(out, dict) else out
                    prob = torch.sigmoid(logits.float()[0, :, 0]).cpu().numpy()  # [T,H,W]
                    # 输出回投原始帧坐标（与 infer_temporal 同一语义）
                    for j in range(warmup, T):
                        if not is_identity_warp(Ws[j]):
                            prob[j] = cv2.warpAffine(
                                prob[j], np.asarray(Ws[j], np.float32), (W_img, H_img),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    alpha = out["alpha"].float().cpu().numpy()  # [1,T,1,h,w]
                    alpha_all.extend(
                        float(v) for v in alpha[0, warmup:, 0].ravel()[::16])  # 抽样 1/16
                    # 背景吸收监控（M3 口径，无标度）：背景残差 RMS + 残差 SCR
                    # （GT 框内残差均值 / 背景残差中位——目标应高残差、背景应被吸收；
                    #   方案 8.1 的 BSF/SCRG 需抑制图，M4 起以 5.3 输出重算）
                    x_main = out["x_main"][0, -1].float().cpu().numpy()  # [C,h,w]
                    resid = np.abs(x_main - out["y_b"][0, -1].float().cpu().numpy()).mean(0)
                    m_tgt = out["m_tgt"][0, -1, 0].float().cpu().numpy()
                    gt2 = np.zeros_like(resid)
                    fno_last = int(s + T)  # 末帧 1-based 帧号
                    for f, x1, y1, x2, y2 in cache["labels"]["boxes"]:
                        if int(f) == fno_last:
                            gt2[max(y1 // 2, 0):y2 // 2 + 1, max(x1 // 2, 0):x2 // 2 + 1] = 1.0
                    # 背景集合只按 GT 排除（m_tgt 随训练扩散会把背景集合挤空，指标退化）
                    bg = gt2 < 0.5
                    if bg.sum() > 100:
                        bg_med = float(np.median(resid[bg]))
                        bg_resid_all.append(float(np.sqrt((resid[bg] ** 2).mean())))
                        bg_frac_all.append(float(bg.mean()))
                        if gt2.sum() > 0:
                            resid_scr_all.append(float(resid[gt2 > 0.5].mean()
                                                       / max(bg_med, 1e-6)))
                    for j in range(warmup, T):
                        fno = int(s + j + 1)  # 1-based 帧号
                        for t in QUICK_THR:
                            all_boxes[t].extend(
                                mask_to_boxes(prob[j], frame=fno, thr=t, min_area=4))
                        n_frames_eval += 1
                    valid = {int(s + j + 1) for j in range(warmup, T)}
                    all_gts.extend(g for g in gt_boxes_from_cache(cache) if g.frame in valid)
    finally:
        model.train(was_training)

    sweep = [{**evaluate_boxes(all_boxes[t], all_gts, n_frames_eval), "conf_thr": t}
             for t in QUICK_THR]
    primary = next(s for s in sweep if s["conf_thr"] == 0.5)
    from scripts.eval_baseline import fa_at_pd90  # 同一工作点实现，口径零偏差

    pd90 = fa_at_pd90(sweep)
    # α"越阈占比"阈值跟随模型（返工 A3：α_th 0.5→0.6，监控与掩码判定同一口径）
    alpha_th = float(getattr(getattr(model, "core", model), "alpha_th", 0.5))
    rec = {
        "n_seqs": len(seqs), "n_frames": n_frames_eval, "n_gt": len(all_gts),
        "subset": subset_desc,
        "primary": primary, "thr_sweep": sweep,
        "fa_frm_pd90": round(pd90["fa_frm"], 6),
        "fa_pix_e6_pd90": round(pd90["fa_pix_e6"], 4),
        "recall_pd90": round(pd90["recall"], 6),
        "pd90_available": bool(pd90.get("available", False)),
        "bg_resid_rms": round(float(np.median(bg_resid_all)), 6) if bg_resid_all else 0.0,
        "bg_frac": round(float(np.median(bg_frac_all)), 4) if bg_frac_all else 0.0,
        "resid_scr": round(float(np.median(resid_scr_all)), 4) if resid_scr_all else 0.0,
        "alpha_mean": round(float(np.mean(alpha_all)), 4) if alpha_all else 0.0,
        "alpha_frac_high": (round(float(np.mean(np.array(alpha_all) > alpha_th)), 4)
                            if alpha_all else 0.0),
        "alpha_p99": (round(float(np.percentile(alpha_all, 99)), 4)
                      if alpha_all else 0.0),
    }
    if cfg.model.type == "dsld_core":
        rec.update(model.tau_report())
    return rec


def early_stop_step(state: dict, fa_frm: float, es_cfg: dict) -> bool:
    """表 T 早停（纯函数）：监控 F_a@P_d 最小化，patience 个 eval 周期无改善即停。

    state = {"best": float, "patience": int}（原地更新）；fa_frm=inf（P_d 未达
    0.90 的 eval 周期）不计改善只计耐心。返回 True 表示应终止训练。
    """
    if not es_cfg.get("enabled", False):
        return False
    patience = int(es_cfg.get("patience", 8))
    if fa_frm < state["best"] - 1e-9:
        state["best"], state["patience"] = fa_frm, 0
        return False
    state["patience"] += 1
    return state["patience"] >= patience


def run_training(cfg) -> dict:
    """完整训练入口：返回轮历史摘要。train.resume 指向 ckpt 时断点续训。"""
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nDSLD 训练启动 @ {t_start}")
    print_config_tree(cfg)
    print_environment(cfg)

    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    report_model_params(model)

    # 断点续训：先建优化器再加载（Adagrad 累积量 / AdamW 动量随 ckpt 恢复），
    # 调度器待 len(loader) 确定后绑定同一优化器并定位到恢复步。
    resume_path = cfg.train.get("resume", None)
    start_epoch, resume_step, optim = 0, 0, None
    if resume_path:
        optim = _make_optimizer(cfg, model)
        start_epoch, resume_state = load_checkpoint(model, resume_path, optimizer=optim)
        resume_step = int(resume_state.get("global_step", 0))
    elif cfg.train.get("ckpt", None):
        load_checkpoint(model, cfg.train.get("ckpt", None))  # 仅权重热启动（不复用优化器/轮次）

    loader = build_dataloader(cfg)
    accum = cfg.train.batch.get("accum", 1)
    iters_total = max(1, len(loader) // max(1, accum)) * cfg.train.epochs
    if optim is None:
        optim = _make_optimizer(cfg, model)
    sched = _make_scheduler(cfg, optim, iters_total)
    if resume_step and sched is not None:  # cosine 相位定位到恢复步
        sched.last_epoch = resume_step
        for pg, lr0 in zip(optim.param_groups, sched.get_lr()):
            pg["lr"] = lr0
    loss_fn = SLSIoULoss()
    temporal = cfg.model.type == "msd3d"
    warm_epoch = cfg.train.get("warm_epoch", 5)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        str(cfg.train.get("amp", "none")), None
    )
    grad_clip = float(cfg.train.get("grad_clip", 1.0) or 0.0)
    steps_per_epoch = len(loader)
    exp_dir = REPO / "experiments" / cfg.experiment.name
    ckpt_dir = exp_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.jsonl"

    print("=" * 72)
    print(f"⑤ 按轮训练：第 {start_epoch + 1}–{cfg.train.epochs} 轮 × {steps_per_epoch}"
          f" step × accum={accum}（loss=SLSIoU 多级深监督 warm_epoch={warm_epoch}，"
          f"AMP={'off' if amp_dtype is None else str(cfg.train.amp)}）")
    print("=" * 72)

    history = []
    outer = tqdm(range(start_epoch + 1, cfg.train.epochs + 1), desc="epoch", position=0)
    global_step = resume_step
    qe_cfg = OmegaConf.to_container(cfg.train.quick_eval, resolve=True) \
        if cfg.train.get("quick_eval") else {}
    es_cfg = OmegaConf.to_container(cfg.train.early_stop, resolve=True) \
        if cfg.train.get("early_stop") else {}
    es_state = {"best": float("inf"), "patience": 0}
    es_min_steps = int(es_cfg.get("min_steps", 0))
    stopped = False
    recon_warned = False

    def save_ckpt(name: str, epoch_no: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "epoch": epoch_no,
                "global_step": global_step,
                "cfg": OmegaConf.to_container(cfg),
                "history": history,
            },
            ckpt_dir / name,
        )
        tqdm.write(f"[ckpt] 已保存 {ckpt_dir / name}（含优化器状态，可断点续训）")

    for epoch in outer:
        model.train()
        pbar = tqdm(loader, desc=f"train {epoch}", position=1, leave=False)
        total_loss = 0.0
        n_steps = 0
        loss_parts: dict | None = None
        optim.zero_grad()
        for step, batch in enumerate(pbar, 1):
            x = batch["windows"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if cfg.model.type == "dryrun":  # M0 链路验证占位损失，保持不变
                out = model(x)
                loss = nn.functional.l1_loss(out, target)
            elif cfg.model.type == "dsld_core":
                quality = batch.get("quality")
                if quality is not None:
                    quality = quality.to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    loss, loss_parts = _dsld_core_loss(
                        model, x, target, quality,
                        OmegaConf.to_container(cfg.train.loss, resolve=True)
                        if cfg.train.get("loss") else {},
                    )
                if not torch.isfinite(loss):  # M3 Gate：训练 50k 无 NaN，坏步立即终止定位
                    raise RuntimeError(
                        f"loss NaN/Inf @ epoch{epoch} step{step} parts={loss_parts}")
            else:
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    loss = _baseline_loss(
                        model, x, target, loss_fn, warm_epoch, epoch, temporal
                    )
            (loss / accum).backward()
            gn = torch.tensor(0.0)
            if step % accum == 0 or step == steps_per_epoch:
                if grad_clip > 0:
                    gn = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    gn = torch.tensor(0.0)
                optim.step()
                if sched is not None:
                    sched.step()
                optim.zero_grad()
                global_step += 1
                # 7.5 快评：每 every 优化步在 val-int 固定子集快评（dsld_core）
                if (qe_cfg.get("enabled", False) and cfg.model.type == "dsld_core"
                        and global_step > 0 and global_step % int(qe_cfg.get("every", 2500)) == 0):
                    qrec = {"kind": "quick_eval", "time": time.strftime("%H:%M:%S"),
                            "epoch": epoch, "step": global_step,
                            **_quick_eval(model, cfg, device)}
                    with open(metrics_path, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(qrec) + "\n")
                    tqdm.write(
                        f"[quick_eval @step {global_step}] "
                        f"P_d={qrec['recall_pd90']:.3f} F_a_frm@pd90={qrec['fa_frm_pd90']:.4f}"
                        f"（可达={qrec['pd90_available']}）"
                        f" F1@0.5={qrec['primary']['f1']:.3f}"
                        f" 背景残差RMS={qrec['bg_resid_rms']:.4f}"
                        f" 残差SCR={qrec['resid_scr']:.2f} α均值={qrec['alpha_mean']:.3f}"
                        f" τ_B={qrec.get('tau_b_median', '-')} τ_T={qrec.get('tau_t_median', '-')}")
                    if global_step >= es_min_steps and early_stop_step(
                            es_state, qrec["fa_frm_pd90"] if qrec["pd90_available"] else float("inf"),
                            es_cfg):
                        tqdm.write(f"[early_stop] 连续 {es_state['patience']} 个快评周期 F_a@P_d "
                                   f"无改善（best={es_state['best']:.4f}），第 {epoch} 轮终止")
                        save_ckpt(f"{cfg.experiment.stage}_{cfg.train.seed}_earlystop.pt",
                                  epoch)  # 早停轮必须落盘（评审 次要#3）
                        stopped = True
            if (not recon_warned and loss_parts
                    and loss_parts.get("recon_valid_frac", 1.0) < 0.01):
                tqdm.write(f"[WARN] L_recon 有效像素占比 {loss_parts['recon_valid_frac']:.4f}"
                           " < 1%——反馈掩码疑似铺满全图（通道窒息），检查 norm_m_frac / alpha_mean")
                recon_warned = True
            total_loss += loss.item()
            n_steps += 1
            mem = torch.cuda.memory_allocated() / 2**30 if device == "cuda" else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}",
                mem=f"{mem:.2f}G", gn=f"{float(gn):.2f}",
            )
            if stopped:
                break
        if stopped:
            break
        avg = total_loss / max(1, n_steps)
        rec = {
            "time": time.strftime("%H:%M:%S"), "epoch": epoch,
            "loss_avg": round(avg, 6), "lr": optim.param_groups[0]["lr"],
            "steps": n_steps, "global_step": global_step,
        }
        if cfg.model.type == "dsld_core":  # τ 监控（M3 Gate 证据链）+ 范数 + 分量
            rec.update(model.tau_report())
            rec.update({f"norm_{k}": round(v, 4) for k, v in model.core.last_norms.items()})
            if loss_parts:
                rec.update({f"loss_{k}": round(v, 6) for k, v in loss_parts.items()})
        history.append(rec)
        with open(metrics_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec) + "\n")
        outer.set_postfix(loss=f"{avg:.4f}")
        tqdm.write(
            f"[epoch {epoch}] loss_avg={avg:.4f} lr={optim.param_groups[0]['lr']:.2e} "
            f"steps={n_steps}"
        )
        if epoch % cfg.train.save_every == 0 or epoch == cfg.train.epochs:
            save_ckpt(f"{cfg.experiment.stage}_{cfg.train.seed}_{epoch}.pt", epoch)
    print(f"\n训练结束：第 {start_epoch + 1}–{cfg.train.epochs} 轮完成，metrics → {metrics_path}")
    return {"history": history, "exp_dir": str(exp_dir)}
