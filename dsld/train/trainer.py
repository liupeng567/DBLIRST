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
        return DsldCore(
            in_ch=cfg.model.get("in_ch", 1),
            width=cfg.model.encoder.get("width", 1.0),
            c_main=cfg.model.neck.get("c_main", 32),
            c_h=liq.get("h", 32),
            liquid_mode=liq.get("mode", "dual"),
            tau_b=tuple(liq.tau_b.get(k) for k in ("min", "max", "init")),
            tau_t=tuple(liq.tau_t.get(k) for k in ("min", "max", "init")),
            mask_radius=cfg.model.feedback.get("mask_radius", 5),
            mask_decay=cfg.model.feedback.get("mask_decay", 0.9),
            alpha_th=cfg.model.feedback.get("alpha_th", 0.5),
            detach_every=liq.get("detach_every", 0),
            use_checkpoint=liq.get("use_checkpoint", False),
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
    parts = {"seg": float(seg), "recon": float(rec)}
    loss = weights.get("seg", 1.0) * seg + weights.get("recon", 0.5) * rec
    if model.liquid_mode == "dual":  # 单状态无解耦对象（消融 a 对照）
        dec = decouple_loss(out["h_t"], out["h_b"])
        loss = loss + weights.get("decouple", 0.1) * dec
        parts["decouple"] = float(dec)
    parts["total"] = float(loss)
    return loss, parts


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
            total_loss += loss.item()
            n_steps += 1
            mem = torch.cuda.memory_allocated() / 2**30 if device == "cuda" else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}",
                mem=f"{mem:.2f}G", gn=f"{float(gn):.2f}",
            )
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
            name = f"{cfg.experiment.stage}_{cfg.train.seed}_{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "cfg": OmegaConf.to_container(cfg),
                    "history": history,
                },
                ckpt_dir / name,
            )
            tqdm.write(f"[ckpt] 已保存 {ckpt_dir / name}（含优化器状态，可断点续训）")
    print(f"\n训练结束：第 {start_epoch + 1}–{cfg.train.epochs} 轮完成，metrics → {metrics_path}")
    return {"history": history, "exp_dir": str(exp_dir)}
