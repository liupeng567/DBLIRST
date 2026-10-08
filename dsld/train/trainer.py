"""DSLD 训练循环（方案 9.1 main.py 规范，main.py 与 scripts/train.py 共用）。

M0 范围：训练链路打通——启动四步（配置树 / 环境 / 参数量 / 权重加载报告）+
按轮训练（tqdm 双进度条）+ 轮摘要 + metrics.jsonl + 按轮断点。
真实模型/数据集/损失在 M1–M3 逐阶段替换，接口保持不变。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]


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


def load_checkpoint(model: nn.Module, ckpt: str | None) -> None:
    print("=" * 72)
    print("④ 权重加载情况")
    print("=" * 72)
    if not ckpt:
        print("  No checkpoint → training from scratch（Xavier 初始化，τ 取范围中点）")
        return
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
    for extra in ("epoch", "best_metric", "ema", "optimizer"):
        if isinstance(state, dict) and extra in state:
            print(f"  附带状态: {extra}={state[extra] if extra != 'ema' else 'EMA权重'}")


def build_model(cfg) -> nn.Module:
    """dryrun = M0 占位网络；其余走 M3 装配（键→kwargs 的唯一读取处在 dsld/train/build.py）。"""
    from dsld.models.dryrun_net import DryRunNet

    if cfg.model.type == "dryrun":
        return DryRunNet(in_ch=cfg.model.in_ch, width=cfg.model.width)
    from dsld.train.build import build_model as _build_dsld

    return _build_dsld(cfg)


def build_dataloader(cfg) -> DataLoader:
    """M0: 合成窗口数据集；M1/M2 替换为 ITTD 真实窗口数据集（接口不变）。"""
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


def run_training(cfg) -> dict:
    """完整训练入口：返回轮历史摘要。"""
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nDSLD 训练启动 @ {t_start}")
    print_config_tree(cfg)
    print_environment(cfg)

    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    report_model_params(model)
    load_checkpoint(model, cfg.train.get("ckpt", None))
    if cfg.model.type != "dryrun":
        raise NotImplementedError(
            f"model.type={cfg.model.type}：装配与损失已就绪（冒烟 tests/test_dsld_core.py、"
            "计时 scripts/bench_core.py），但本循环仍是 M0 dryrun 的逐帧 L1 口径——"
            "四分量损失接线、梯度累积、§5.5 哨兵 strict 拦截、EMA 与 teacher→gate 课程"
            "都在 P2 落地，不用 dryrun 循环跑真实训练")

    loader = build_dataloader(cfg)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.optim.lr, weight_decay=cfg.train.optim.wd
    )
    steps_per_epoch = len(loader)
    exp_dir = REPO / "experiments" / cfg.experiment.name
    ckpt_dir = exp_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.jsonl"

    print("=" * 72)
    print(f"⑤ 按轮训练：{cfg.train.epochs} 轮 × {steps_per_epoch} step"
          f"（1 轮 = 遍历一次采样计划；M0 dry-run 为合成窗口）")
    print("=" * 72)

    history = []
    outer = tqdm(range(1, cfg.train.epochs + 1), desc="epoch", position=0)
    for epoch in outer:
        model.train()
        pbar = tqdm(loader, desc=f"train {epoch}", position=1, leave=False)
        total_loss = 0.0
        for step, batch in enumerate(pbar, 1):
            x = batch["windows"].to(device)
            target = batch["target"].to(device)
            out = model(x)
            loss = nn.functional.l1_loss(out, target)
            optim.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
            total_loss += loss.item()
            mem = torch.cuda.memory_allocated() / 2**30 if device == "cuda" else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}",
                mem=f"{mem:.2f}G", gn=f"{float(gn):.2f}",
            )
        avg = total_loss / max(1, steps_per_epoch)
        rec = {
            "time": time.strftime("%H:%M:%S"), "epoch": epoch,
            "loss_avg": round(avg, 6), "lr": optim.param_groups[0]["lr"],
            "steps": steps_per_epoch,
        }
        history.append(rec)
        with open(metrics_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec) + "\n")
        outer.set_postfix(loss=f"{avg:.4f}")
        tqdm.write(
            f"[epoch {epoch}] loss_avg={avg:.4f} lr={optim.param_groups[0]['lr']:.2e} "
            f"steps={steps_per_epoch}"
        )
        if epoch % cfg.train.save_every == 0 or epoch == cfg.train.epochs:
            name = f"{cfg.experiment.stage}_{cfg.train.seed}_{epoch}.pt"
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "cfg": OmegaConf.to_container(cfg)},
                ckpt_dir / name,
            )
            tqdm.write(f"[ckpt] 已保存 {ckpt_dir / name}")
    print(f"\n训练结束：{cfg.train.epochs} 轮完成，metrics → {metrics_path}")
    return {"history": history, "exp_dir": str(exp_dir)}
