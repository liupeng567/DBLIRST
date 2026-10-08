"""config → 对象（M3 起）：模型、窗口数据集、损失权重的唯一装配处。

为什么单独一个文件：L5 要求"config 全键有消费者"，而消费者必须是**可见的一次读取**
（`cfg.model.liquid.m_max` 这类属性链），不能散落在训练循环的 if 分支里。tests/
test_config_consumers.py 用 AST 扫描本文件的属性访问链，与 configs/dsld_core.yaml 的
叶子键一一对账——加一个没人读的键，测试当场变红。

训练循环（P2）在此之上加：优化器分组（θ_λ/门控偏置/GN 不衰减，§5.3）、梯度累积、
哨兵 strict 拦截、EMA 权重、课程（teacher:gate 混合，§4.4）。
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]


def _tau(node) -> tuple[float, float, float]:
    """{min,max,init} → (min,max,init)；顺序由构造保证，配置写错次序即 assert 失败。"""
    out = (float(node["min"]), float(node["max"]), float(node["init"]))
    assert out[0] < out[1], f"tau 区间非法: {node}"
    assert out[0] < out[2] < out[1], f"tau_init 必须严格在区间内: {node}"
    return out


def build_model(cfg):
    """cfg.model → DsldCore（§8.4 骨架的 model 块逐键消费）。"""
    from dsld.models.dsld_core import DsldCore

    m = cfg.model
    assert m.type == "dsld_core", f"build_model 只接 dsld_core，收到 {m.type}"
    liq, enc = m.liquid, m.encoder
    return DsldCore(
        in_ch=int(m.in_ch), enc_mid=int(enc.mid_ch), enc_out=int(enc.out_ch),
        c_h=int(liq.c_h), state_mode=str(liq.state_mode), tau_mode=str(liq.tau_mode),
        mask_source=str(liq.mask_source), bg_mode=str(liq.bg_mode),
        tau_b=_tau(liq.tau_b), tau_t=_tau(liq.tau_t),
        mask_decay=float(liq.mask_decay), mask_radius=int(liq.mask_radius),
        m_max=float(liq.m_max), alpha_th=float(liq.alpha_th),
        ema_momentum=float(liq.ema_momentum), s_max=float(liq.s_max),
        beta_max=float(liq.beta_max), state_dependent=bool(liq.state_dependent),
        gate_hidden=int(m.gating.hidden), gate_bias_init=float(m.gating.bias_init),
        use_checkpoint=bool(liq.use_checkpoint))


def build_windows(cfg, split: str, augment: bool | None = None, limit_seqs: int = 0):
    """cfg.data + cfg.train.window → IttdWindows（§4.7 训练档尺度、§5.2 采样口径）。

    augment 缺省按 split 推断（train-* 开、val-* 关）：评测窗必须与目检/快评同数据分布，
    增强只在训练侧出现，这条不能靠调用方记得传。
    """
    from dsld.data.ittd_window_dataset import IttdWindows

    d, w = cfg.data, cfg.train.window
    aug = (split.startswith("train") if augment is None else bool(augment))
    return IttdWindows(
        manifest_path=str(REPO / "data" / "manifests" / d.manifest),
        cache_root=str(REPO / d.cache_root), split=split,
        T=int(w.T), stride=int(w.stride), crop=tuple(int(v) for v in w.crop),
        augment=aug, align=bool(w.align), teacher_dilate=int(w.teacher_dilate),
        feature_stride=int(w.feature_stride), crop_margin=int(w.crop_margin),
        max_reg_fail_frac=float(w.max_reg_fail_frac), seed=int(cfg.train.seed),
        limit_seqs=limit_seqs)


def build_loader(cfg, split: str, batch_size: int, augment: bool | None = None,
                 limit_seqs: int = 0) -> DataLoader:
    """窗口数据集 → DataLoader（num_workers=0：Windows 上多进程 + memmap 实测更慢）。"""
    return DataLoader(build_windows(cfg, split, augment, limit_seqs),
                      batch_size=int(batch_size), shuffle=split.startswith("train"),
                      num_workers=0, drop_last=split.startswith("train"))


def build_loss_args(cfg) -> dict:
    """cfg.train.loss / cfg.train.sentinels → dsld.train.losses.total_loss 的 kwargs。

    权重四分量必须齐备（losses.total_loss 缺项即抛错），recon_valid_min 是 §5.5 的
    背景监督面塌缩哨兵阈值，与权重分开传——它是拦截线不是缩放因子。
    """
    return {"weights": cfg.train.loss,
            "recon_valid_min": float(cfg.train.sentinels.recon_valid_min)}
