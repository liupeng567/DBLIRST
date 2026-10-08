"""装配 + 端到端冒烟（M3 v2.0 §8.2 P1 Gate：params ≤0.2M、损失有限、四臂可跑）。

覆盖 §8.3 清单里"跨文件"的那几条：config → 模型 → 前向 → 四分量损失 → 反向的梯度可达性
（θ_λ、门控末层、seg/bg 头、dw 上下文都必须拿到梯度——任何一条断了就说明监督通路有洞）。
真实缓存用例走小档（T=8、crop[96,128]）控 CPU 时间；口径与训练档一致，只是尺度小。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dsld.models.dsld_core import DsldCore          # noqa: E402
from dsld.train.build import build_loss_args, build_model, build_windows  # noqa: E402
from dsld.train.losses import total_loss            # noqa: E402

CFG = OmegaConf.load(REPO / "configs" / "dsld_core.yaml")
CACHE = REPO / "data" / "cache" / "ittd"
HAS_CACHE = (CACHE / "seq_0021").is_dir()


def tiny_model(**kw) -> DsldCore:
    kw.setdefault("enc_mid", 4)
    kw.setdefault("enc_out", 8)
    kw.setdefault("c_h", 8)
    kw.setdefault("mask_source", "teacher")
    return DsldCore(**kw)


def batch(B=1, T=4, H=32, W=48, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    hw = (H // 2, W // 2)
    x = torch.rand(B, T, 1, H, W, generator=g)
    tgt = torch.zeros(B, T, 1, *hw)
    tgt[:, :, 0, 4:6, 6:9] = 1.0
    tm = torch.zeros(B, T, 1, *hw)
    tm[:, :, 0, 3:8, 5:11] = 1.0
    dt = torch.ones(B, T)
    dt[:, 2::3] = 0.0                    # ITTD 周期-3 补帧口径
    return {"windows": x, "target": tgt, "teacher_mask": tm, "dt": dt,
            "quality": torch.rand(B, generator=g)}


def run(b: dict, model: DsldCore) -> dict:
    return model(b["windows"], dt=b["dt"], quality=b["quality"],
                 teacher_mask=b["teacher_mask"])


# ---- P1 Gate：config 装配 + 参数预算 ------------------------------------------
def test_model_from_config_shapes_and_param_budget(capsys):
    m = build_model(CFG)
    n = sum(p.numel() for p in m.parameters())
    out = run(batch(B=1, T=3), m)
    assert tuple(out["logits"].shape) == (1, 3, 1, 16, 24)
    assert tuple(out["x_main"].shape) == (1, 3, 24, 16, 24)
    assert tuple(out["h_t"].shape) == (1, 3, 24, 16, 24)
    assert n <= 0.2e6, "§4.6 预算：核心链 ≤0.2M"
    with capsys.disabled():
        print(f"\n  [P1 实测] 参数量 = {n:,}（编码器 {sum(p.numel() for p in m.encoder.parameters()):,}"
              f" + 核心 {sum(p.numel() for p in m.core.parameters()):,}），预算 0.2M")


def test_teachermask_is_passed_through_without_re_dilation():
    """装配层不再二次膨胀：teacher_mask 进核心即数据侧那张图（同几何只有一份实现，L4）。"""
    m = tiny_model()
    b = batch(B=1, T=3, H=32, W=32)
    out = run(b, m)
    assert torch.equal(out["m_tgt"][:, 2], b["teacher_mask"][:, 2])  # 递推 max(源, 0.9·prev)=源


# ---- 反向：监督通路可达性 ------------------------------------------------------
def test_backward_grads_reach_every_module():
    m = tiny_model()
    b = batch(B=1, T=3, H=32, W=32)
    out = run(b, m)
    w = {"seg": 1.0, "recon": 0.5, "gate": 0.3, "dec": 0.1}
    res = total_loss(out, b, w)
    assert torch.isfinite(res["loss"]), res
    res["loss"].backward()
    want = {
        "encoder.stem.0.weight": None,
        "core.dw_ctx.weight": None,
        "core.ch_bg.theta_lambda": None,
        "core.ch_tg.theta_lambda": None,
        "core.ch_bg.f_head.weight": None,
        "core.ch_tg.g_head.weight": None,
        "core.bg_head.weight": None,
        "core.seg_head.weight": None,
        "core.gate.0.weight": None,
        "core.gate.2.bias": None,
        "core.t_readout.weight": None,
    }
    named = dict(m.named_parameters())
    for k in want:
        g = named[k].grad
        assert g is not None, f"{k} 无梯度（监督通路断）"
        assert torch.isfinite(g).all(), f"{k} 梯度非有限"
    for k in ("core.ch_bg.theta_lambda", "core.seg_head.weight", "core.bg_head.weight",
              "core.gate.2.bias", "core.ch_tg.theta_lambda", "core.ch_bg.f_head.weight",
              "core.dw_ctx.weight", "encoder.stem.0.weight"):
        assert float(named[k].grad.abs().sum()) > 0, f"{k} 梯度恒零"


def test_zero_init_gates_are_not_dead_branches():
    """零初始化头的"下一拍可达性"：末层权重为 0 ⇒ 首步只有它自己有梯度（含 t_readout 这类
    纯输入通道）；按该梯度走一步后，前一层与 t_readout 必须拿到非零梯度。否则"零初始化"
    就是永久死支（L5 假解）。
    """
    m = tiny_model()
    b = batch(B=1, T=3, H=32, W=32)
    res = total_loss(run(b, m), b, {"seg": 0.0, "recon": 0.0, "gate": 1.0, "dec": 0.0})
    res["loss"].backward()
    w2 = m.core.gate[2].weight
    assert float(w2.grad.abs().sum()) > 0.0                            # 末层自己可达
    assert float(m.core.gate[0].weight.grad.abs().sum()) == 0.0         # 穿过零权重 ⇒ 首步为 0
    assert float(m.core.t_readout.weight.grad.abs().sum()) == 0.0
    with torch.no_grad():
        w2 += 0.01 * w2.grad                                            # 一步更新
    m.zero_grad(set_to_none=True)
    total_loss(run(b, m), b, {"seg": 0.0, "recon": 0.0, "gate": 1.0, "dec": 0.0})["loss"].backward()
    assert float(m.core.gate[0].weight.grad.abs().sum()) > 0.0, "门控第一层永久死支"
    assert float(m.core.t_readout.weight.grad.abs().sum()) > 0.0, "门控 h_T 读出永久死支"


# ---- 四臂可跑 + 构造性差异（§4.5） --------------------------------------------
@pytest.mark.parametrize("kw", [
    {}, {"state_mode": "single"}, {"tau_mode": "swap"}, {"bg_mode": "ema"},
    {"mask_source": "gate"}, {"mask_source": "off"}, {"state_dependent": True},
    {"bg_mode": "ema", "mask_source": "gate"}, {"state_mode": "single", "bg_mode": "ema"},
])
def test_every_arm_runs_finite(kw):
    m = tiny_model(**{"mask_source": "teacher", **kw})
    b = batch(B=1, T=3, H=32, W=32)
    if kw.get("mask_source") in ("gate", "off"):
        b = dict(b, teacher_mask=None)
    out = m(b["windows"], dt=b["dt"], quality=b["quality"], teacher_mask=b["teacher_mask"])
    res = total_loss(out, b, {"seg": 1.0, "recon": 0.5, "gate": 0.3, "dec": 0.1})
    assert torch.isfinite(res["loss"])
    for k in ("logits", "alpha", "y_b", "r", "h_t", "h_b", "m_tgt"):
        assert torch.isfinite(out[k]).all(), k
    assert float(out["alpha"].min()) >= 0.0 and float(out["alpha"].max()) <= 1.0


def test_single_arm_dec_constant_and_protection_freezes_its_only_stream():
    """single 臂两件事一起锁：① h_b≡h_t ⇒ cos² 在健康位置恒 1（两臂损失组成一致）；
    ② 保护区 h ≡ 0 —— 单状态下"保护式更新"冻结的就是唯一那条流，GT 区的目标读出
    结构性为零。这不是 bug，而是"双状态必要性"的机制层含义：一条流无法既贴住背景
    又读出目标。判据 ⑥ 的 dual-vs-single 差里含这一项，验收报告归因时必须写明。
    """
    from dsld.train.losses import decouple_loss
    m = tiny_model(state_mode="single")
    b = batch(B=1, T=3, H=32, W=32)
    out = run(b, m)
    l = decouple_loss(out["h_t"], out["h_b"], min_norm=0.0)
    assert torch.isfinite(l) and float(l) > 0.8
    nt = out["h_t"].norm(dim=2)                          # [B,T,H/2,W/2]
    box = nt[..., 3:8, 5:11]                             # batch() 的 teacher 方块
    assert float(box.max()) == 0.0, "保护区应被冻结在零初始状态"
    assert float(nt[..., 0:2, 0:2].max()) > 0.0          # 区外照常读写


def test_gradient_checkpoint_matches_plain_forward():
    import copy as _copy
    m1 = tiny_model()
    m1.train()
    b = batch(B=1, T=4, H=32, W=32)
    plain = run(b, m1)["logits"]
    m2 = _copy.deepcopy(m1)
    m2.core.use_checkpoint = True
    ck = run(b, m2)["logits"]
    assert torch.allclose(plain, ck, atol=1e-5), float((plain - ck).abs().max())


# ---- 真实缓存端到端（ITTD 缓存存在时才跑；小档控时间） ------------------------
@pytest.mark.skipif(not HAS_CACHE, reason="ITTD 缓存不在本机")
def test_real_window_to_loss_is_finite():
    cfg = OmegaConf.create(OmegaConf.to_container(CFG, resolve=True))
    cfg.train.window.T = 8
    cfg.train.window.crop = [96, 128]
    cfg.train.window.stride = 8
    ds = build_windows(cfg, split="val-int", augment=False, limit_seqs=1)
    assert len(ds) > 0
    item = ds[0]
    x = torch.from_numpy(item["windows"])[None]
    tgt = torch.from_numpy(item["target"])[None]
    tm = torch.from_numpy(item["teacher_mask"])[None]
    dt = torch.from_numpy(item["dt"])[None]
    m = build_model(cfg)
    out = m(x, dt=dt, quality=torch.tensor([item["quality"]]), teacher_mask=tm)
    assert tuple(out["logits"].shape) == tuple(tgt.shape), (out["logits"].shape, tgt.shape)
    res = total_loss(out, {"target": tgt, "teacher_mask": tm}, **build_loss_args(cfg))
    assert torch.isfinite(res["loss"]), res["loss"]
    assert res["diag"]["recon_valid_frac"] >= float(cfg.train.sentinels.recon_valid_min)
    # 容器时间守恒（D-P1-1 的数据侧不变量）：Σ dt = T，补帧槽 0 与新曝光槽 1/2 必须配平
    assert float(dt.sum()) == float(cfg.train.window.T), dt.tolist()
    assert float(dt.min()) == 0.0, "真实 ITTD 段应含补帧槽（周期-3 实测，P0 报告 §3）"
