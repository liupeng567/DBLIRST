"""DSLD 双状态液态核心单元测试（M3 阶段 A 返工版）。

对应 M3_核心层推进方案 阶段 A 出口判据：
  ① 实现的 a 恒在 [exp(−λ_hi), exp(−λ_lo)]        → test_realized_retention_within_structural_interval
  ② 实测衰减 == exp(−Δt·λ)                        → test_decay_matches_exp_negative_lambda
  ③ τ_eff_B/τ_eff_T ≥ 3（1000 组随机权重恒成立）  → test_scale_ratio_structural_over_random_weights
  ④ 状态有界（200 步对抗输入）                    → test_state_bounded_under_adversarial_inputs
  ⑤ 脉冲响应半衰期比 ≥ 3                          → test_pulse_response_and_half_life_ratio
  ⑥ 任意 α 下 max(M) ≤ m_max                      → test_feedback_mask_bounded_never_starves
  ⑦ α ≡ 0.119（起步）⇒ M ≡ 0                      → test_mask_zero_below_threshold
  ⑧ h_T ≡ 0 ⇒ L_dec > 0                           → test_decouple_loss_bounds
全部为可失败断言：旧实现（CfC Default 门控 / 无上界掩码 / cos² 无范数下限）在
②③④⑤⑧ 上必然不通过。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn as _nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.models.dsld_core import DsldCore, FpnLite  # noqa: E402
from dsld.models.encoder_ghostnetv2 import GhostNetV2Backbone  # noqa: E402
from dsld.models.liquid_core import (  # noqa: E402
    DualStateLiquidCore,
    _LiquidChannel,
    scene_stats,
    update_feedback_mask,
)
from dsld.train.losses import decouple_loss, focal_dice_loss, recon_loss  # noqa: E402

torch.manual_seed(0)


# ---- 骨干 / 颈 ----------------------------------------------------------

def test_backbone_taps_channels_and_strides():
    m = GhostNetV2Backbone(in_ch=1, width=1.0)
    s1, s2, s3 = m(torch.rand(1, 1, 480, 640))
    assert s1.shape == (1, 16, 240, 320)   # /2（表 M）
    assert s2.shape == (1, 24, 120, 160)   # /4
    assert s3.shape == (1, 40, 60, 80)     # /8
    assert m.out_channels == (16, 24, 40)


def test_fpn_lite_shapes():
    neck = FpnLite(16, 24, 40, 32, with_region=False)
    main, region = neck(torch.rand(1, 16, 240, 320), torch.rand(1, 24, 120, 160),
                        torch.rand(1, 40, 60, 80))
    assert main.shape == (1, 32, 240, 320)
    assert region is None


# ---- τ/λ 参数化（返工 A1：λ 域、σ 重参数化、不相交区间）------------------

def test_lambda_base_bounded_and_init_exact():
    """θ_λ 任意推远，λ_base 恒在通道区间内（σ 重参数化，构造性）；init 精确落在
    τ_init=48/6（f_head 零初始化 ⇒ 起步 mod≡1）。"""
    ch = DualStateLiquidCore(c_in=8, c_h=16).ch_bg
    tg = DualStateLiquidCore(c_in=8, c_h=16).ch_tg
    # init 精确性先验（fresh 通道）
    assert abs(float(ch.tau().median()) - 48.0) < 1e-4
    assert abs(float(tg.tau().median()) - 6.0) < 1e-4
    with torch.no_grad():
        ch.theta_lambda.fill_(-100.0)  # σ→0 → λ_base→λ_lo = 1/192
        assert float(ch.lam_base().max()) <= 1.0 / 192.0 + 1e-7
        ch.theta_lambda.fill_(100.0)   # σ→1 → λ_base→λ_hi = 1/24
        assert float(ch.lam_base().min()) >= 1.0 / 24.0 - 1e-7
    tau_tg = tg.tau()
    assert 2.0 <= tau_tg.min() and tau_tg.max() <= 8.0   # 返工：不相交区间 [2,8]


def test_decay_matches_exp_negative_lambda():
    """出口判据②：实测单步衰减 == exp(−Δt·λ)（τ 在指数里，非自由门控分母）。

    g/m 头置零 ⇒ cand≡0 ⇒ h' = a·h 精确成立；f/FiLM 零初始化 ⇒ λ=λ_base。
    旧实现（h' = p·g+(1−p)·m，h 仅作拼接输入）在此断言上必然失败。"""
    ch = DualStateLiquidCore(c_in=4, c_h=8).ch_bg
    with torch.no_grad():
        for head in (ch.g_head, ch.m_head):
            _nn.init.zeros_(head.weight)
            _nn.init.zeros_(head.bias)
    h0 = torch.randn(1, 8, 6, 6)
    scene = torch.zeros(1, 5)
    h1 = ch.step(h0, torch.zeros(1, 4, 6, 6), scene, None, dt=1.0)
    a_expected = torch.exp(-ch.lam_base()[None, :, None, None])
    assert torch.allclose(h1, a_expected * h0, atol=1e-5)


def test_realized_retention_within_structural_interval():
    """出口判据①：任意权重/输入/保持项下 λ ∈ [λ_lo, λ_hi]（终 clamp），
    即 a ∈ [exp(−λ_hi), exp(−λ_lo)]——不变量 0 加在实现出来的保留系数上。"""
    rng = torch.Generator().manual_seed(7)
    for draw in range(50):
        for (lo, hi, init, keep) in ((24.0, 192.0, 48.0, False),
                                     (2.0, 8.0, 6.0, True)):
            ch = _LiquidChannel(4, 8, lo, hi, init, keep=keep)
            with torch.no_grad():  # 对抗性大权重 + θ_λ 远超区间
                for head in (ch.f_head, ch.g_head, ch.m_head):
                    head.weight.normal_(0.0, 5.0, generator=rng)
                    head.bias.normal_(0.0, 5.0, generator=rng)
                ch.theta_lambda.uniform_(-8, 8, generator=rng)
            x = torch.randn(2, 4, 8, 8, generator=rng) * 10
            h = torch.randn(2, 8, 8, 8, generator=rng)
            scene = torch.randn(2, 5, generator=rng) * 3
            ap = torch.rand(2, 1, 8, 8, generator=rng) if keep else None
            ch.step(h, x, scene, ap)
            assert ch.last_lam_min >= 1.0 / hi - 1e-6, f"draw{draw} λ 越下界"
            assert ch.last_lam_max <= 1.0 / lo + 1e-6, f"draw{draw} λ 越上界"


def test_scale_ratio_structural_over_random_weights():
    """出口判据③：τ_eff_B/τ_eff_T ≥ 3 恒成立（1000 组随机权重/输入，构造性）。

    判据口径：通道内最快 τ_B（=1/λ_max_B）对通道内最慢 τ_T（=1/λ_min_T）之比——
    终 clamp 下 = τ_B_min 区间下界 24 / τ_T 区间上界 8 = 3。"""
    worst = float("inf")
    rng = torch.Generator().manual_seed(11)
    for _ in range(1000):
        chb = _LiquidChannel(4, 4, 24.0, 192.0, 48.0)
        cht = _LiquidChannel(4, 4, 2.0, 8.0, 6.0, keep=True)
        with torch.no_grad():
            for ch in (chb, cht):
                for head in (ch.f_head, ch.g_head, ch.m_head):
                    head.weight.normal_(0.0, 3.0, generator=rng)
                    head.bias.normal_(0.0, 3.0, generator=rng)
                ch.theta_lambda.uniform_(-8, 8, generator=rng)
        x = torch.randn(1, 4, 4, 4, generator=rng) * 5
        scene = torch.randn(1, 5, generator=rng) * 2
        ap = torch.rand(1, 1, 4, 4, generator=rng)
        chb.step(torch.zeros(1, 4, 4, 4), x, scene, None)
        cht.step(torch.zeros(1, 4, 4, 4), x, scene, ap)
        tau_b_fastest = 1.0 / chb.last_lam_max
        tau_t_slowest = 1.0 / cht.last_lam_min
        worst = min(worst, tau_b_fastest / tau_t_slowest)
    assert worst >= 3.0 - 1e-4


def test_state_bounded_under_adversarial_inputs():
    """出口判据④：任意输入喂 200 步，逐元素 |h| ≤ max(|h₀|,1)+ε
    （tanh 候选 + a∈(0,1) 压缩映射）。旧实现（两个无界学习量的凸组合）必然失败。"""
    ch = _LiquidChannel(4, 8, 2.0, 8.0, 6.0)  # 快通道最坏情形
    with torch.no_grad():
        for head in (ch.f_head, ch.g_head, ch.m_head):
            head.weight.normal_(0.0, 5.0)
            head.bias.normal_(0.0, 5.0)
    h = torch.zeros(1, 8, 8, 8)
    scene = torch.zeros(1, 5)
    g = torch.Generator().manual_seed(3)
    for _ in range(200):
        x = torch.randn(1, 4, 8, 8, generator=g) * 20
        h = ch.step(h, x, scene, None)
    assert float(h.abs().max()) <= 1.0 + 1e-5


def test_pulse_response_and_half_life_ratio():
    """出口判据⑤：常数背景 + 单帧亮点。g/m 头置常偏置（cand 确定性、与输入无关），
    f 零初始化（mod≡1）⇒ 一切解析可算：
      响应比 |Δh_T|/|Δh_B| = (1−a_T)/(1−a_B) ≈ 7.5 ≥ 3（快通道对脉冲更敏感）；
      脉冲后半衰期 t* 满足 a^{t*}=0.5 ⇒ t* = τ·ln2（B 33.3 / T 4.2 帧），比值 ≥ 3。"""
    def make(tau_min, tau_max, tau_init):
        ch = _LiquidChannel(2, 4, tau_min, tau_max, tau_init)
        with torch.no_grad():
            _nn.init.zeros_(ch.g_head.weight)
            ch.g_head.bias.fill_(1.0)
            _nn.init.zeros_(ch.m_head.weight)
            ch.m_head.bias.fill_(0.5)
        return ch

    scene = torch.zeros(1, 5)
    x = torch.zeros(1, 2, 4, 4)
    chb, cht = make(24.0, 192.0, 48.0), make(2.0, 8.0, 6.0)
    hb = chb.step(torch.zeros(1, 4, 4, 4), x, scene, None)
    ht = cht.step(torch.zeros(1, 4, 4, 4), x, scene, None)
    resp_ratio = float(ht.abs().mean() / hb.abs().mean())
    assert resp_ratio > 3.0

    cand0 = math.tanh(1.0 + 0.5 * 0.5)  # 平衡点（cand 常数 ⇒ h → cand0）

    def half_life(ch, h1, steps=300):
        gap0 = float((h1 - cand0).abs().mean())
        for t in range(1, steps + 1):
            h1 = ch.step(h1, x, scene, None)
            if float((h1 - cand0).abs().mean()) <= 0.5 * gap0:
                return t
        return steps

    t_b, t_t = half_life(chb, hb), half_life(cht, ht)
    assert 0.6 * 48.0 * math.log(2) <= t_b <= 1.4 * 48.0 * math.log(2)
    assert 0.6 * 6.0 * math.log(2) <= t_t <= 1.4 * 6.0 * math.log(2)
    assert t_b / max(t_t, 1) >= 3.0


def test_keep_term_monotonic_and_capped():
    """4.4/返工：κ init≈0.2（仅目标通道）；α_prev ↑ ⇒ τ_eff 单调不减；
    终 clamp 下 τ_eff ≤ τ_max=8（保持项延长上限 = 区间上界，可行方案 §4.2 原设计）。"""
    tg = DualStateLiquidCore(c_in=8, c_h=8).ch_tg
    assert abs(float(tg.keep_gain()) - 0.2) < 0.02
    assert not DualStateLiquidCore(c_in=8, c_h=8).ch_bg.keep
    taus = []
    for a in (0.0, 0.3, 0.6, 0.9, 1.0):
        ap = torch.full((1, 1, 4, 4), a)
        tg.step(torch.zeros(1, 8, 4, 4), torch.zeros(1, 8, 4, 4),
                torch.zeros(1, 5), ap)
        taus.append(tg.last_tau_eff)
    assert all(b >= a_ - 1e-6 for a_, b in zip(taus, taus[1:]))
    assert taus[-1] <= 8.0 + 1e-4


# ---- 反馈掩码（返工 A3）--------------------------------------------------

def test_feedback_mask_bounded_never_starves():
    """出口判据⑥：任意 α 下 max(M) ≤ m_max=0.8、min(1−M) ≥ 0.2（窒息结构性不可达）；
    膨胀足迹 / hold-or-decay 语义保持（峰值钳到 0.8，衰减链 0.8·0.9^t）。"""
    alpha = torch.zeros(1, 1, 16, 16)
    alpha[0, 0, 8, 8] = 0.9  # 孤立高峰（>α_th），背景 0.3
    alpha[0, 0, 3, 3] = 0.3
    M0 = update_feedback_mask(alpha, torch.zeros(1, 1, 16, 16), radius=2,
                              decay=0.9, alpha_th=0.6, m_max=0.8)
    assert abs(float(M0[0, 0, 8, 8]) - 0.8) < 1e-7           # 新峰 = m_max（非 1.0）
    assert abs(float(M0[0, 0, 6, 8]) - 0.8) < 1e-7           # 半径 2 覆盖
    assert abs(float(M0[0, 0, 8, 10]) - 0.8) < 1e-7
    assert float(M0[0, 0, 5, 8]) == 0.0 and float(M0[0, 0, 3, 3]) == 0.0  # 低峰/远点不掩
    M = M0
    for t, expect in zip(range(1, 4), (0.72, 0.648, 0.5832)):  # 0.8·0.9^t
        M = update_feedback_mask(torch.zeros(1, 1, 16, 16), M, radius=2,
                                 decay=0.9, alpha_th=0.6, m_max=0.8)
        assert abs(float(M[0, 0, 8, 8]) - expect) < 1e-6
    # 任意 α + 满掩码历史：仍不越上界、通道输入永 ≥ 20%
    alpha_r = torch.rand(2, 1, 32, 32)
    M = update_feedback_mask(alpha_r, torch.ones(2, 1, 32, 32), radius=5,
                             decay=0.9, alpha_th=0.6, m_max=0.8)
    assert float(M.max()) <= 0.8 + 1e-7
    assert float((1.0 - M).min()) >= 0.2 - 1e-7


def test_mask_zero_below_threshold():
    """出口判据⑦：gate 偏置 −2 起步 α≡σ(−2)≈0.119 < α_th=0.6 ⇒ M≡0（无证据不掩码）；
    整机口径：偏置抬高越阈后掩码 = m_max 而非 1（背景输入 ≥20% 保持）。"""
    core = DualStateLiquidCore(c_in=4, c_h=8, mask_radius=2, mask_decay=0.9,
                               alpha_th=0.6, mask_m_max=0.8)
    core.eval()
    out = core(torch.rand(1, 3, 4, 16, 16))
    assert torch.allclose(out["alpha"], torch.full_like(out["alpha"], 0.1192029),
                          atol=1e-4)
    assert float(out["m_tgt"].abs().max()) == 0.0
    assert float(core.last_norms["m_frac"]) == 0.0
    assert abs(float(core.last_norms["alpha_mean"]) - 0.1192029) < 1e-4
    with torch.no_grad():
        core.gate_head.bias.fill_(2.0)  # 模拟确认响应：α≈0.881 > 0.6
    out = core(torch.rand(1, 3, 4, 16, 16))
    assert abs(float(out["m_tgt"].min()) - 0.8) < 1e-7  # 全图掩码也只到 m_max


def test_state_dependent_heads_variant():
    """消融开关：state_dependent=true 时头吃 [h,x]（状态相关 τ 语义，递推非线性）；
    默认 false = x-only（阶段 E 并行扫描前提，可行方案 §4.1）。"""
    core = DualStateLiquidCore(c_in=4, c_h=8, state_dependent=True)
    out = core(torch.rand(1, 3, 4, 16, 16))
    assert out["logits"].shape == (1, 3, 1, 16, 16)
    assert core.ch_bg.f_head.in_channels == 12  # c_in + c_h
    core_x = DualStateLiquidCore(c_in=4, c_h=8)
    assert core_x.ch_bg.f_head.in_channels == 4  # 仅 x


def test_realized_tau_equals_base_at_init():
    """返工 A1/A6：起步实现值与基准值的关系（f/FiLM 零初始化 ⇒ mod≡1）：
    背景通道（无保持项）τ_eff == τ_base 精确；目标通道起步即被保持项延长
    （α_prev=σ(−2)≈0.119 → τ_eff≈6.15）且不越上界 8。tau_report 输出动力学空间量。"""
    core = DualStateLiquidCore(c_in=4, c_h=8)
    core.eval()
    core(torch.rand(1, 3, 4, 16, 16))
    r = core.tau_report()
    assert abs(r["tau_eff_b"] - r["tau_b_median"]) < 0.05   # bg：无保持项，精确
    assert r["tau_t_median"] < r["tau_eff_t"] <= 8.0        # tg：保持项延长且不越界
    assert r["tau_scale_ratio"] > 3.0
    assert r["lam_at_bound_b"] < 0.5 and r["lam_at_bound_t"] < 0.5  # 起步不顶边


# ---- 整机与前向语义 ------------------------------------------------------

def test_dual_vs_single_modes_forward():
    for mode in ("dual", "single"):
        core = DualStateLiquidCore(c_in=8, c_h=16, mode=mode)
        out = core(torch.rand(2, 4, 8, 24, 32))
        assert out["logits"].shape == (2, 4, 1, 24, 32)
        assert out["y_b"].shape == (2, 4, 8, 24, 32)
        assert out["alpha"].shape == (2, 4, 1, 24, 32)
    single = DualStateLiquidCore(c_in=8, c_h=16, mode="single")
    assert single.ch_bg is None
    ts = single.ch_tg.tau()
    assert 2.0 <= ts.min() and ts.max() <= 192.0  # 单状态 τ 区间 = 并集 [2,192]
    rep = single.tau_report()
    assert "tau_single_median" in rep and "tau_eff_single" in rep


def test_truncated_bptt_detaches_state():
    """4.9-② 截断 BPTT：detach_every 切断跨段梯度，参数仍有梯度。"""
    core = DualStateLiquidCore(c_in=4, c_h=8, detach_every=2)
    out = core(torch.rand(1, 6, 4, 16, 16))
    out["logits"].sum().backward()
    assert core.ch_tg.f_head.weight.grad is not None
    assert torch.isfinite(core.ch_tg.f_head.weight.grad).all()


def test_state_and_gating_are_fp32_under_autocast():
    """4.8-①：autocast(bf16) 下核心输出仍为 fp32。"""
    core = DualStateLiquidCore(c_in=4, c_h=8)
    if not torch.cuda.is_available():
        x = torch.rand(1, 2, 4, 16, 16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = core(x)
        assert out["h_t"].dtype == torch.float32
    else:  # CUDA 路径在冒烟训练覆盖
        pass


def test_scene_stats_dims():
    x = torch.rand(2, 8, 16, 16)
    q = torch.tensor([1.0, 2.0])
    s = scene_stats(x, q, t=3, T=8)
    assert s.shape == (2, 5)
    assert torch.allclose(s[:, -1], q)  # 清晰度直通


# ---- 整机与损失 ----------------------------------------------------------

def test_dsld_core_forward_backward_smoke():
    m = DsldCore(width=0.8, c_main=32, c_h=16)
    x = torch.rand(1, 4, 1, 96, 128)
    q = torch.tensor([1.0])
    out = m(x, quality=q)
    assert out["logits"].shape == (1, 4, 1, 96, 128)
    assert out["y_b"].shape[2] == 32
    target = (torch.rand(1, 4, 1, 96, 128) > 0.995).float()
    seg = focal_dice_loss(out["logits"], target)
    rec = recon_loss(out["y_b"], out["x_main"], out["m_tgt"], target)
    dec = decouple_loss(out["h_t"], out["h_b"])
    loss = seg + 0.5 * rec + 0.1 * dec
    loss.backward()
    assert torch.isfinite(loss)
    g = m.core.ch_bg.theta_lambda.grad
    assert g is not None and torch.isfinite(g).all()  # λ 收到梯度（消融 b 前提；σ 重参数化无死区）
    n = sum(p.numel() for p in m.parameters())
    assert n < 5e6  # 方案硬闸门：Params ≤ 5M


def test_recon_loss_valid_frac_sentinel():
    """评审 2.1：recon_loss 暴露有效像素占比（静默零可观测）。"""
    from dsld.train.losses import recon_loss as rl_fn

    B, T = 1, 2
    x = torch.zeros(B, T, 1, 32, 32)
    y_b = torch.zeros(B, T, 1, 32, 32)
    box = torch.zeros(B, T, 1, 64, 64)
    rl_fn(y_b, x, torch.zeros(B, T, 1, 32, 32), box)
    assert getattr(rl_fn, "last_valid_frac", 0.0) > 0.99  # 无排除 → 几乎全有效
    rl_fn(y_b, x, torch.zeros(B, T, 1, 32, 32), torch.ones(B, T, 1, 64, 64),
          strict=False)  # 全排除：strict=False 只告警（strict=True 见下一条）
    assert getattr(rl_fn, "last_valid_frac", 1.0) < 1e-3


def test_recon_loss_strict_intercept():
    """返工 B-2：有效像素 <1% 且 strict=True（默认）⇒ RuntimeError——
    与 trainer"loss NaN 即停"同策略，窒息不再静默烧 GPU。"""
    B, T = 1, 2
    x = torch.zeros(B, T, 1, 32, 32)
    y_b = torch.zeros(B, T, 1, 32, 32)
    box = torch.ones(B, T, 1, 64, 64)
    with pytest.raises(RuntimeError, match="窒息"):
        recon_loss(y_b, x, torch.zeros(B, T, 1, 32, 32), box)


def test_recon_loss_excludes_target_and_top_residual():
    B, T, C, H, W = 1, 2, 1, 32, 32
    x = torch.zeros(B, T, C, H, W)
    y_b = torch.zeros(B, T, C, H, W)
    x[..., 5, 5] = 1.0   # 未标注亮点（应被 top-5% 剔除）
    x[..., 20, 20] = 1.0  # 背景残差（应参与）
    box = torch.zeros(B, T, 1, 2 * H, 2 * W)
    box[..., 8:12, 8:12] = 1.0  # 原生分辨率 GT 框（stride-2 处 [4:6,4:6]）
    l_all = recon_loss(y_b, x, torch.zeros(B, T, 1, H, W), box, top_ratio=0.05)
    l_no_top = recon_loss(y_b, x, torch.zeros(B, T, 1, H, W), box, top_ratio=0.0)
    assert l_no_top > 2 * l_all  # top-5% 剔除确实把亮点残差挡在监督外
    box_full = torch.ones(B, T, 1, 2 * H, 2 * W)
    assert float(recon_loss(y_b, x, torch.zeros(B, T, 1, H, W), box_full,
                            strict=False)) == 0.0


def test_decouple_loss_bounds():
    """返工 B-3：健康状态行为保持（同向→1、正交→≈0）；
    坍缩（任一状态 ≡0）现在必然产生正损失（出口判据⑧，旧实现返回 0 放过坍缩）。"""
    a = torch.randn(2, 3, 8, 16, 16)
    b = torch.randn(2, 3, 8, 16, 16)
    z = torch.zeros_like(a)
    l = decouple_loss(a, b)
    assert 0.0 <= float(l) <= 1.0 + 0.1  # cos²∈[0,1] + 非负 norm floor
    assert abs(float(decouple_loss(a, a)) - 1.0) < 1e-5  # 同向 → cos²=1（最大惩罚）
    assert float(decouple_loss(a, z)) > 0.0    # 单侧坍缩 → norm floor 生效
    assert float(decouple_loss(z, z)) > 0.0    # 双零坍缩 → 不再被 eps 抹平


def test_focal_dice_empty_frames():
    """空标注帧（合法负样本）：Dice 项约定 0、Focal 正常，损失有限。"""
    lg = torch.randn(2, 1, 32, 32)
    tg = torch.zeros(2, 1, 32, 32)
    assert torch.isfinite(focal_dice_loss(lg, tg))


def test_param_count_budget():
    """方案 4.7 预算口径：1.0× 整机实测（骨干截断 + 颈 + 核心 + 头）≤ 5M；
    液态核心保持 0.01M 量级（x-only 头较旧实现还略小）。"""
    m = DsldCore(width=1.0, c_main=32, c_h=32)
    groups: dict[str, int] = {}
    for name, p in m.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    total = sum(groups.values())
    assert total < 5e6
    assert groups["core"] < 2e4  # 返工后核心 ≈ 0.010M（3 头 c_in→c_h ×2 通道 + FiLM）
