"""DSLD 双状态液态核心单元测试（M3，方案 4.2–4.8 逐条验证）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.models.dsld_core import DsldCore, FpnLite  # noqa: E402
from dsld.models.encoder_ghostnetv2 import GhostNetV2Backbone  # noqa: E402
from dsld.models.liquid_core import DualStateLiquidCore, scene_stats  # noqa: E402
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


# ---- τ 参数化（4.2）------------------------------------------------------

def test_tau_hard_clipped_to_range():
    ch = DualStateLiquidCore(c_in=8, c_h=16).ch_bg
    with torch.no_grad():
        ch.theta_tau.fill_(0.0)  # log(1) ≪ log(16) → 应被 clip 到下界
        assert torch.allclose(ch.tau(), torch.full((16,), 16.0))
        ch.theta_tau.fill_(100.0)  # ≫ log(256) → 上界
        assert torch.allclose(ch.tau(), torch.full((16,), 256.0))
    ch2 = DualStateLiquidCore(c_in=8, c_h=16).ch_tg
    assert 2.0 <= ch2.tau().min() and ch2.tau().max() <= 16.0
    assert abs(float(ch2.tau().median()) - 6.0) < 1e-4  # init 在范围中点附近


def test_keep_term_initial_beta_about_2():
    """4.4：β 初始化使 gate 偏移 ≈ −2（保持时长约 10 帧）。"""
    tg = DualStateLiquidCore(c_in=8, c_h=8).ch_tg
    beta = torch.nn.functional.softplus(tg.theta_beta)
    assert abs(float(beta) - 2.0) < 0.1
    bg = DualStateLiquidCore(c_in=8, c_h=8).ch_bg
    assert not bg.keep  # 背景通道无保持项（4.3）


# ---- 递推语义（4.3–4.5）--------------------------------------------------

def test_feedback_mask_decay_and_dilation():
    """4.5 更新纯函数：膨胀足迹 / hold-or-decay / 值域 {0}∪{0.9^k}∪1。"""
    from dsld.models.liquid_core import update_feedback_mask

    alpha = torch.zeros(1, 1, 16, 16)
    alpha[0, 0, 8, 8] = 0.9  # 孤立高峰（>α_th），背景 0.3
    alpha[0, 0, 3, 3] = 0.3
    M0 = update_feedback_mask(alpha, torch.zeros(1, 1, 16, 16), radius=2,
                              decay=0.9, alpha_th=0.5)
    assert float(M0[0, 0, 8, 8]) == 1.0
    assert float(M0[0, 0, 6, 8]) == 1.0 and float(M0[0, 0, 8, 10]) == 1.0  # 半径 2 覆盖
    assert float(M0[0, 0, 5, 8]) == 0.0 and float(M0[0, 0, 3, 3]) == 0.0   # 低峰/远点不掩
    # 峰消失后按 0.9 衰减：M(t)=0.9^t，hold-or-decay 精确成立
    M = M0
    for t, expect in zip(range(1, 4), (0.9, 0.81, 0.729)):
        M = update_feedback_mask(torch.zeros(1, 1, 16, 16), M, radius=2,
                                 decay=0.9, alpha_th=0.5)
        assert abs(float(M[0, 0, 8, 8]) - expect) < 1e-6
    # 新峰刷新为 1（max 语义）
    alpha2 = torch.zeros(1, 1, 16, 16)
    alpha2[0, 0, 8, 8] = 0.99
    M = update_feedback_mask(alpha2, M, radius=2, decay=0.9, alpha_th=0.5)
    assert float(M[0, 0, 8, 8]) == 1.0


def test_feedback_mask_alpha_threshold():
    """α_th 语义 + 零初始化中性门：α≡0.5 不越过 α_th=0.5（背景不被打码）。"""
    core = DualStateLiquidCore(c_in=4, c_h=8, mask_radius=2, mask_decay=0.9,
                               alpha_th=0.5)
    core.eval()
    out = core(torch.rand(1, 3, 4, 16, 16))
    # 零初始化 gate_head ⇒ α≡0.5，(α>0.5)=空 ⇒ 掩码恒 0
    assert torch.allclose(out["alpha"], torch.full_like(out["alpha"], 0.5), atol=1e-6)
    assert float(out["m_tgt"].abs().max()) == 0.0
    # 偏置抬高（模拟确认响应）：α≈0.88 > 阈 → 掩码全 1
    with torch.no_grad():
        core.gate_head.bias.fill_(2.0)
    out = core(torch.rand(1, 3, 4, 16, 16))
    assert float(out["m_tgt"].min()) == 1.0


def test_dual_vs_single_modes_forward():
    for mode in ("dual", "single"):
        core = DualStateLiquidCore(c_in=8, c_h=16, mode=mode)
        out = core(torch.rand(2, 4, 8, 24, 32))
        assert out["logits"].shape == (2, 4, 1, 24, 32)
        assert out["y_b"].shape == (2, 4, 8, 24, 32)
        assert out["alpha"].shape == (2, 4, 1, 24, 32)
    single = DualStateLiquidCore(c_in=8, c_h=16, mode="single")
    assert single.ch_bg is None
    rep = single.tau_report()
    assert "tau_single_median" in rep


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
    g = m.core.ch_bg.theta_tau.grad
    assert g is not None and torch.isfinite(g).all()  # τ 收到梯度（消融 b 前提）
    n = sum(p.numel() for p in m.parameters())
    assert n < 5e6  # 方案硬闸门：Params ≤ 5M


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
    assert float(recon_loss(y_b, x, torch.zeros(B, T, 1, H, W), box_full)) == 0.0


def test_decouple_loss_bounds():
    a = torch.randn(2, 3, 8, 16, 16)
    b = torch.randn(2, 3, 8, 16, 16)
    l = decouple_loss(a, b)
    assert 0.0 <= float(l) <= 1.0
    assert abs(float(decouple_loss(a, a)) - 1.0) < 1e-5  # 同向 → cos²=1（最大惩罚）
    assert float(decouple_loss(a, torch.zeros_like(a))) < 1e-3  # 零状态无惩罚


def test_focal_dice_empty_frames():
    """空标注帧（合法负样本）：Dice 项约定 0、Focal 正常，损失有限。"""
    lg = torch.randn(2, 1, 32, 32)
    tg = torch.zeros(2, 1, 32, 32)
    assert torch.isfinite(focal_dice_loss(lg, tg))


def test_param_count_budget():
    """方案 4.7 预算口径：1.0× 整机实测（骨干截断 + 颈 + 核心 + 头）≤ 5M。"""
    m = DsldCore(width=1.0, c_main=32, c_h=32)
    groups: dict[str, int] = {}
    for name, p in m.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    total = sum(groups.values())
    assert total < 5e6
    assert groups["core"] < 1e5  # 方案 4.7：液态核心 ≈ 0.03M 量级
