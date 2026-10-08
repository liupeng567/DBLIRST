"""DualStateLiquidCore 不变量单测（M3 v2.0 §8.3 第 3/4/5/6/7/9/11 项 + Δt 三项）。

每条都是"能红"的断言：把实现改坏（去掉终 clamp、去掉保护式更新、把 Δt 挪进自由门控分母、
逐帧收集诊断、EMA 吃 dw 上下文……）都会当场失败，不依赖肉眼判读。

坐标系/几何口径不在本文件（P0 已锁 tests/test_register.py + test_ittd_window.py）；
这里只锁**动力学**。
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dsld.models.liquid_core import (  # noqa: E402
    DualStateLiquidCore,
    _LiquidChannel,
    alpha_peaks,
    feedback_mask,
    scene_stats,
)

C, CH, H, W = 8, 6, 10, 12
BOX_R, BOX_C = slice(4, 8), slice(4, 8)        # EMA 用例的保护方块
SM_R, SM_C = slice(2, 5), slice(3, 7)          # teacher 冻结用例的保护方块
BOX_HW = torch.zeros(H, W, dtype=torch.bool)
BOX_HW[BOX_R, BOX_C] = True                    # [H,W] 方块布尔（EMA 用例）


def core(**kw) -> DualStateLiquidCore:
    kw.setdefault("mask_source", "off")
    kw.setdefault("c_h", CH)
    return DualStateLiquidCore(c_in=C, **kw)


def feats(B: int = 1, T: int = 6, seed: int = 0) -> torch.Tensor:
    return torch.rand(B, T, C, H, W, generator=torch.Generator().manual_seed(seed))


# ---- §8.3-3 状态有界（不变量 ②） ---------------------------------------------
def test_state_bounded_under_extreme_input_and_random_modulation():
    """|h| ≤ 1：极端幅值输入 + 随机 f/g 权重（tanh 顶到饱和）下仍逐位有界。

    能红方式：候选去掉 tanh、或更新式不再是凸组合（a_eff 可 >1）。
    """
    torch.manual_seed(1)
    mdl = core()
    with torch.no_grad():
        for ch in (mdl.ch_bg, mdl.ch_tg):
            ch.f_head.weight.normal_(0, 20.0)
            ch.g_head.weight.normal_(0, 5.0)
    out = mdl(torch.randn(1, 40, C, 6, 6) * 1e3)
    assert float(out["h_t"].abs().max()) <= 1.0 + 1e-6
    assert float(out["h_b"].abs().max()) <= 1.0 + 1e-6


# ---- §8.3-4 τ 区间不相交 ⇒ scale_ratio ≥ 3（不变量 ③） -------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_tau_intervals_disjoint_scale_ratio_ge_3_for_random_theta(seed):
    """随机 θ_λ + 随机调制下实测 τ_B≥24、τ_T≤8 ⇒ 比值 ≥3 是构造性质，与权重无关。

    能红方式：去掉 λ 的终 clamp（前版 461974a 实测比值可反转，判据 ④ 自证不了自己）。
    """
    mdl = core()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for ch in (mdl.ch_bg, mdl.ch_tg):
            ch.theta_lambda.uniform_(-15.0, 15.0, generator=g)
            ch.f_head.weight.normal_(0, 10.0, generator=g)
            ch.f_head.bias.uniform_(-3.0, 3.0, generator=g)
    mdl(feats(T=8))
    rep = mdl.tau_report()
    assert rep["tau_eff_b"] >= 24.0 - 1e-2
    assert rep["tau_eff_t"] <= 8.0 + 1e-2
    assert rep["tau_scale_ratio"] >= 3.0
    with torch.no_grad():
        for ch, lo, hi in ((mdl.ch_bg, 24.0, 192.0), (mdl.ch_tg, 2.0, 8.0)):
            t = ch.tau()
            assert float(t.min()) >= lo - 1e-3
            assert float(t.max()) <= hi + 1e-3


# ---- §8.3-5 f_head 零初始化 ⇒ tau_eff 精确 48/6 -------------------------------
def test_zero_init_fhead_gives_exact_starting_tau():
    """首帧 τ_eff 必须**精确**等于 config 的 48/6（L3：τ 叙事必须与配置同口径）。

    能红方式：f_head 改默认初始化（mod≠1）、σ 重参数化 init 公式写错、场景条件未零初始化。
    """
    m = core()
    m(feats())
    rep = m.tau_report()
    assert rep["tau_eff_b"] == pytest.approx(48.0, abs=1e-6)
    assert rep["tau_eff_t"] == pytest.approx(6.0, abs=1e-6)
    assert rep["tau_b_median"] == pytest.approx(48.0, abs=1e-6)
    assert rep["tau_t_median"] == pytest.approx(6.0, abs=1e-6)
    assert rep["lam_at_bound_b"] == 0.0 and rep["lam_at_bound_t"] == 0.0


# ---- §8.3-6 teacher 完美保护：膨胀区 h_B 跨帧逐位不变（不变量 ④，判据 ② 本体） ----
def test_teacher_mask_freezes_background_bitwise():
    """M≡1 ⇒ a_eff≡1 ⇒ h_B 自保护起始帧起逐位不变（torch.equal，不是 allclose）。

    两条防伪断言：保护区**外**照常更新、保护开始**前**一步确有更新——否则
    "h_B 恒零"这种退化实现也能通过逐位不变。
    """
    T = 8
    tm = torch.zeros(1, T, 1, H, W)
    tm[:, 4:, :, SM_R, SM_C] = 1.0
    out = core(mask_source="teacher")(feats(T=T), teacher_mask=tm)
    hb = out["h_b"][0]
    assert torch.equal(hb[4, :, SM_R, SM_C], hb[7, :, SM_R, SM_C])
    assert not torch.equal(hb[2, :, SM_R, SM_C], hb[3, :, SM_R, SM_C])  # 保护前该位置在更新
    assert not torch.equal(hb[7, :, 0, 0], hb[3, :, 0, 0])
    assert float(out["m_tgt"][0, 7, 0, 3, 4]) == 1.0   # teacher 上限 1.0，不被 m_max 截断


# ---- §8.3-7 gate 模式在核心层忽略 teacher（语义防护）+ 不变量 ⑤⑥ ----------------
def test_gate_mode_ignores_teacher_mask_and_starts_unmasked():
    """α₀ = σ(−2) = 0.119 < α_th ⇒ 起步"无证据不掩码"：m_tgt ≡ 0，teacher 传了也无效。

    能红方式：核心层去掉 mask_source 判定（误传 GT 就静默变成阳性对照臂，课程 C 失效）。
    """
    T = 6
    x = feats(T=T)
    tm = torch.zeros(1, T, 1, H, W)
    tm[:, :, :, SM_R, SM_C] = 1.0
    m = core(mask_source="gate")
    with_tm = m(x, teacher_mask=tm)
    without = m(x, teacher_mask=None)
    assert float(with_tm["m_tgt"].abs().max()) == 0.0
    assert torch.equal(with_tm["logits"], without["logits"])
    assert float(with_tm["alpha"][0, 0, 0, 0, 0]) == pytest.approx(0.11920292, abs=1e-6)


def test_teacher_source_without_mask_raises():
    """装配错误显式失败：mask_source=teacher 缺 GT ⇒ ValueError（不许静默退回 gate）。"""
    with pytest.raises(ValueError):
        core(mask_source="teacher")(feats(), teacher_mask=None)


def test_off_source_never_masks_even_with_teacher():
    """保护关闭臂（判据 ② 的对照）：即使误传 teacher_mask，M 也恒 0。"""
    out = core(mask_source="off")(feats(T=6), teacher_mask=torch.ones(1, 6, 1, H, W))
    assert float(out["m_tgt"].abs().max()) == 0.0


def test_gate_source_respects_m_max_cap():
    """把 α 顶过阈值（偏置 +3、末层权重仍零）⇒ M ≡ m_max=0.8，(1−M) ≥ 0.2 不被清零。

    第 0 帧必须不掩码：gate 源用的是**上一帧**的 α（t=0 无证据），这条同时锁住
    "自举滞后一帧"的时序语义。能红方式：cap 写成 1.0 或忘 clamp——L1 窒息故障
    被重新请回可行域。
    """
    m = core(mask_source="gate")
    torch.nn.init.constant_(m.gate[2].bias, 3.0)
    out = m(feats(T=4))
    assert float(out["m_tgt"][:, 0].abs().max()) == 0.0
    assert float(out["m_tgt"][:, 1:].min()) == pytest.approx(0.8, abs=1e-6)
    assert float(out["m_tgt"][:, 1:].max()) == pytest.approx(0.8, abs=1e-6)


# ---- Δt 语义（D-P1-1：ITTD 周期-3 补帧，实测见 P0 报告 §3） --------------------
def test_dt_zero_slot_freezes_state_bitwise():
    """补帧槽 dt=0 ⇒ a=exp(0)=1 ⇒ 状态逐位不变；等间隔对照必须变（防伪通过）。"""
    T = 6
    x = feats(T=T)
    dt = torch.ones(1, T)
    dt[0, 3] = 0.0
    hb = core()(x, dt=dt)["h_b"][0]
    assert torch.equal(hb[3], hb[2])
    hb_ref = core()(x, dt=torch.ones(1, T))["h_b"][0]
    assert not torch.equal(hb_ref[3], hb_ref[2])


def test_two_unit_steps_equal_one_double_step():
    """精确离散律：a=exp(−Δt·λ) ⇒ 两个 Δt=1 步 == 一个 Δt=2 步（常数输入 + 零初始化 FiLM）。

    能红方式：Δt 不进出指数（前版 p·g+(1−p)·m 的自由门控里 τ 根本不在指数上）。
    """
    m1 = core()
    x0 = feats(T=1)
    out2 = m1(x0.repeat(1, 2, 1, 1, 1), dt=torch.ones(1, 2))["h_b"][0, 1]
    m2 = copy.deepcopy(m1)
    out1 = m2(x0, dt=torch.tensor([[2.0]]))["h_b"][0, 0]
    assert torch.allclose(out2, out1, atol=1e-6), float((out2 - out1).abs().max())


def test_dt_with_zero_slots_runs_and_reports():
    """dt 含 0 槽时前向照常出图；时间进度在 GPU 上由前缀和算出（循环内不 float()）。"""
    out = core()(feats(T=4), dt=torch.tensor([[1.0, 0.0, 2.0, 1.0]]))
    assert tuple(out["logits"].shape) == (1, 4, 1, H, W)
    s = scene_stats(feats(T=1)[:, 0], None, torch.tensor([0.5]))
    assert tuple(s.shape) == (1, 5) and torch.isfinite(s).all()


# ---- §8.3-9 EMA 臂：指数收敛 + 掩码处恒旧值 + 目标零渗入（判据 ③ 对照臂） --------
def test_ema_background_converges_exponentially_with_perfect_exclusion():
    """常数场景下 y_b ≡ (1−0.9^{t+1})·x（解析真值），保护区恒 0，末帧残差比 ≥10×。

    能红方式：EMA 吃 dw 上下文（两臂预测的量不同 ⇒ 判据 ③ 的比较不对等）、
    忘做 ViBe 选择性更新、或动量未按 dt 幂次换算。
    """
    T, mom = 32, 0.9
    x = feats(T=1).repeat(1, T, 1, 1, 1)
    tm = torch.zeros(1, T, 1, H, W)
    tm[:, :, :, BOX_R, BOX_C] = 1.0
    out = core(bg_mode="ema", mask_source="teacher", ema_momentum=mom)(
        x, dt=torch.ones(1, T), teacher_mask=tm)
    yb, r = out["y_b"], out["r"]
    t = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1)
    expect = (1.0 - mom ** (t + 1.0)) * x
    sel = (~BOX_HW).view(1, 1, 1, H, W).expand_as(yb)
    assert torch.allclose(yb[sel], expect[sel], atol=1e-5)
    assert torch.equal(yb[:, :, :, BOX_R, BOX_C],
                       torch.zeros_like(yb[:, :, :, BOX_R, BOX_C]))
    r_last = r[0, -1].abs().mean(dim=0)                    # [H,W] 通道均值
    prot = float(r_last[BOX_R, BOX_C].mean())
    bg = float(r_last[~BOX_HW].mean())
    assert prot / (bg + 1e-12) >= 10.0, (prot, bg)


# ---- §8.3-11 诊断末帧门控（L5：禁逐帧 GPU→CPU 同步） ---------------------------
def test_diagnostics_collected_only_on_last_frame(monkeypatch):
    """T 帧只允许 2 次 collect_stats=True（双通道各一次），其余 2(T−1) 次为 False。"""
    T = 7
    calls: list[bool] = []
    orig = _LiquidChannel.step

    def spy(self, *a, **kw):
        calls.append(bool(kw.get("collect_stats", True)))
        return orig(self, *a, **kw)

    monkeypatch.setattr(_LiquidChannel, "step", spy)
    m = core()
    m(feats(T=T))
    assert len(calls) == 2 * T
    assert calls.count(True) == 2 and calls.count(False) == 2 * (T - 1)
    assert {"h_t_rms", "h_b_rms", "m_frac", "alpha_mean"} <= set(m.last_norms)


# ---- 四臂开关的构造性差异（§4.5） ---------------------------------------------
def test_single_arm_aliases_hb_and_union_tau():
    """single 臂 h_b 与 h_t 同一张量（cos²≡1），τ 区间取并集、init 取几何均值。"""
    m = core(state_mode="single")
    out = m(feats(T=4))
    assert torch.equal(out["h_t"], out["h_b"])
    rep = m.tau_report()
    assert rep["tau_eff_single"] == pytest.approx(math.sqrt(48.0 * 6.0), abs=0.01)
    assert "tau_b_median" not in rep and "tau_single_median" in rep
    assert rep["tau_single_median"] == pytest.approx(math.sqrt(48.0 * 6.0), abs=0.01)


def test_swap_arm_exchanges_tau_ranges():
    """tau_mode=swap：两通道 τ 范围整体互换（判据 ④ 的因果臂，其余一律不变）。"""
    m = core(tau_mode="swap")
    m(feats(T=4))
    rep = m.tau_report()
    assert rep["tau_b_median"] == pytest.approx(6.0, abs=1e-6)
    assert rep["tau_t_median"] == pytest.approx(48.0, abs=1e-6)
    assert rep["tau_eff_b"] < rep["tau_eff_t"]


def test_state_dependent_head_takes_state_input():
    """state_dependent=true ⇒ 头输入含 h（真"液态"消融 a5），通道数翻倍于默认臂。"""
    a = core(state_dependent=False).ch_bg
    b = core(state_dependent=True).ch_bg
    assert a.f_head.in_channels == C and b.f_head.in_channels == C + CH


# ---- 掩码递推与峰值膨胀的纯函数口径（§4.3.4，与核心解耦可单测） -----------------
def test_feedback_mask_recursion_and_cap():
    prev = torch.full((1, 1, 4, 4), 0.7)
    src = torch.zeros(1, 1, 4, 4)
    src[0, 0, 1, 1] = 0.5
    m = feedback_mask(prev, src, decay=0.9, cap=0.8)
    assert float(m[0, 0, 1, 1]) == pytest.approx(0.63, abs=1e-6)   # 衰减项胜出（0.5<0.63）
    assert float(m.max()) == pytest.approx(0.63, abs=1e-6)
    m2 = feedback_mask(prev, torch.full((1, 1, 4, 4), 1.0), decay=0.9, cap=0.8)
    assert float(m2.max()) == pytest.approx(0.8, abs=1e-6)         # 上限生效
    pk = alpha_peaks(torch.arange(16.0).reshape(1, 1, 4, 4) / 15.0, radius=1, alpha_th=0.6)
    assert float(pk[0, 0, 3, 3]) == 1.0 and float(pk[0, 0, 0, 0]) == 0.0
    assert float(pk[0, 0, 2, 2]) == 1.0                            # 膨胀覆盖邻域
