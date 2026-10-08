"""四分量损失单测（M3 v2.0 §8.3 第 8/10 项 + L_recon 排除几何 + 诊断口径）。

口径要点（每条对应一个能红的断言）：
  · L_gate 平衡 BCE 的不变量 α≡c ⇒ L = ln(1/(1−c)) + ln(1/c)，与框大小/正例数无关
    （c=0.5 即 2·ln2）——朴素 BCE 的"背景项淹没"死锁必须被这一项结构性排除；
  · L_dec 在 h≡0 坍缩时必须给**正**损失（旧式 cos² 在双零时抹平为 0，恰好放过坍缩）；
    single 臂 h_b≡h_t ⇒ cos²≡1 常数，两臂损失组成一致；
  · L_recon 的排除集 = teacher ∪ 内环掩码 ∪ 残差 top-k，且**不含任何内部膨胀**
    （膨胀/下采样只在 register 一处，L4）；有效面塌缩 ⇒ strict 抛错；
  · 残差诊断无前景时给 NaN 不给 0（L2：不许有"不可能失败"的读数）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dsld.train.losses import (  # noqa: E402
    decouple_loss,
    focal_dice_loss,
    gate_bce_loss,
    recon_loss,
    resid_diagnostics,
    total_loss,
)

B, T, C, H, W = 2, 3, 4, 8, 10
LN2 = math.log(2.0)


def gt_with_boxes(n_boxes_per_frame: list[int]) -> torch.Tensor:
    """[B,T,1,H,W] 的框填充 GT：每帧按给定个数放 2×2 方块（位置固定、互不重叠）。"""
    gt = torch.zeros(B, T, 1, H, W)
    for b in range(B):
        for t, n in enumerate(n_boxes_per_frame):
            for i in range(n):
                gt[b, t, 0, 1 + 2 * i:3 + 2 * i, 1 + 3 * (i % 3):3 + 3 * (i % 3)] = 1.0
    return gt


# ---- §8.3-8 平衡 BCE 不变量 --------------------------------------------------
def test_gate_bce_invariant_value_independent_of_box_size():
    """α≡c ⇒ L ≡ −ln(c) − ln(1−c)，正例 1/3/9 个框全部同值（朴素 BCE 会随框数漂移）。

    c=0.5 即 2·ln2——这条是"目标极稀少时背景项淹没正例、把 α 推成全局 0"（门控死锁
    成因）被结构性排除的机器证明。
    """
    for counts in ([1, 0, 0], [3, 3, 3], [1, 3, 2]):
        gt = gt_with_boxes(counts)
        alpha = torch.full((B, T, 1, H, W), 0.5)
        assert float(gate_bce_loss(alpha, gt)) == pytest.approx(2 * LN2, abs=1e-5), counts
        alpha2 = torch.full((B, T, 1, H, W), 0.2)
        assert float(gate_bce_loss(alpha2, gt)) == pytest.approx(
            -math.log(0.2) - math.log(0.8), abs=1e-5), counts


def test_gate_bce_empty_frame_counts_background_only():
    """全背景窗（合法负样本）只计背景项 ⇒ α≡0.5 时损失 = ln2，不是 2·ln2。"""
    gt = torch.zeros(B, T, 1, H, W)
    alpha = torch.full((B, T, 1, H, W), 0.5)
    assert float(gate_bce_loss(alpha, gt)) == pytest.approx(LN2, abs=1e-5)


def test_gate_bce_pushes_alpha_up_only_where_gt():
    """GT 处 α 高、背景处 α 低 ⇒ 损失显著低于反向安排（方向性，不只是数值）。"""
    gt = gt_with_boxes([2, 2, 2])
    good = torch.where(gt > 0.5, 0.9, 0.1)
    bad = torch.where(gt > 0.5, 0.1, 0.9)
    assert float(gate_bce_loss(good, gt)) < float(gate_bce_loss(bad, gt))
    assert float(gate_bce_loss(good, gt)) < 1.0 < float(gate_bce_loss(bad, gt))


# ---- §8.3-10 L_dec：坍缩必正损失 + single 臂常数化 ----------------------------
def test_dec_punishes_zero_collapse_and_is_symmetric():
    z = torch.zeros(B, T, C, H, W)
    assert float(decouple_loss(z, z)) > 0.0                  # 双零不得抹平
    assert torch.isfinite(decouple_loss(z, z, min_norm=0.0))  # 关掉下限项也不许 0/0→NaN
    h = torch.randn(B, T, C, H, W)
    assert float(decouple_loss(h, h.clone())) > 1.0 - 1e-3    # 同一表示 ⇒ cos²≈1
    a = torch.zeros(B, T, C, H, W)
    a[:, :, : C // 2] = 1.0
    b = torch.zeros(B, T, C, H, W)
    b[:, :, C // 2:] = 1.0
    assert float(decouple_loss(a, b)) < 1e-6                  # 正交 ⇒ 无惩罚
    assert float(decouple_loss(a, b)) == pytest.approx(float(decouple_loss(b, a)), abs=1e-6)


def test_dec_norm_floor_fires_only_below_min_norm():
    """范数下限项：小范数（低于 min_norm）叠加惩罚，健康范数不叠加。"""
    small = torch.full((1, 1, 4, 4, 4), 1e-3)
    other = torch.rand(1, 1, 4, 4, 4)
    base = float(decouple_loss(small, other, min_norm=1e-2, norm_weight=1.0))
    no_floor = float(decouple_loss(small, other, min_norm=1e-2, norm_weight=0.0))
    assert base > no_floor
    big = torch.rand(1, 1, 4, 4, 4) + 1.0
    assert float(decouple_loss(big, other, min_norm=1e-2, norm_weight=1.0)) == pytest.approx(
        float(decouple_loss(big, other, min_norm=1e-2, norm_weight=0.0)), abs=1e-6)


# ---- L_seg：Focal+soft-Dice --------------------------------------------------
def test_focal_dice_direction_and_empty_frame_handling():
    gt = gt_with_boxes([2, 2, 2])
    perfect = torch.where(gt > 0.5, 8.0, -8.0)
    terrible = torch.where(gt > 0.5, -8.0, 8.0)
    assert float(focal_dice_loss(perfect, gt)) < 1e-3
    assert float(focal_dice_loss(terrible, gt)) > float(focal_dice_loss(perfect, gt))
    empty = torch.zeros(B, T, 1, H, W)
    l_quiet = focal_dice_loss(torch.full_like(empty, -8.0), empty)
    assert torch.isfinite(l_quiet) and float(l_quiet) < 0.05   # 空帧上低置信预测 ≈ 无代价
    l_noisy = focal_dice_loss(torch.full_like(empty, 8.0), empty)
    assert float(l_noisy) > 0.4                               # 空帧全图高置信 = 满屏虚警，重罚


# ---- L_recon：排除集几何 + valid_frac + strict 拦截 ---------------------------
def test_recon_excludes_teacher_and_mask_regions():
    x = torch.rand(B, T, C, H, W)
    yb = torch.zeros_like(x)                                   # 残差 = x
    teacher = torch.zeros(B, T, 1, H, W)
    teacher[:, :, 0, :4, :] = 1.0                              # 上半图排除
    m = torch.zeros_like(teacher)
    loss, frac = recon_loss(yb, x, teacher, m, top_ratio=0.0, strict=False)
    assert frac == pytest.approx(0.5, abs=0.02)                # 排除一半 ⇒ 0.5
    expect = x[:, :, :, 4:, :].mean()
    assert float(loss) == pytest.approx(float(expect), rel=1e-5)


def test_recon_topk_removes_bright_outliers():
    x = torch.full((1, 1, 1, 100, 100), 0.1)
    x[:, :, :, 0, 0] = 9.0                                     # 一个瞬时亮点
    yb = torch.zeros_like(x)
    zero = torch.zeros(1, 1, 1, 100, 100)
    loss_off, frac_off = recon_loss(yb, x, zero, zero, top_ratio=0.0, strict=False)
    loss_on, frac_on = recon_loss(yb, x, zero, zero, top_ratio=0.01, strict=False)
    assert float(loss_on) < float(loss_off)                    # 亮点被剔出背景监督
    assert frac_on < frac_off


def test_recon_strict_raises_when_background_face_collapses():
    """掩码铺满全图 ⇒ 背景监督面塌缩：strict 必须抛错（前版静默返回 0 的窒息实录）。"""
    x = torch.rand(1, 2, 2, 6, 6)
    m_all = torch.ones(1, 2, 1, 6, 6)
    with pytest.raises(RuntimeError):
        recon_loss(torch.zeros_like(x), x, m_all, m_all, min_valid_frac=0.5, strict=True)
    with pytest.warns(RuntimeWarning):
        out = recon_loss(torch.zeros_like(x), x, m_all, m_all, min_valid_frac=0.5,
                         strict=False)
    assert out[1] == pytest.approx(0.0, abs=1e-6)


# ---- 残差诊断（§4.3.5 口径挪到损失层：需要 GT，核心不碰监督） ------------------
def test_resid_diagnostics_nan_instead_of_zero_when_no_gt():
    r = torch.rand(1, 2, 3, 6, 6)
    no_gt = torch.zeros(1, 2, 1, 6, 6)
    d = resid_diagnostics(r, no_gt)
    assert math.isnan(d["resid_scr"])                       # 不许读成"完美=0"
    assert d["bg_resid_rms"] > 0


def test_resid_diagnostics_scr_measures_protection_gain():
    """保护得好 ⇒ GT 处残差远高于背景区 ⇒ resid_scr > 1；背景吸收目标 ⇒ scr ≈ 1。"""
    Hh = 8
    gt = torch.zeros(1, 1, 1, Hh, Hh)
    gt[0, 0, 0, 3:5, 3:5] = 1.0
    r_good = torch.ones(1, 1, 3, Hh, Hh) * 0.05
    r_good[:, :, :, 3:5, 3:5] = 1.0                          # 目标仍在残差里
    r_bad = torch.ones(1, 1, 3, Hh, Hh) * 0.05
    d_good = resid_diagnostics(r_good, gt, gt)
    d_bad = resid_diagnostics(r_bad, gt, gt)
    assert d_good["resid_scr"] > 5.0
    assert d_bad["resid_scr"] == pytest.approx(1.0, abs=1e-3)


# ---- total_loss：组装、权重、哨兵字段 ----------------------------------------
def _fake_out() -> dict:
    torch.manual_seed(0)
    x = torch.rand(B, T, C, H, W)
    yb = 0.9 * x
    return {"logits": torch.zeros(B, T, 1, H, W, requires_grad=True),
            "alpha": torch.full((B, T, 1, H, W), 0.5, requires_grad=True),
            "y_b": yb.clone().requires_grad_(), "r": x - yb, "x_main": x,
            "h_t": torch.randn(B, T, C, H, W, requires_grad=True),
            "h_b": torch.randn(B, T, C, H, W, requires_grad=True),
            "m_tgt": torch.zeros(B, T, 1, H, W), "norms": {"alpha_mean": 0.5},
            "tau": {"tau_eff_b": 48.0}}


def test_total_loss_weights_and_diag_fields():
    out = _fake_out()
    batch = {"target": gt_with_boxes([2, 2, 2]), "teacher_mask": torch.zeros(B, T, 1, H, W)}
    w = {"seg": 1.0, "recon": 0.5, "gate": 0.3, "dec": 0.1}
    r1 = total_loss(out, batch, w)
    w2 = dict(w, recon=1.0)
    r2 = total_loss(out, batch, w2)
    assert float(r2["loss"]) > float(r1["loss"])             # recon 权重加倍 ⇒ 总损失变大
    assert float(r1["gate"]) == pytest.approx(2 * LN2, abs=1e-4)
    assert {"recon_valid_frac", "resid_scr", "bg_resid_rms", "alpha_mean"} <= set(r1["diag"])
    assert r1["tau"]["tau_eff_b"] == 48.0
    assert torch.isfinite(r1["loss"]) and r1["loss"].requires_grad


def test_total_loss_requires_all_four_weights():
    out = _fake_out()
    batch = {"target": gt_with_boxes([1, 1, 1]), "teacher_mask": torch.zeros(B, T, 1, H, W)}
    for drop in ("seg", "recon", "gate", "dec"):
        w = {"seg": 1.0, "recon": 0.5, "gate": 0.3, "dec": 0.1}
        del w[drop]
        with pytest.raises(KeyError):
            total_loss(out, batch, w)
