"""快评（7.5）与早停（表 T）单元测试：合成缓存端到端 + 早停状态机。

M3 代码审核口径修正的回归测试：
  - gt_mask_anchor：GT 掩码随末帧 warp 到锚点坐标（残差图同坐标系）；
  - 快评 quality 与训练同口径（窗口 quality.npy 中位数 → FiLM 第 5 维）；
  - single/dual 损失组成拉平（single 也计 L_dec）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsld.models.dsld_core import DsldCore  # noqa: E402
from dsld.train.trainer import (  # noqa: E402
    _dsld_core_loss,
    _quick_eval,
    early_stop_step,
    gt_mask_anchor,
)
from tests.test_ittd_window import _make_aligned_cache  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def _make_manifest_val(root: Path, seq_id: int, n: int) -> Path:
    """val-int 划分的测试 manifest（快评读 val-int）。"""
    data = {"splits": {"val-int": {"seqs": [seq_id]}},
            "sequences": [{"seq_id": seq_id, "n_frames": n}]}
    ck = "md5:" + hashlib.md5(json.dumps(
        {"splits": data["splits"], "sequences": data["sequences"]},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    data["checksum"] = ck
    p = root / "manifest_qe.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _mini_cfg() -> OmegaConf:
    return OmegaConf.create({
        "train": {"window": {"T": 8},
                  "quick_eval": {"n_seqs": 1, "n_windows": 2},
                  "early_stop": {"enabled": False}},
        "data": {"manifest": "manifest_qe.json", "cache_root": ""},
        "model": {"type": "dsld_core"},
    })


def test_quick_eval_end_to_end_synthetic(tmp_path):
    """合成缓存上跑通快评：记录键齐全、数值有限、τ/α/质量字段在位。"""
    seq_id, n = 9101, 60
    _make_aligned_cache(tmp_path, seq_id, n)
    mf = _make_manifest_val(tmp_path, seq_id, n)
    cfg = _mini_cfg()
    model = DsldCore(width=0.8, c_main=16, c_h=8)
    model.eval()
    rec = _quick_eval(model, cfg, "cpu", manifest_path=str(mf), cache_root=str(tmp_path))

    assert rec["n_seqs"] == 1 and rec["n_frames"] > 0
    # 有效帧数 = 窗数 × (T − warmup) = 2 × 6
    assert rec["n_frames"] == 2 * 6
    assert rec["subset"][0]["seq_id"] == seq_id
    assert rec["subset"][0]["starts"] == [0, 52]  # 固定起点（确定性子集）
    for key in ("primary", "thr_sweep", "fa_frm_pd90", "fa_pix_e6_pd90",
                "recall_pd90", "pd90_available", "bg_resid_rms", "resid_scr", "bg_frac",
                "alpha_mean", "alpha_frac_high", "alpha_p99",
                "quality_med", "per_seq", "stratified",
                "tau_b_median", "tau_t_median",
                "tau_eff_b", "tau_eff_t", "tau_scale_ratio",
                "lam_at_bound_b", "lam_at_bound_t"):
        assert key in rec, f"缺快评字段 {key}"
    # C2 分层：合成段无属性 → 归 clean 组；per_seq 每段一行
    assert rec["stratified"]["clean"]["n_seqs"] == 1
    assert rec["stratified"]["distractor"]["n_seqs"] == 0
    assert len(rec["per_seq"]) == 1 and rec["per_seq"][0]["seq_id"] == seq_id
    assert np.isfinite(rec["fa_frm_pd90"]) and np.isfinite(rec["bg_resid_rms"])
    assert 0.0 <= rec["alpha_mean"] <= 1.0
    assert rec["primary"]["iou_thr"] == 0.5
    # 合成缓存 quality.npy 恒 1 → 窗口质量中位 1（训练/快评 FiLM 同口径的前提）
    assert abs(rec["quality_med"] - 1.0) < 1e-6
    # 记录可 json 序列化（metrics.jsonl 落盘前提，np 类型会炸）
    json.dumps(rec)


def test_quick_eval_deterministic(tmp_path):
    """固定子集两次快评指标逐位一致（监控可比性前提）。"""
    seq_id, n = 9102, 60
    _make_aligned_cache(tmp_path, seq_id, n)
    mf = _make_manifest_val(tmp_path, seq_id, n)
    cfg = _mini_cfg()
    model = DsldCore(width=0.8, c_main=16, c_h=8).eval()
    a = _quick_eval(model, cfg, "cpu", manifest_path=str(mf), cache_root=str(tmp_path))
    b = _quick_eval(model, cfg, "cpu", manifest_path=str(mf), cache_root=str(tmp_path))
    assert a["primary"] == b["primary"]
    assert a["fa_frm_pd90"] == b["fa_frm_pd90"]


def test_gt_mask_anchor_warp_alignment():
    """评审口径修正：GT 掩码随末帧 warp 到锚点坐标（整数平移，NEAREST 无边界歧义），
    再 2×2 块最大下采样；恒等 warp 不重采样。"""
    from dsld.train.trainer import gt_mask_anchor

    boxes = [(6, 20, 10, 30, 20)]  # (f, x1, y1, x2, y2) 末帧原始坐标
    # 恒等：native [10:20, 20:30] → 2×2 块最大 [5:10, 10:15]
    m0 = gt_mask_anchor(boxes, 6, None, (32, 40), (16, 20))
    assert m0.shape == (16, 20)
    assert m0[5:10, 10:15].all() and int(m0.sum()) == 25
    assert not m0[:5, :].any() and not m0[10:, :].any()
    # 整数平移 (dx=4, dy=6)：WARP_INVERSE_MAP 下 dst(x,y)=src(x+4, y+6)
    # ⇒ 锚点坐标框 = [4:14, 16:26] → 块最大 [2:7, 8:13]
    warp = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 6.0]], np.float32)
    m1 = gt_mask_anchor(boxes, 6, warp, (32, 40), (16, 20))
    assert m1[2:7, 8:13].all() and int(m1.sum()) == 25
    assert not m1[:2, :].any() and not m1[7:, :].any()
    # warp 方向证伪（旧 bug：无 WARP_INVERSE_MAP 时框落在 [8:13, 12:17]）
    assert not m1[8:, :].any()

def test_dsld_core_loss_single_mode_decouple_present():
    """损失组成两臂拉平：single 模式也计 L_dec（h_b≡h_t ⇒ cos²≡1 常数 + 范数守卫），
    单/双消融的 F_a 差不再混杂正则项有无。"""
    model = DsldCore(width=0.8, c_main=16, c_h=8, liquid_mode="single")
    x = torch.rand(1, 2, 1, 32, 32)
    target = (torch.rand(1, 2, 1, 32, 32) > 0.995).float()
    loss, parts = _dsld_core_loss(model, x, target, None,
                                  {"seg": 1.0, "recon": 0.5, "decouple": 0.1})
    assert "decouple" in parts
    assert 0.99 <= parts["decouple"] <= 1.01  # h_b≡h_t ⇒ cos²≡1
    assert torch.isfinite(loss)
    # dual 臂同字段在位（对照）
    model_d = DsldCore(width=0.8, c_main=16, c_h=8, liquid_mode="dual")
    _, parts_d = _dsld_core_loss(model_d, x, target, None,
                                 {"seg": 1.0, "recon": 0.5, "decouple": 0.1})
    assert "decouple" in parts_d and parts_d["decouple"] < 1.0


def test_early_stop_state_machine():
    """表 T 早停：改善归零 / 连续无改善计数 / 关闭开关 / inf（P_d 未达）不计改善。"""
    cfg_on = {"enabled": True, "patience": 3}
    cfg_off = {"enabled": False, "patience": 1}
    st = {"best": float("inf"), "patience": 0}
    assert not early_stop_step(st, 5.0, cfg_on) and st["best"] == 5.0 and st["patience"] == 0
    assert not early_stop_step(st, 3.0, cfg_on) and st["best"] == 3.0
    assert not early_stop_step(st, 4.0, cfg_on) and st["patience"] == 1
    assert not early_stop_step(st, 4.0, cfg_on) and st["patience"] == 2
    assert early_stop_step(st, 4.0, cfg_on) and st["patience"] == 3  # patience 用尽 → 停
    # 改善后重新计数
    st2 = {"best": 2.0, "patience": 2}
    assert not early_stop_step(st2, 1.5, cfg_on) and st2["patience"] == 0
    # P_d 未达 0.90（fa=inf）只计耐心、不刷新 best
    st3 = {"best": 2.0, "patience": 0}
    assert not early_stop_step(st3, float("inf"), cfg_on) and st3["best"] == 2.0
    # 关闭早停：永不判停
    st4 = {"best": float("inf"), "patience": 0}
    for _ in range(10):
        assert not early_stop_step(st4, 9.0, cfg_off)
