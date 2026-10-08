"""P0 窗口管线单测（M3 v2.0 方案 §8.2 P0 Gate + §8.3 判据清单 P0 部分）。

Gate 构成（每条都能红）：
  · 对齐语义：锚点矩形与像素 warp 同治（真实 reg.npz）、GT 随帧进锚点系；
  · §4.7 裁剪保完整约束：合成真值四场景（all_tracks / one_track / negative / fallback 重采）
    + 真实缓存统计不出现 fallback；
  · 监督张量几何：target/teacher 的 stride-2 关系、teacher ⊇ target、
    与 losses 侧 F.max_pool2d 公式逐位等价（"与 L_recon 剔除区同几何"的机器证明）；
  · 合法负样本、R4 配准失败密度门限、逐样本确定性、quality_med 口径、track_index 消费；
  · 合成注入：幅值 ∈[μ+4σ, μ+8σ]、存活 1–3 帧、禁落区不碰 GT、热斑 30 帧缓升。

合成缓存用真实 manifest（属性/帧数走 v3）+ tmp_path 下的假 seq_0001（96×128、250 帧），
使对齐/裁剪的期望值可以**解析算出**；真实缓存用例只取少量窗口控时间。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dsld.data.ittd_window_dataset import IttdWindows, _level, _to_anchor_rect  # noqa: E402
from dsld.data.preprocess.register import (  # noqa: E402
    box_fill,
    dilate_mask,
    downsample_max,
    gt_masks_anchor,
    window_anchor_warps,
)

MANIFEST = str(REPO / "data" / "manifests" / "ittd_split_v3.json")
CACHE = str(REPO / "data" / "cache" / "ittd")
H_F, W_F = 96, 128  # 合成缓存尺寸（小图，测快）
N_F = 250          # 与 manifest seq_0001 的 n_frames 一致


# --------------------------------------------------------------------------- #
# 合成缓存构造
# --------------------------------------------------------------------------- #
def write_fake_cache(root: Path, *, boxes, trans, failed=None, n_frames=N_F,
                     h=H_F, w=W_F, track_ids=None, frames=None):
    """写一份最小可用缓存：frames/norm_stats/nuc/deadpix/labels/quality/reg。

    trans[t] = (dx, dy)：第 t 帧相对参考帧的**正向**平移（M[t]=[ [1,0,dx],[0,1,dy] ]），
    与 register.py 的 ref→帧 约定一致；锚点系内容位移为 −(Δt−Δa)。
    """
    (root / "seq_0001").mkdir(parents=True, exist_ok=True)
    d = root / "seq_0001"
    if frames is None:
        frames = np.zeros((n_frames, h, w), np.uint8)
        rng = np.random.default_rng(0)
        frames[:] = rng.integers(100, 110, (n_frames, h, w), dtype=np.uint8)  # 弱纹理背景
    mm = np.lib.format.open_memmap(d / "frames.u8.npy", mode="w+", dtype=np.uint8,
                                   shape=frames.shape)
    mm[:] = frames
    mm.flush()
    np.save(d / "norm_stats.npy", np.stack(
        [np.full(n_frames, 105.0, np.float32), np.full(n_frames, 3.0, np.float32)], 1))
    np.save(d / "nuc_field.npy", np.zeros((h, w), np.float32))
    np.save(d / "deadpix.npy", np.zeros((0, 2), np.int32))
    np.save(d / "quality.npy", np.linspace(80.0, 140.0, n_frames).astype(np.float32))
    tid = np.asarray(track_ids if track_ids is not None
                     else [1] * len(boxes), np.int32)
    np.savez_compressed(
        d / "labels.npz",
        boxes=np.asarray(boxes, np.int32).reshape(-1, 5),
        track_ids=tid,
        track_ids_index=np.unique(tid).astype(np.int32),
        track_offsets=np.array([0, len(tid)], np.int64),
        track_frames=np.asarray(boxes, np.int32).reshape(-1, 5)[:, 0],
    )
    M = np.zeros((n_frames, 2, 3), np.float64)
    for t, (dx, dy) in enumerate(trans):
        M[t] = [[1.0, 0, dx], [0, 1.0, dy]]
    bridge = np.full((n_frames, 2, 3), np.nan)
    np.savez_compressed(
        d / "reg.npz", M=M.astype(np.float32), rmse=np.zeros(n_frames, np.float32),
        failed=np.asarray(failed if failed is not None else np.zeros(n_frames, bool), bool),
        method=np.array(["klt"] * n_frames, dtype="U4"),
        ref_idx=np.zeros(n_frames, np.int32), bridge_M=bridge.astype(np.float32),
        bridge_rmse=np.full(n_frames, np.inf, np.float32),
    )
    return d


def ident_trans(n):
    return [(0.0, 0.0)] * n


def ds_from_fake(root: Path, **kw):
    kw.setdefault("manifest_path", MANIFEST)
    kw.setdefault("cache_root", str(root))
    kw.setdefault("split", "train-int")
    kw.setdefault("limit_seqs", 1)
    kw.setdefault("crop", None)
    kw.setdefault("augment", False)
    kw.setdefault("start_jitter", False)  # 合成用例要解析真值：起点固定在基窗
    return IttdWindows(**kw)


def has_complete_box_with_margin(mask_tw: np.ndarray, margin: int = 3) -> bool:
    """独立判据：窗内**至少一帧**存在一个外接框完整落在离四边 margin 像素以外的连通域。

    用连通域标记核对（不复用数据集的区间算术），使"≥1 完整 GT 框含边距"这一
    §4.7 约束可被外部证伪；被裁半的目标其连通域必然贴边，故判据成立即约束成立。
    """
    for t in range(mask_tw.shape[0]):
        m = (mask_tw[t] > 0).astype(np.uint8)
        if not m.any():
            continue
        n, lab = cv2.connectedComponents(m, connectivity=8)
        for i in range(1, n):
            ys, xs = np.nonzero(lab == i)
            if (ys.min() >= margin and xs.min() >= margin
                    and ys.max() <= mask_tw.shape[1] - 1 - margin
                    and xs.max() <= mask_tw.shape[2] - 1 - margin):
                return True
    return False


# --------------------------------------------------------------------------- #
# 1) 对齐语义
# --------------------------------------------------------------------------- #
def test_anchor_rect_matches_pixel_warp_on_real_reg():
    """锚点矩形（_to_anchor_rect）与像素 warp（warp_frame）必须同治（真实 reg.npz）。

    独立验证路径：把框填成掩码、按 warp_frame 的语义 warp 后取外接框，
    与解析矩形对照——两者若方向取逆就会差 2·位移，绝无可能巧合通过。
    """
    sid = 21
    with np.load(f"{CACHE}/seq_{sid:04d}/reg.npz") as z:
        reg = {k: z[k] for k in ("M", "ref_idx", "bridge_M")}
    lab = np.load(f"{CACHE}/seq_{sid:04d}/labels.npz")
    boxes = np.asarray(lab["boxes"], np.int32)
    start, T = 40, 32
    Ws = window_anchor_warps(reg, start, T)
    hw = (480, 640)
    checked = 0
    for row in boxes[(boxes[:, 0] >= start + 1) & (boxes[:, 0] < start + T + 1)]:
        f, x1, y1, x2, y2 = (int(v) for v in row)
        t = f - 1 - start
        rect = _to_anchor_rect(Ws[t], x1, y1, x2, y2)
        warped = cv2.warpAffine(
            box_fill(np.array([row]), hw), Ws[t].astype(np.float32), (hw[1], hw[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        ys, xs = np.nonzero(warped)
        assert len(ys) > 0, "真实配准 warp 把 GT 移出了画面（应仍可核对）"
        assert abs(xs.min() - rect[0]) <= 1.5 and abs(ys.min() - rect[1]) <= 1.5, (
            f"锚点矩形左上 {rect[:2]} vs 像素 warp {xs.min()},{ys.min()}")
        assert abs(xs.max() + 1 - rect[2]) <= 1.5 and abs(ys.max() + 1 - rect[3]) <= 1.5, (
            f"锚点矩形右下 {rect[2:]} vs 像素 warp {xs.max() + 1},{ys.max() + 1}")
        checked += 1
    assert checked >= 5, f"真实窗口仅核对 {checked} 框，样本不足以证伪方向"


def test_synthetic_window_frames_and_gt_move_together(tmp_path):
    """合成纯平移窗：帧内容与 GT 掩码进锚点系的位移必须完全一致（L4 同治契约）。"""
    n, T = N_F, 16
    trans = [(1.5 * t, 0.8 * t) for t in range(n)]  # 正向 ref→帧（register.py 约定）
    # 目标画在各帧**同一像素位置**（不随平移变化）→ 对齐后在锚点系逐帧漂移
    frames = np.full((n, H_F, W_F), 105, np.uint8)
    for t in range(n):
        frames[t, 50:54, 70:74] = 250
    boxes = [[f + 1, 70, 50, 73, 53] for f in range(T)]  # 帧 1..16（1-based）
    write_fake_cache(tmp_path, boxes=boxes, trans=trans, frames=frames)

    ds = ds_from_fake(tmp_path, T=T, stride=8, align=True, feature_stride=1)
    it = ds[0]
    assert it["start0"] == 0 and it["windows"].shape == (T, 1, H_F, W_F)
    for t in range(T):
        m_t = it["target"][t, 0]
        ys, xs = np.nonzero(m_t)
        exp = (71.5 - 1.5 * t, 51.5 - 0.8 * t)  # 锚点系 = 帧系位移 −(Δt−Δ0)
        assert m_t.sum() == 16, f"帧 {t} GT 面积被 warp 破坏（{m_t.sum()}）"
        assert abs(xs.mean() - exp[0]) <= 1.0 and abs(ys.mean() - exp[1]) <= 1.0, (
            f"帧 {t} GT 中心 {xs.mean():.2f},{ys.mean():.2f} ≠ 期望 {exp}")
        # 同治性：掩码覆盖处的像素确实是那个亮块。双线性重采样会让 4px 方块边缘
        # 拖尾（阈值 0.9 打的是物理不是错位），故另加"整体平移 3px"的对照下界。
        hit = float(it["windows"][t, 0][m_t > 0].mean())
        assert hit > 0.75, f"帧 {t} 掩码与像素错位（覆盖率 {hit:.3f}）"
        if t >= 2:  # 前 2 帧位移 <2px，平移对照尚与真值重叠，跳过
            m_shift = np.roll(m_t, 3, axis=1)
            miss = float(it["windows"][t, 0][m_shift > 0].mean())
            assert hit - miss > 0.2, f"帧 {t} 平移对照未拉开（{hit:.3f} vs {miss:.3f}）"

    # 对照组：关对齐 → 帧与掩码都不 warp，掩码固定在绘制位置
    ds_off = ds_from_fake(tmp_path, T=T, stride=8, align=False, feature_stride=1)
    it_off = ds_off[0]
    ys, xs = np.nonzero(it_off["target"][3, 0])
    assert abs(xs.mean() - 71.5) <= 1.0 and abs(ys.mean() - 51.5) <= 1.0
    assert float(it_off["windows"][3, 0][it_off["target"][3, 0] > 0].mean()) > 0.9
    # 两组的位移模式必须不同（否则说明 align 开关没接上）
    assert not np.array_equal(it["target"][8, 0], it_off["target"][8, 0])


def test_gt_masks_anchor_drops_out_of_window_and_keeps_binary(tmp_path):
    boxes = [[1, 10, 10, 13, 13], [35, 20, 20, 23, 23]]  # 第二框在窗外
    write_fake_cache(tmp_path, boxes=boxes, trans=ident_trans(N_F))
    m = gt_masks_anchor(np.array(boxes, np.int32), 0, 32, None, (H_F, W_F))
    assert m.shape == (32, H_F, W_F)
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert m[0].sum() == 16
    assert m.sum() == 16, "窗外帧号不得混进掩码（只该有首帧的框）"
    assert m[1:].sum() == 0.0


# --------------------------------------------------------------------------- #
# 2) §4.7 裁剪保完整约束
# --------------------------------------------------------------------------- #
def test_crop_prefers_all_tracks_then_one_track(tmp_path):
    """轨迹可行集决定档位：贴全幅的长轨迹永远装不下 → one_track；两条都装得下 → all_tracks。"""
    trans = ident_trans(N_F)
    boxes = [
        [1, 40, 0, 45, 95],    # tid 1：纵贯全幅的长轨迹（任何 48×64 裁剪都裁半）→ 不可行
        [10, 40, 0, 45, 95],
        [1, 60, 40, 69, 49],   # tid 2：可行目标
        [10, 60, 40, 69, 49],
    ]
    write_fake_cache(tmp_path, boxes=boxes, trans=trans, track_ids=[1, 1, 2, 2])
    ds = ds_from_fake(tmp_path, T=16, stride=8, crop=(48, 64), feature_stride=1)
    it = ds[0]
    assert it["crop_tier"] == "one_track", f"应退化到单轨迹完整：{it['crop_tier']}"
    y0, x0 = it["crop_offset"]
    # tid2 框（行 40..49、列 60..69）加 3px 边距须完整落入裁剪区——解析真值核对
    assert y0 <= 40 - 3 and y0 + 48 >= 50 + 3, f"被选轨迹行向不完整：y0={y0}"
    assert x0 <= 60 - 3 and x0 + 64 >= 70 + 3, f"被选轨迹列向不完整：x0={x0}"
    assert it["gt_clipped"] is True, "纵贯全幅的轨迹必然被裁半，须如实标记"
    # 独立复核（连通域口径）：裁剪区里确实存在含边距的完整框
    m = gt_masks_anchor(np.array(boxes, np.int32), it["start0"], 16, None, (H_F, W_F))
    assert has_complete_box_with_margin(m[:, y0:y0 + 48, x0:x0 + 64], 3)

    # 两条都可容纳且可行区间有交 → all_tracks
    boxes2 = [[1, 30, 30, 39, 39], [1, 45, 20, 54, 29]]
    root2 = tmp_path / "both"
    write_fake_cache(root2, boxes=boxes2, trans=trans, track_ids=[1, 2])
    ds2 = ds_from_fake(root2, T=16, stride=8, crop=(48, 64), feature_stride=1)
    it2 = ds2[0]
    assert it2["crop_tier"] == "all_tracks", it2["crop_tier"]
    assert it2["gt_clipped"] is False
    y2, x2 = it2["crop_offset"]
    for (bx1, by1, bx2, by2) in ((30, 30, 39, 39), (45, 20, 54, 29)):
        assert y2 <= by1 - 3 and y2 + 48 >= by2 + 4, f"框行向未完整：{by1},{y2}"
        assert x2 <= bx1 - 3 and x2 + 64 >= bx2 + 4, f"框列向未完整：{bx1},{x2}"


def test_crop_fallback_triggers_window_resample(tmp_path):
    """全窗无可行框 → tier=fallback，__getitem__ 必须换窗重采（§4.7"否则重采"）。"""
    trans = ident_trans(N_F)
    # 基窗 [0,32) 只有两条贴画面边的框（0..5 与 120..127）：任何偏移都给不出 3px 边距
    boxes = [[1, 0, 0, 5, 5], [1, 120, 90, 127, 95], [100, 60, 50, 69, 59]]
    write_fake_cache(tmp_path, boxes=boxes, trans=trans, track_ids=[1, 1, 2])
    ds = ds_from_fake(tmp_path, T=32, stride=8, crop=(48, 64), feature_stride=1,
                     max_crop_attempts=8)
    it = ds[0]  # 首窗不可行 → 记账 fallback 后换窗
    assert ds.crop_stats["fallback"] >= 1, "未触发 fallback 计数，重采路径没被走到"
    assert it["crop_tier"] != "fallback", "重采后仍交回不可行窗"
    assert it["crop_tier"] in ("all_tracks", "one_track", "one_frame", "negative")
    assert it["seq_id"] == 1 and 0 <= it["start0"] <= N_F - 32


def test_crop_never_returns_fallback_on_real_cache():
    """真实 train-int 缓存：取样永不交回 fallback 窗，且 §4.7"≥1 完整框含边距"独立复核。

    分档计数同时是 P0 报告的数据：横跨全幅的长轨迹使 all_tracks 不可能恒成立，
    one_track/one_frame 是真实主档（退化如实记账，不假装约束总是最强）。
    """
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", T=32, stride=8,
                     crop=(240, 320), augment=False, limit_seqs=4, seed=1,
                     start_jitter=False)
    tiers, clipped, complete = [], [], []
    for i in range(24):
        it = ds[i]
        tiers.append(it["crop_tier"])
        clipped.append(it["gt_clipped"])
        assert it["windows"].shape[-2:] == (240, 320)
        assert it["crop_tier"] != "fallback", f"idx={i} 交回了不可行窗"
        if it["crop_tier"] != "negative":
            # 复核判据（连通域口径，独立于数据集的区间算术）：GT 原生掩码须用
            # 同一批 Ws 重算，才能与 crop_offset 描述的坐标系对齐（L4 同治）。
            seq = ds._load_seq(it["seq_id"])
            Ws = window_anchor_warps(seq["reg"], it["start0"], ds.T)
            m = gt_masks_anchor(seq["boxes"], it["start0"], ds.T, Ws,
                                seq["frames"].shape[1:3])
            y0, x0 = it["crop_offset"]
            ch, cw = ds.crop
            complete.append(
                has_complete_box_with_margin(m[:, y0:y0 + ch, x0:x0 + cw], ds.crop_margin))
    assert ds.crop_stats["attempts"] >= 24, "重采计数异常"
    # fallback 允许 >0：它是"该窗确实装不下任何框→换窗"的记账，返回的样本必须已换好
    assert all(complete), (
        f"{sum(1 for c in complete if not c)}/{len(complete)} 个窗未满足"
        f"'≥1 完整 GT 框含 {ds.crop_margin}px 边距'")
    dist = {t: tiers.count(t) for t in set(tiers)}
    print(f"[P0 裁剪分档] {dist}  GT 贴边率={float(np.mean(clipped)):.2f}  "
          f"计数={ds.crop_stats}")


# --------------------------------------------------------------------------- #
# 3) 监督张量几何（与 L_recon 同几何）
# --------------------------------------------------------------------------- #
def test_teacher_mask_equals_losses_maxpool_path():
    """teacher_mask ≡ torch 侧 F.max_pool2d(膨胀3px)→maxpool(2)（跨框架同几何证明）。"""
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(11)
    T, h, w = 5, 40, 56
    tgt = np.zeros((T, 1, h, w), np.float32)
    for t in range(T):
        y, x = rng.integers(2, h - 8), rng.integers(2, w - 8)
        sz = int(rng.integers(1, 3))  # 含 1px 级小目标
        tgt[t, 0, y:y + sz, x:x + sz] = 1.0
    np_path = downsample_max(dilate_mask(tgt, 3), 2)
    x = torch.from_numpy(tgt)
    torch_path = F.max_pool2d(
        F.max_pool2d(x, 2 * 3 + 1, stride=1, padding=3), 2, 2).numpy()
    assert np.array_equal(np_path, torch_path), "numpy/torch 膨胀+下采样口径分叉"
    assert (np_path >= (tgt[:, :, ::2, ::2] > 0)).all(), "teacher 必须包含 target 全集"


def test_output_tensors_domain_shapes_and_binary(tmp_path):
    n = N_F
    boxes = [[f + 1, 40, 30, 47, 37] for f in range(4, 14)]
    write_fake_cache(tmp_path, boxes=boxes, trans=ident_trans(n))
    ds = ds_from_fake(tmp_path, T=16, stride=8, crop=(48, 64), feature_stride=2)
    it = ds[0]
    assert it["windows"].shape == (16, 1, 48, 64)
    assert it["windows"].dtype == np.float32
    assert it["target"].shape == (16, 1, 24, 32)
    assert it["teacher_mask"].shape == (16, 1, 24, 32)
    assert it["windows"].min() >= 0.0 and it["windows"].max() <= 1.0
    assert set(np.unique(it["target"])) <= {0.0, 1.0}
    assert float(it["target"].sum()) > 0 and float(it["teacher_mask"].sum()) > float(
        it["target"].sum())
    assert it["quality"] == pytest.approx(
        float(np.median(np.load(tmp_path / "seq_0001" / "quality.npy")[
            it["start0"]:it["start0"] + 16]))), \
        "quality 必须取窗口中位数（与 IttdWindows 训练/快评同口径）"


def test_small_target_survives_downsample(tmp_path):
    """5px 级目标（原生 2×2 / 1×1）在 stride-2 图上不得消失——小目标是验收对象。"""
    for sz, expect_min in ((2, 1), (1, 1)):
        x0, y0 = 41, 25
        boxes = [[f + 1, x0, y0, x0 + sz - 1, y0 + sz - 1] for f in range(4, 10)]
        root = tmp_path / f"s{sz}"
        write_fake_cache(root, boxes=boxes, trans=ident_trans(N_F))
        ds = ds_from_fake(root, T=16, stride=8, crop=(48, 64), feature_stride=2)
        it = ds[0]
        s0 = it["start0"]
        hits = [float(it["target"][f - 1 - s0, 0].sum()) for f in range(5, 11)]
        assert all(h >= expect_min for h in hits), (
            f"{sz}px 目标在下采样中丢失：{hits}")
        assert all(h <= 4 for h in hits), f"{sz}px 目标被放大成 {hits}（几何失真）"


# --------------------------------------------------------------------------- #
# 4) 合法负样本 / R4 门限 / 确定性 / track_index
# --------------------------------------------------------------------------- #
def test_empty_window_is_legal_negative_not_dropped(tmp_path):
    """无标注窗按正常样本产出（全 0 掩码），禁止当损坏剔除（§2.1/10.9）。"""
    boxes = [[200, 40, 30, 47, 37]]  # 只出现在窗 [0,32) 之外
    write_fake_cache(tmp_path, boxes=boxes, trans=ident_trans(N_F))
    ds = ds_from_fake(tmp_path, T=32, stride=8, crop=(48, 64))
    assert len(ds) > 0, "全部窗被剔除——负样本窗不得丢弃"
    it = ds[0]
    assert it["crop_tier"] == "negative" and float(it["target"].sum()) == 0.0
    assert it["gt_clipped"] is False


def test_reg_failed_density_windows_dropped(tmp_path):
    """R4：reg_failed 密度 >10% 的窗在索引阶段剔除，其余窗保留。"""
    failed = np.zeros(N_F, bool)
    failed[0:32] = True  # 与基窗 0/8/16/24 的重叠比例 = 100/75/50/25% 均 >10%
    boxes = [[50, 40, 30, 47, 37]]
    write_fake_cache(tmp_path, boxes=boxes, trans=ident_trans(N_F), failed=failed)
    ds = ds_from_fake(tmp_path, T=32, stride=8, max_reg_fail_frac=0.10)
    starts = {s for _sid, s in ds.index}
    for s in (0, 8, 16, 24):
        assert s not in starts, f"配准失败密度超限的基窗 {s} 未被剔除（鬼影簇毒化慢通道）"
    assert (len(ds), ds.n_dropped_reg) == (28 - 4, 4), (
        f"应剔 4 个窗，实得 len={len(ds)} dropped={ds.n_dropped_reg}")
    # 门限放宽后同一批窗必须回来（证明剔除依据确是 failed 而非别的原因）
    ds_loose = ds_from_fake(tmp_path, T=32, stride=8, max_reg_fail_frac=1.0)
    assert len(ds_loose) == 28 and ds_loose.n_dropped_reg == 0


def test_determinism_per_seed(tmp_path):
    boxes = [[f + 1, 40, 30, 47, 37] for f in range(20, 30)]
    write_fake_cache(tmp_path, boxes=boxes, trans=ident_trans(N_F))
    kw = dict(T=16, stride=8, crop=(48, 64), augment=True, feature_stride=2)
    a = ds_from_fake(tmp_path, seed=3, **kw)[5]
    b = ds_from_fake(tmp_path, seed=3, **kw)[5]
    c = ds_from_fake(tmp_path, seed=4, **kw)[5]
    assert np.array_equal(a["windows"], b["windows"]) and a["start0"] == b["start0"]
    assert np.array_equal(a["target"], b["target"])
    assert not np.array_equal(a["windows"], c["windows"]), "换 seed 应换随机实现"


def test_window_info_uses_track_index(tmp_path):
    """track_index 消费：窗内轨迹数/框数与 labels 一致（裁剪可行性据此判定）。"""
    boxes = [[1, 30, 30, 39, 39], [2, 30, 30, 39, 39], [20, 45, 20, 54, 29]]
    write_fake_cache(tmp_path, boxes=boxes, trans=ident_trans(N_F), track_ids=[1, 1, 2])
    ds = ds_from_fake(tmp_path, T=32, stride=8, crop=None, augment=False)
    info = ds.window_info(0)
    assert info["seq_id"] == 1 and info["start"] == 0
    assert info["n_tracks"] == 2, f"窗内应有 2 条轨迹，实得 {info['n_tracks']}"
    assert info["n_gt_boxes"] == 3
    # 窗外框不计入（帧 200 的框属于别的窗）
    boxes_out = boxes + [[200, 30, 30, 39, 39]]
    root2 = tmp_path / "out"
    write_fake_cache(root2, boxes=boxes_out, trans=ident_trans(N_F),
                     track_ids=[1, 1, 2, 3])
    ds2 = ds_from_fake(root2, T=32, stride=8, crop=None, augment=False)
    assert ds2.window_info(0)["n_tracks"] == 2, "窗外轨迹漏进了窗统计"


# --------------------------------------------------------------------------- #
# 5) 合成注入（§2.6 特色项）
# --------------------------------------------------------------------------- #
def test_highlight_injection_amplitude_life_and_gt_exclusion():
    """瞬时亮点：幅值 ∈[μ+4σ, μ+8σ] 的增量、存活 1–3 帧、不污染 GT 膨胀区。"""
    T, H, W = 32, 96, 128
    base = np.full((T, H, W), 0.5, np.float32)  # 归一化域背景 = 中位
    mask = np.zeros((T, H, W), np.float32)
    mask[:, 10:16, 10:16] = 1.0                 # GT 轨迹固定区
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", T=T, stride=8, crop=None,
                     augment=False, align=False, limit_seqs=1, highlight=True,
                     hotspot=False)
    for seed in range(40):  # p=0.4/窗，抽样若干次直到真的注入
        diff = ds._inject(base.copy(), mask, np.random.default_rng(seed)) - base
        if diff.max() > 0:
            break
    else:
        pytest.fail("40 次抽样未注入亮点（p=0.4 不应如此）")
    assert diff.min() >= 0.0, "注入只做增量，不得削弱背景"
    peak = float(diff.max())
    assert _level(4.0) - 0.5 - 1e-6 <= peak <= _level(8.0) - 0.5 + 1e-6, (
        f"增量峰值 {peak:.3f} 不在 [μ+4σ, μ+8σ] = [{_level(4.0) - 0.5:.3f},"
        f"{_level(8.0) - 0.5:.3f}]")
    # GT 不被污染：真目标膨胀(3px)区内增量必须可忽略
    core = dilate_mask(mask.max(axis=0), ds.teacher_dilate) > 0
    assert float(diff[:, core].max()) < 0.01, (
        f"GT 膨胀区被注入污染 {float(diff[:, core].max()):.4f}")
    life = int((diff.max(axis=(1, 2)) > 1e-6).sum())
    assert 1 <= life <= 3, f"存活帧数 {life} 违反 1–3 帧约定"
    # 注入必须落在无 GT 的位置（否则同一像素既是正例又是负例，监督自相矛盾）
    _tt, ys, xs = np.unravel_index(np.argmax(diff), diff.shape)
    assert not mask[:, ys, xs].any(), "注入落在有 GT 的行/列轨迹上"


def test_hotspot_rises_slowly_and_stays():
    """缓变热斑：30 帧内线性升至 μ+5σ 增量后维持，单调不降（慢通道剥离训练的信号）。"""
    rng = np.random.default_rng(2)
    T, H, W = 32, 96, 128
    base = np.full((T, H, W), 0.5, np.float32)
    mask = np.zeros((T, H, W), np.float32)
    mask[:, 0:4, 0:4] = 1.0
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", T=T, stride=8, crop=None,
                     augment=False, align=False, limit_seqs=1, highlight=False,
                     hotspot=True, hotspot_p=1.0)
    out = ds._inject(base.copy(), mask, rng)
    prof = (out - base).max(axis=(1, 2))
    assert prof.max() > 0, "hotspot_p=1 仍未注入热斑"
    rise = prof[prof > 0]
    assert np.all(np.diff(prof) >= -1e-6), f"热斑剖面非单调：{prof}"
    assert prof[-1] == pytest.approx(float(rise.max()), rel=0.02), "升温后应维持平台"
    assert prof.max() <= _level(5.0) - 0.5 + 1e-6


def test_frame_copy_freezes_window_and_mask_together():
    """帧复制：图像与掩码同帧替换（只冻图像不冻掩码 = 监督与像素矛盾）。"""
    ds = IttdWindows(MANIFEST, CACHE, split="train-int", T=16, stride=8, crop=None,
                     augment=False, align=False, limit_seqs=1)
    win = np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8)
    mask = np.zeros((16, 8, 8), np.float32)
    mask[:] = np.arange(16)[:, None, None]
    for attempt in range(200):
        r = np.random.default_rng([1000 + attempt, 0])
        w2, m2 = ds._frame_copy(win.copy(), mask.copy(), r)
        rows = np.nonzero(~(w2 == win).all(axis=(1, 2)))[0]
        if len(rows) == 0:
            continue
        for t in rows:
            assert np.array_equal(w2[t], w2[t - 1]), f"帧 {t} 未复制 {t - 1} 图像"
            assert np.array_equal(m2[t], m2[t - 1]), f"帧 {t} 掩码未与图像同治"
        untouched = [t for t in range(16) if t not in set(rows.tolist())]
        assert np.array_equal(m2[untouched], mask[untouched]), "未选中帧被改动"
        assert rows.min() >= 1, "帧 0 无前帧可复制"
        return
    pytest.fail("200 次抽样未触发帧复制（p=0.3 不应如此）")


def test_real_cache_end_to_end_batch(tmp_path):
    """真实缓存 4 段端到端：DataLoader 成批可跑（默认 collate 处理 numpy/str/bool）。"""
    from torch.utils.data import DataLoader

    ds = IttdWindows(MANIFEST, CACHE, split="val-int", T=32, stride=24,
                     crop=(240, 320), augment=True, limit_seqs=2, seed=7)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    b = next(iter(dl))
    assert tuple(b["windows"].shape) == (2, 32, 1, 240, 320)
    assert tuple(b["target"].shape) == (2, 32, 1, 120, 160)
    assert b["windows"].dtype.is_floating_point
    assert isinstance(b["scene"], list) and len(b["scene"]) == 2
    assert b["seq_id"].tolist()[0] in ds.manifest["splits"]["val-int"]["seqs"]


# --------------------------------------------------------------------------- #
# 6) Δt 口径（容器补帧 → 实采 ≈33⅓ Hz）
# --------------------------------------------------------------------------- #
def test_slot_gaps_synthetic_padding_pattern(tmp_path):
    """显式"2 实采 + 1 补帧"流（周期-3，与真实缓存同构）：dt 必须逐槽精确回收。"""
    n = N_F
    # 补帧槽 = t % 3 == 0（t>0）：内容编号 value(t) = t − t//3
    frames = np.stack([np.full((24, 32), (t - t // 3) % 250, np.uint8) for t in range(n)])
    write_fake_cache(tmp_path, boxes=[[1, 4, 4, 7, 7]], trans=ident_trans(n),
                     frames=frames, h=24, w=32)
    ds = ds_from_fake(tmp_path, T=16, stride=8, crop=None)
    got = ds._load_seq(1)["dt"]
    want = [1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0]
    assert got[:12].tolist() == want, f"dt 回收错：{got[:12].tolist()}"
    assert float((got[1:] == 0).mean()) == pytest.approx(1 / 3, abs=0.01)
    # 守恒式：Σ_{s≤t} dt = ≤t 的最后一个新曝光槽号 + 1（补帧槽不吞时间也不 duplicated 计）
    for t in (1, 3, 4, 6, 9, 12, 60, 249):
        last_new = max(s for s in range(t + 1) if got[s] > 0)
        assert abs(float(got[:t + 1].sum()) - (last_new + 1)) < 1e-6, f"t={t} 时间不守恒"


def test_dt_real_cache_padding_is_one_third_and_zero_iff_duplicate():
    """真实缓存：dt=0 当且仅当字节级重复；值域 ⊆{0,1,2}；补帧占比 ≈1/3（周期-3 格点）。"""
    ds = IttdWindows(MANIFEST, CACHE, split="val-int", T=32, stride=8, crop=None,
                     augment=False, align=False, limit_seqs=0, seed=0)
    sid = 21
    dt = ds._load_seq(sid)["dt"]
    fr = np.asarray(ds._load_seq(sid)["frames"])
    dup = np.array([np.array_equal(fr[t], fr[t - 1]) for t in range(1, len(fr))])
    assert np.array_equal(dt[1:] == 0, dup), "dt=0 与字节级重复帧不一一对应"
    assert set(np.unique(dt).tolist()) <= {0.0, 1.0, 2.0}, f"dt 值域异常 {np.unique(dt)}"
    assert abs(float((dt[1:] == 0).mean()) - 1 / 3) < 0.02, "补帧占比偏离 1/3 周期-3 结构"
    it = ds[ds.index.index((sid, 0))]
    assert it["dt"].shape == (32,) and it["dt"].dtype == np.float32
    assert float(it["dt"].sum()) > 20, "32 槽窗的时间预算被吞掉（去重/补帧处理错误）"


def test_augmented_frame_copy_does_not_rewrite_dt(tmp_path):
    """增强造成的"重复帧"不是传感器补帧：dt 保持实测值（物理时间照样流逝）。"""
    frames = np.stack([np.full((24, 32), 10 + (t % 5), np.uint8) for t in range(N_F)])
    write_fake_cache(tmp_path, boxes=[[1, 4, 4, 7, 7]], trans=ident_trans(N_F),
                     frames=frames, h=24, w=32)
    ds = ds_from_fake(tmp_path, T=16, stride=8, crop=None, augment=True,
                      feature_stride=1, seed=1)
    dt_before = ds._load_seq(1)["dt"].copy()
    assert set(dt_before.tolist()) == {1.0}, "该合成流不应含补帧槽"
    it = ds[0]
    assert np.array_equal(it["dt"], dt_before[:16]), "dt 被增强改写了（应只反映传感器曝光）"
    assert float(it["dt"].sum()) > 0
