"""DSLD 四分量损失（M3 v2.0 方案 §5.1）+ 锚点系残差诊断（§4.3.5）。

    L = 1.0·L_seg + 0.5·L_recon + 0.3·L_gate + 0.1·L_dec

几何口径（L4 纪律，全部 stride-2、全部锚点坐标系）：
  - `target`      = GT 框填充 → 块最大下采样（IttdWindows 产出）→ L_seg / L_gate 的正例；
  - `teacher_mask`= 同一框先膨胀 3px 再块最大下采样 → 内环掩码（teacher 源）与 L_recon
                    剔除区的**同一张图**；
  - 本文件**不做任何膨胀/下采样**：需要几个像素的排除区，就消费数据侧给的那张图。
    前版在损失内部又写一遍 maxpool 膨胀，与装配层那份迟早分叉（坐标系、核大小、
    端点约定三处都可能漂），故整体移除。

值域与不变量（各有可失败单测，见 tests/test_losses.py）：
  - focal ≥ 0、soft-Dice ∈ [0,1]；空 GT 帧的 Dice 项退化为对预测前景质量的软惩罚
    （虚警在空帧上被计价，见 focal_dice_loss docstring），全背景窗是合法负样本
    （总方案 10.9），不许当损坏剔除；
  - L_recon ≥ 0，附带 `valid_frac` 返回值（不塞函数属性：前版用 `recon_loss.last_valid_frac`
    是隐式全局态，多卡/多次调用即串味）；valid_frac < min_valid_frac ⇒ 背景监督面塌缩，
    strict 抛错（§5.5 窒息哨兵的运行期版本）；
  - L_gate = 正项均值 + 负项均值：**不变量 α≡0.5 ⇒ L ≡ 2·ln2**，与框大小、正例数量无关。
    朴素 BCE 会被稀少正例的背景项淹没、把 α 推成全局 0（= 门控死锁的成因，5k 诊断实录）；
  - L_dec = 位置级 cos² + 范数下限，h≡0 坍缩必给正损失（旧式 cos² 在双零时抹平为 0，
    恰好放过它本该防止的坍缩）；single 臂 h_b≡h_t ⇒ cos²≡1 常数，两臂损失组成一致，
    F_a 差异全归因结构分立。
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

_EPS = 1e-6


def focal_dice_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75,
                    gamma: float = 2.0) -> torch.Tensor:
    """L_seg（§5.1）：Focal(α=0.75, γ=2) 与 soft-Dice 各半，逐像素 logits vs 框填充掩码。

    logits/target: [B,T,1,h,w]（时间维展平后逐帧算，Dice 逐样本 = 逐帧）。logits 未过
    sigmoid——监督直接施加在 head([α⊙r, h_T]) 这条物理抑制通路上（总方案 7.1①）。

    空 GT 帧（全背景窗，合法负样本）：Dice 项不是"无监督"，而是退化成对预测前景质量的
    软惩罚 1 − 1/(1+Σpred) ≈ Σpred（smooth=1）——虚警在空帧上被直接计价，比方案 §5.1
    原写的"约定 0"更强；Focal 的负例项照常参与。只有"预测与 GT 同时为空"才严格 0。
    """
    logits = logits.float()
    target = target.float()
    logits, target = logits.flatten(0, 1), target.flatten(0, 1)
    pred = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = target * pred + (1.0 - target) * (1.0 - pred)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    focal = (alpha_t * (1.0 - p_t) ** gamma * bce).mean()
    inter = (pred * target).sum(dim=(1, 2, 3))
    card = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = torch.where(card > 0, 1.0 - (2.0 * inter + 1.0) / (card + 1.0),
                       torch.zeros_like(card))
    return 0.5 * focal + 0.5 * dice.mean()


def recon_loss(y_b: torch.Tensor, x_feat: torch.Tensor, teacher_mask: torch.Tensor,
               m_tgt: torch.Tensor, top_ratio: float = 0.05, min_valid_frac: float = 0.5,
               strict: bool = True) -> tuple[torch.Tensor, float]:
    """L_recon（§5.1）：L1(ŷ_B, x) 只在背景区算，返回 (loss, valid_frac)。

    排除三类（防背景吸收目标/亮点）：① teacher_mask（GT 膨胀区，与内环掩码同几何）；
    ② 当帧生效的内环掩码 m_tgt（gate 臂下即自举保护区）；③ 其余区域残差 |x−ŷ_B| 的
    top-`top_ratio`（未标注的瞬时亮点——总方案 7.1②"最后一项是关键"）。
    残差幅值按通道均值排序，剔除名额只在候选背景区内竞争。

    valid_frac < min_valid_frac（§5.5 口径 0.5）= 背景监督面塌缩：strict 抛错立即定位
    （与"loss NaN 即停"同策略，不烧 GPU 追一个已无监督意义的梯度），否则只告警。
    """
    y_b = y_b.float()
    x_feat = x_feat.float()
    B, T = y_b.shape[:2]
    excl = ((teacher_mask > 0.5) | (m_tgt > 0.0)).float() if teacher_mask is not None \
        else (m_tgt > 0.0).float()
    valid = 1.0 - excl
    res_mag = (x_feat - y_b).abs().mean(dim=2, keepdim=True)          # [B,T,1,H,W]
    v_flat = valid.flatten(2)
    r_flat = res_mag.flatten(2).masked_fill(v_flat == 0, float("inf"))
    k = int(v_flat.shape[-1] * top_ratio)
    if k > 0:
        thr = r_flat.topk(k, dim=-1).values[..., -1:]                 # [B,T,1]
        # `<=`：并列值（平场残差全零）不整片剔除——top-k 是软守卫不是精确截断
        v_flat = v_flat * (r_flat <= thr)
    valid = v_flat.view(B, T, 1, *y_b.shape[-2:])
    valid_frac = float(valid.mean())
    if valid_frac < min_valid_frac:
        msg = (f"L_recon 有效像素占比 {valid_frac:.4f} < {min_valid_frac}——掩码疑似铺满"
               "全图（背景通道窒息），本损失已失去监督意义")
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    denom = (valid.sum() * y_b.shape[2]).clamp_min(1.0)
    return ((x_feat - y_b).abs() * valid).sum() / denom, valid_frac


def gate_bce_loss(alpha_map: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """L_gate（§4.3.3 / §5.1）：平衡 BCE(α, GT 框掩码) = 正项均值 + 负项均值。

    alpha_map [B,T,1,h,w] post-sigmoid；gt_mask [B,T,1,h,w]（0/1，框填充下采样，**不膨胀**
    ——膨胀属 teacher/剔除区口径，门控要学的是"目标像素本身"）。空标注帧只计背景项
    （α 被推到 0 是正确监督，不是退化）。不变量：α≡c ⇒ L = ln(1/(1−c)) + ln(1/c)，
    c=0.5 时精确 2·ln2，与框大小无关。
    """
    a = alpha_map.float().clamp(1e-4, 1.0 - 1e-4)
    pos = a[gt_mask > 0.5]
    neg = a[gt_mask <= 0.5]
    loss_pos = (-torch.log(pos)).mean() if pos.numel() else a.new_zeros(())
    loss_neg = (-torch.log(1.0 - neg)).mean() if neg.numel() else a.new_zeros(())
    return loss_pos + loss_neg


def decouple_loss(h_t: torch.Tensor, h_b: torch.Tensor, min_norm: float = 1e-2,
                  norm_weight: float = 1.0) -> torch.Tensor:
    """L_dec（§5.1）：逐位置 cos²(h_T,h_B) + 范数下限项（坍缩哨兵）。

    h_t/h_b [B,T,C,H,W]，按通道维求内积/范数（逐帧逐位置）。分母 clamp_min(min_norm²)
    让"双零"不再把 cos² 抹平成 0；范数下限项保证 h≡0 必给正损失。single 臂 h_b 与 h_t 是
    同一张量 ⇒ cos²≡1（常数），损失组成与 dual 臂一致。
    分母再套一层 1e-6 数值保险：min_norm=0 时（诊断/消融用法）保护冻结区 0/0 会变 NaN。
    """
    h_t = h_t.float()
    h_b = h_b.float()
    dot = (h_t * h_b).sum(dim=2)                                      # [B,T,H,W]
    nt = h_t.norm(dim=2)
    nb = h_b.norm(dim=2)
    denom = (nt * nb).clamp_min(min_norm * min_norm).clamp_min(1e-6)
    cos2 = (dot / denom) ** 2
    cos2 = cos2.clamp(max=1.0)
    norm_floor = F.relu(min_norm - torch.minimum(nt, nb)).mean()
    return cos2.mean() + norm_weight * norm_floor


@torch.no_grad()
def resid_diagnostics(r: torch.Tensor, gt_mask: torch.Tensor,
                      teacher_mask: torch.Tensor | None = None,
                      m_tgt: torch.Tensor | None = None) -> dict[str, float]:
    """锚点系残差诊断（§4.3.5）：resid_scr（目标残差 SNR）与 bg_resid_rms。

    resid_scr = GT 处 |r| 均值 / 背景区 |r| 均值（背景区 = GT ∪ teacher ∪ m_tgt 之外，与
    L_recon 同一套排除，保证"判据 ② 的分子分母与训练目标同域"）；
    bg_resid_rms = 背景区残差 RMS（判据 ③ learned vs EMA 的度量）。无 GT/无前景时置 NaN
    而非 0——0 会被误读成"完美"，NaN 让快评表格里一眼可见（L2 不可能失败指标纪律）。
    """
    mag = r.float().abs().mean(dim=2, keepdim=True)                   # [B,T,1,H,W]
    gt = gt_mask > 0.5
    bg = ~gt
    if teacher_mask is not None:
        bg &= ~(teacher_mask > 0.5)
    if m_tgt is not None:
        bg &= ~(m_tgt > 0.0)
    out: dict[str, float] = {}
    if gt.any():
        scr = mag[gt].mean() / (mag[bg].mean() + _EPS) if bg.any() else float("nan")
        out["resid_scr"] = round(float(scr), 4)
    else:
        out["resid_scr"] = float("nan")
    out["bg_resid_rms"] = round(float(mag[bg].mean()), 6) if bg.any() else float("nan")
    return out


def total_loss(out: dict, batch: dict, weights: dict, recon_valid_min: float = 0.5,
               strict: bool = True) -> dict:
    """四分量组装（§5.1 权重）+ 哨兵/诊断量。返回 {loss, seg, recon, gate, dec, 诊断}。

    batch 需含 `target` 与 `teacher_mask`（均 stride-2、锚点系，IttdWindows 产出）。
    weights 为 {seg, recon, gate, dec}，缺项即报错（防静默用默认权重跑出不一致的课程）。
    recon_valid_min 来自 §5.5 哨兵档（train.sentinels.recon_valid_min），与损失权重分开传。
    """
    required = ("seg", "recon", "gate", "dec")
    for k in required:
        if k not in weights:
            raise KeyError(f"total_loss 缺权重 {k!r}（须显式给出，可选 {required}）")
    tgt = batch["target"]
    teacher = batch.get("teacher_mask")
    seg = focal_dice_loss(out["logits"], tgt)
    recon, valid_frac = recon_loss(out["y_b"], out["x_main"], teacher, out["m_tgt"],
                                   min_valid_frac=recon_valid_min, strict=strict)
    gate = gate_bce_loss(out["alpha"], tgt)
    dec = decouple_loss(out["h_t"], out["h_b"])
    loss = (float(weights["seg"]) * seg + float(weights["recon"]) * recon
            + float(weights["gate"]) * gate + float(weights["dec"]) * dec)
    diag = resid_diagnostics(out["r"], tgt, teacher, out["m_tgt"])
    diag.update(out.get("norms", {}))
    diag["recon_valid_frac"] = round(valid_frac, 4)
    return {"loss": loss, "seg": seg, "recon": recon, "gate": gate, "dec": dec,
            "diag": diag, "tau": out.get("tau", {})}
