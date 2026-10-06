"""双状态液态核心（M3，方案 4.2–4.8：CfC 闭式更新 + τ 硬约束 + 保持项 + FiLM + 内环）。

设计要点（对照方案条文）：
  - CfC 更新（4.2）：p = σ(−f·Δt/τ − β·α_prev)，h' = p⊙g + (1−p)⊙m；
    τ 在 log 域硬 clip 到 [τ_min, τ_max]（背景 [16,256] init 48，目标 [2,16] init 6），
    双状态时间尺度分立假设由结构保证（消融 b 去约束）。
  - 逐位置算子（4.6）：f/g/m 全部 1×1 conv（[h, x_in] 拼接输入），空间全并行、
    时间维 Python 串行；输入前 3×3 depthwise conv 提供邻域上下文。
  - 级联（4.3/4.4）：x_B = x⊙(1−M)（内环掩码清零）→ h_B → ŷ_B →
    x_T = (x − ŷ_B)⊙(1−M) → h_T。目标通道从出生只接触背景解释残差。
  - 保持项（4.4）：仅目标通道，p_T 额外减 β·α_prev，β = softplus(θ_β) ≥ 0，
    初始化 β≈2（gate 偏移 −2，保持时长约 10 帧）。
  - 场景条件化（4.3 外环）：全帧统计 s(t)（均值/方差/亮斑密度/帧序/清晰度）→
    FiLM 线性映射 (γ_s, β_s) 仿射调制 f/g/m 各头输出；γβ 零初始化=恒等起步。
  - 反馈掩码（4.5）：M(t) = max(α 峰值膨胀 r=5px, 0.9·M(t−1))——目标经过处背景
    通道被打码约 10 帧衰减期。
  - 数值稳定（4.8）：核心全程 fp32（autocast 关闭），h 零初始化，隐状态范数监控。

单状态消融（方案 10.8-a）：liquid.mode="single"——单条液态流（无背景/目标结构
分立、无级联），τ 范围取并集 [2,256]（让数据自己决定每单元时间尺度，正是结构
先验假设的对照组），其余（dw/FiLM/门控头/分割头）逐一同构。
临时门控 α（M3 占位）：α = σ(conv1x1([x−ŷ_B, h_T]))——M4 换成三重一致性门控
（z_B, r_T, LR），数据流（α→分割头输入加权 + 反馈掩码）保持不变。

语义约定（与 3.2 表 M / 7.1① 的口径互洽）：ŷ_B 为 32ch 特征域重构（bg_head
c_h→c_in），分割头输入 [α⊙残差, h_T] = 64ch 对应表 M"64→1"；表 M"背景预测头
32→1"按"1×1 conv、1 倍尺度"解读（7.1① 的 64→1 只有在残差为 32ch 时才成立）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _LiquidChannel(nn.Module):
    """单条液态通道：f/g/m 1×1 头 + log 域 τ 硬约束 + 可选保持项 + 逐头 FiLM。"""

    def __init__(self, c_in: int, c_h: int, tau_min: float, tau_max: float,
                 tau_init: float, keep: bool = False, scene_dim: int = 5):
        super().__init__()
        self.f_head = nn.Conv2d(c_in + c_h, c_h, 1)
        self.g_head = nn.Conv2d(c_in + c_h, c_h, 1)
        self.m_head = nn.Conv2d(c_in + c_h, c_h, 1)
        # τ 参数化（4.2）：θ_τ 逐单元向量（通道内共享空间），log 域硬 clip
        self.theta_tau = nn.Parameter(torch.full((c_h,), math.log(tau_init)))
        self.log_tau_min = math.log(tau_min)
        self.log_tau_max = math.log(tau_max)
        # FiLM（4.3）：s(t) → 逐头 (γ_s, β_s)，零初始化 = 恒等起步
        self.film = nn.ModuleList(
            [nn.Linear(scene_dim, 2 * c_h) for _ in range(3)])
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # 保持项（4.4）：仅目标通道；β=softplus(θ_β)，init β≈2 → θ=log(e²−1)
        self.keep = keep
        if keep:
            self.theta_beta = nn.Parameter(torch.tensor(math.log(math.expm1(2.0))))

    def tau(self) -> torch.Tensor:
        """有效时间常数 τ（帧）——log 域硬约束后的只读视图。"""
        return torch.exp(self.theta_tau.clamp(self.log_tau_min, self.log_tau_max))

    def step(self, h: torch.Tensor, x_in: torch.Tensor, scene: torch.Tensor,
             alpha_prev: torch.Tensor | None, dt: float = 1.0) -> torch.Tensor:
        """单步 CfC 更新。h/x_in: [B,C,H,W]；scene: [B,5]；alpha_prev: [B,1,H,W]。"""
        inp = torch.cat([h, x_in], dim=1)
        outs = []
        for i, head in enumerate((self.f_head, self.g_head, self.m_head)):
            o = head(inp)
            gamma, beta = self.film[i](scene).chunk(2, dim=1)
            o = o * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            outs.append(o)
        f, g, m = outs
        tau = self.tau()  # [C_h]，fp32
        gate = -f * (dt / tau[None, :, None, None])
        if self.keep and alpha_prev is not None:
            gate = gate - F.softplus(self.theta_beta) * alpha_prev
        p = torch.sigmoid(gate)
        return p * g + (1.0 - p) * m


def scene_stats(x: torch.Tensor, quality: torch.Tensor | None,
                t: int, T: int) -> torch.Tensor:
    """外环场景统计 s(t)（3.2 表 M：均值/方差/亮斑密度估计/帧序归一化 [+ 清晰度]）。

    x: [B,C,H,W] 主尺度特征；quality: [B] 窗口清晰度（M1 quality.npy，缺省 0）。
    返回 [B,5] fp32。
    """
    mean = x.mean(dim=(1, 2, 3))
    std = x.std(dim=(1, 2, 3))
    # 亮斑密度：超过 μ+3σ 的像素占比（廉价代理，M4 换 SCR/形态判据）
    thr = x.mean(dim=1, keepdim=True) + 3.0 * x.std(dim=1, keepdim=True)
    blob = (x > thr).float().mean(dim=(1, 2, 3))
    frame_idx = torch.full_like(mean, t / max(T, 1))
    q = quality if quality is not None else torch.zeros_like(mean)
    return torch.stack([mean, std, blob, frame_idx, q], dim=1)


def update_feedback_mask(alpha: torch.Tensor, M_prev: torch.Tensor,
                        radius: int = 5, decay: float = 0.9,
                        alpha_th: float = 0.5) -> torch.Tensor:
    """内环反馈掩码（4.5）：M(t) = max(dilate(1[α>α_th], r), decay·M(t−1))。

    "高 α 峰值"= 超过门控阈 α_th（表 I）的二值峰；掩码取值域 {0} ∪ {decay^k} ∪ 1，
    目标经过处背景通道被打码约 1/(1−decay) ≈ 10 帧衰减期。
    """
    peaks = (alpha > alpha_th).float()
    dil = F.max_pool2d(peaks, 2 * radius + 1, stride=1, padding=radius)
    return torch.maximum(dil, decay * M_prev)


class DualStateLiquidCore(nn.Module):
    """主尺度双状态液态核心（方案 4.1 主尺度行；区域尺度 M3 后期接入）。

    forward 输入 [B,T,C,H,W]（FPN 主尺度特征），返回逐帧堆叠的
    logits / y_b / alpha / h_t / h_b / m_tgt。全程 fp32（4.8-①）。
    """

    def __init__(
        self,
        c_in: int,
        c_h: int = 32,
        mode: str = "dual",
        tau_b: tuple[float, float, float] = (16.0, 256.0, 48.0),
        tau_t: tuple[float, float, float] = (2.0, 16.0, 6.0),
        mask_radius: int = 5,
        mask_decay: float = 0.9,
        alpha_th: float = 0.5,
        detach_every: int = 0,
        use_checkpoint: bool = False,
        scene_dim: int = 5,
    ):
        super().__init__()
        assert mode in ("dual", "single")
        self.mode = mode
        self.c_h = c_h
        self.mask_radius = mask_radius
        self.mask_decay = mask_decay
        self.alpha_th = alpha_th  # 4.5"高 α 峰值"阈值（表 I 门控阈 α_th=0.5）
        self.detach_every = detach_every
        # 梯度检查点（4.9-② 显存回退）：逐帧重算换显存，T=32 全幅 BPTT 8GB 卡可训
        self.use_checkpoint = use_checkpoint
        self.dw_ctx = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.bg_head = nn.Conv2d(c_h, c_in, 1)  # ŷ_B：背景一步预测（特征域）
        self.gate_head = nn.Conv2d(c_in + c_h, 1, 1)  # M3 临时门控（M4 换三重门控）
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)  # 零初始化：α≡0.5 中性起步，峰值语义由 α_th 控制
        # 分割头（7.1①）：输入 [α⊙(x−ŷ_B), h_T]（64ch @ 1.0×），监督直接施加在
        # 物理抑制通路上，梯度同时回传门控与双状态
        self.seg_head = nn.Conv2d(c_in + c_h, 1, 1)
        if mode == "dual":
            self.ch_bg = _LiquidChannel(c_in, c_h, *tau_b, keep=False, scene_dim=scene_dim)
            self.ch_tg = _LiquidChannel(c_in, c_h, *tau_t, keep=True, scene_dim=scene_dim)
        else:  # 消融 a：单状态（τ 范围并集，无结构分立）
            tau_single = (min(tau_b[0], tau_t[0]), max(tau_b[1], tau_t[1]),
                          math.sqrt(tau_b[2] * tau_t[2]))
            self.ch_bg = None
            self.ch_tg = _LiquidChannel(c_in, c_h, *tau_single, keep=True,
                                        scene_dim=scene_dim)
        self.last_norms: dict[str, float] = {}  # 4.8-③ 隐状态范数监控（最近一窗）

    # ---- 单帧递推 ----------------------------------------------------------
    def _step_frame(self, h_t, h_b, x, M, alpha_prev, scene, t, T):
        """单帧递推（方案 4.6 伪代码）。返回 (h_t, h_b, ŷ_B, α, 残差 x−ŷ_B)。"""
        x_ctx = self.dw_ctx(x)
        if self.mode == "dual":
            x_b = x_ctx * (1.0 - M)              # 内环：背景通道输入打码
            h_b = self.ch_bg.step(h_b, x_b, scene, None)
            y_b = self.bg_head(h_b)              # ŷ_B [B,C,H,W]
            x_res = x - y_b
            x_t = x_res * (1.0 - M)              # 级联残差输入
            h_t = self.ch_tg.step(h_t, x_t, scene, alpha_prev)
        else:  # 消融 a：单状态——同一液态流兼任背景记忆与目标读出，无级联
            x_b = x_ctx * (1.0 - M)
            h_t = self.ch_tg.step(h_t, x_b, scene, alpha_prev)
            h_b = h_t
            y_b = self.bg_head(h_t)
            x_res = x - y_b
        alpha = torch.sigmoid(self.gate_head(torch.cat([x_res, h_t], dim=1)))
        with torch.no_grad():  # 4.8-③ 范数监控（方案阈：>100 触发检查）
            self.last_norms = {
                "h_t_rms": float(h_t.pow(2).mean().sqrt()),
                "h_b_rms": float(h_b.pow(2).mean().sqrt()),
            }
        return h_t, h_b, y_b, alpha, x_res

    def forward(self, feats: torch.Tensor, quality: torch.Tensor | None = None) -> dict:
        """feats [B,T,C,H,W] → dict(逐帧堆叠输出)。状态/门控/τ 强制 fp32。"""
        with torch.autocast(device_type=feats.device.type, enabled=False):
            feats = feats.float()
            B, T, C, H, W = feats.shape
            h_t = feats.new_zeros(B, self.c_h, H, W)
            h_b = feats.new_zeros(B, self.c_h, H, W)
            M = feats.new_zeros(B, 1, H, W)
            alpha_prev = feats.new_zeros(B, 1, H, W)
            if quality is not None:
                quality = quality.float()
            outs = {k: [] for k in ("logits", "y_b", "alpha", "h_t", "h_b", "m_tgt")}
            for t in range(T):
                x = feats[:, t]
                scene = scene_stats(x, quality, t, T)
                if self.use_checkpoint and self.training:
                    h_t, h_b, y_b, alpha, x_res = checkpoint(
                        self._step_frame, h_t, h_b, x, M, alpha_prev, scene, t, T,
                        use_reentrant=False)
                else:
                    h_t, h_b, y_b, alpha, x_res = self._step_frame(
                        h_t, h_b, x, M, alpha_prev, scene, t, T)
                outs["logits"].append(self.seg_head(torch.cat([alpha * x_res, h_t], dim=1)))
                outs["y_b"].append(y_b)
                outs["alpha"].append(alpha)
                outs["h_t"].append(h_t)
                outs["h_b"].append(h_b)
                # 反馈掩码（4.5）：高 α 峰值膨胀 ∪ decay·M_prev（纯函数见上）
                M = update_feedback_mask(alpha, M, self.mask_radius,
                                         self.mask_decay, self.alpha_th)
                outs["m_tgt"].append(M)
                alpha_prev = alpha
                if self.detach_every and (t + 1) % self.detach_every == 0 and t < T - 1:
                    h_t, h_b, M, alpha_prev = (h_t.detach(), h_b.detach(),
                                               M.detach(), alpha_prev.detach())
            for k in outs:
                outs[k] = torch.stack(outs[k], dim=1)  # [B,T,...]
            return outs

    def tau_report(self) -> dict[str, float]:
        """τ 监控（4.8 / M3 Gate：τ_B 中位 ∈[24,128]，τ_T ∈[3,12]）。逐单元分位数。"""
        def qs(v: torch.Tensor) -> dict[str, float]:
            p = torch.quantile(v.detach().float(),
                               torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32,
                                            device=v.device))
            return {"p10": round(float(p[0]), 2), "median": round(float(p[1]), 2),
                    "p90": round(float(p[2]), 2)}

        out: dict[str, float] = {}
        if self.mode == "dual":
            for k, v in qs(self.ch_bg.tau()).items():
                out[f"tau_b_{k}"] = v
            for k, v in qs(self.ch_tg.tau()).items():
                out[f"tau_t_{k}"] = v
        else:
            for k, v in qs(self.ch_tg.tau()).items():
                out[f"tau_single_{k}"] = v
        return out
