"""双状态液态核心（M3 阶段 A 返工版：λ 域指数泄漏 + 显式保留项 + 有界掩码）。

返工依据：reports/audit/独立代码评审、双状态液态核_可行设计方案 §4.0–4.2、
M3_核心层推进方案 阶段 A（A1–A4）。旧实现（CfC Default 式 p·g+(1−p)·m）的问题
与证据见返工报告；本文件顶部只陈述新动力学的语义与不变量。

更新式（不变量 0，设计文档 §4.1）：
    λ_base = λ_lo + (λ_hi − λ_lo)·σ(θ_λ)            # σ 重参数化：光滑、基值恒在区间内
    mod    = exp(s_max·tanh(f))                     # 输入相关速率调制，包络 ×[e^−s, e^s]
    λ      = clamp(λ_base·mod·(1−κ·α_prev), λ_lo, λ_hi)   # ★ 终 clamp：不变量 0 在此生效
    a      = exp(−Δt·λ)                             # 保留系数 ∈ (0,1)，构造性夹住
    cand   = tanh(g + m_scale·m)                    # 有界写候选 ⇒ ‖h‖ 永不发散
    h'     = a⊙h + (1−a)⊙cand                       # ★ 旧状态是显式保留项

结构不变量（均可被单测证伪，见 tests/test_dsld_core.py）：
  ① a ∈ [exp(−λ_hi), exp(−λ_lo)]，与权重/输入无关（终 clamp）；
  ② 实测单步衰减 == exp(−Δt·λ)（τ 在指数里，非自由门控分母）；
  ③ 两通道 τ 区间不相交（B [24,192] / T [2,8]）⇒ τ_eff_B/τ_eff_T ≥ 3 恒成立；
  ④ 状态有界：|h| ≤ max(|h₀|, 1)（tanh 候选 + a∈(0,1) 的压缩映射）；
  ⑥⑦ 掩码 M ≤ m_max=0.8 且 α ≤ α_th 时 M ≡ 0（"保护慢通道"不可能变成"饿死慢通道"）。

设计决策（解决两份设计文档的矛盾，详见返工报告）：
  - **λ 终 clamp**：补丁草案 A-3 的 λ_base·mod·(1−κα) 无终 clamp——mod×keep 可把
    τ_T 推到 ~32、τ_B 压到 ~12，比值可反转，无法通过 A1 出口判据③。终 clamp 是
    使判据③成立的必要修正；代价是保持项延长上限 = τ_T 区间上界 8 帧（可行方案
    §4.2 原设计即如此，与"约 10 帧"同量级）。
  - **头输入默认仅 x（state_dependent=False）**：可行方案 §4.1 要求 a、c 只依赖 x_t
    （递推对 h 线性，窗口内可并行扫描——阶段 E 的前提）；补丁草案 A-3 保留 [h,x]
    输入则递推非线性。默认 x-only，state_dependent=true 作为消融保留状态相关 τ。
    注意：ŷ_B/M 的间接状态耦合仍在（残差与掩码依赖通道历史），窗口内完全并行需
    阶段 E 的两遍调度；这里只消除直接 h 依赖。
  - f_head 零初始化（权重+偏置）⇒ 起步 mod≡1、τ_eff ≡ τ_base（init 48/6 精确成立，
    τ 监控叙事与 config 一致）；FiLM 零初始化同理。

诚实边界：
  - 旧 ckpt 不兼容（θ_τ→θ_λ 语义变更，参数化不同），从零重训；
  - bound_f 已废弃（指数形式下无意义）：传 >0 会告警；
  - 掩码对静态热点杂波存在"打码→残差自持"锁定模式（ViBe 选择性更新两难的反面），
    由 M4 尺度选择性抑制对冲，M3 Gate 判读 F_a 时须知；
  - 单状态消融（10.8-a）mode="single"：单流 τ 区间取并集 [2,192]，L_dec 在 single
    下不参与（trainer 侧），单双对照的损失组成差异已在返工报告登记。
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _LiquidChannel(nn.Module):
    """单条指数泄漏式液态通道：f/g/m 1×1 头 + σ 重参数化 λ_base + 终 clamp + 保持项。"""

    def __init__(self, c_in: int, c_h: int, tau_min: float, tau_max: float,
                 tau_init: float, keep: bool = False, scene_dim: int = 5,
                 s_max: float = 0.7, kappa_max: float = 0.5, m_scale: float = 0.5,
                 state_dependent: bool = False):
        super().__init__()
        assert tau_min < tau_init < tau_max, \
            "tau_init 必须严格在 (tau_min, tau_max) 内（σ 重参数化起点）"
        self.state_dependent = state_dependent
        self.s_max = float(s_max)
        self.kappa_max = float(kappa_max)
        self.m_scale = float(m_scale)
        self.keep = keep
        head_in = c_in + c_h if state_dependent else c_in
        self.f_head = nn.Conv2d(head_in, c_h, 1)   # 速率调制头（方案 f）
        self.g_head = nn.Conv2d(head_in, c_h, 1)   # 写候选头（方案 g）
        self.m_head = nn.Conv2d(head_in, c_h, 1)   # 候选精修头（方案 m 再解释）
        # f_head 零初始化 ⇒ 起步 mod ≡ 1、τ_eff ≡ τ_base（梯度经权重仍可达，无死路）
        nn.init.zeros_(self.f_head.weight)
        nn.init.zeros_(self.f_head.bias)
        # λ_base = λ_lo + (λ_hi−λ_lo)·σ(θ_λ)：光滑且基值恒在区间内（无 clip 死区）
        self.lam_lo = 1.0 / tau_max
        self.lam_hi = 1.0 / tau_min
        p0 = (1.0 / tau_init - self.lam_lo) / (self.lam_hi - self.lam_lo)
        theta0 = math.log(p0 / (1.0 - p0))
        self.theta_lambda = nn.Parameter(torch.full((c_h,), theta0))
        # FiLM（4.3）：s(t) → 逐头 (γ_s, β_s)，零初始化 = 恒等起步
        self.film = nn.ModuleList(
            [nn.Linear(scene_dim, 2 * c_h) for _ in range(3)])
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # 保持项（4.4）：仅目标通道；κ = kappa_max·σ(θ_κ)，init κ≈0.2（τ 延长 1.33×），
        # 终 clamp 下 τ_eff 上限 = τ_max（可行方案 §4.2："λ 下限 1/τ_min → τ 最多到 τ_max"）
        if keep:
            kappa_init = 0.2
            r = kappa_init / kappa_max
            self.theta_kappa = nn.Parameter(torch.tensor(math.log(r / (1.0 - r))))
        # 诊断：实现出来的动力学量（评审 2.3 / 设计文档 §4.4——Gate 证据必须是动力学空间）
        self.last_f_abs = 0.0
        self.last_tau_eff = float(tau_init)
        self.last_lam_min = self.lam_lo
        self.last_lam_max = self.lam_hi
        self.last_lam_frac_at_bound = 0.0

    def lam_base(self) -> torch.Tensor:
        """基准泄漏速率 λ_base [C_h]（1/帧）——σ 重参数化，恒在 [λ_lo, λ_hi] 内。"""
        return self.lam_lo + (self.lam_hi - self.lam_lo) * torch.sigmoid(self.theta_lambda)

    def tau(self) -> torch.Tensor:
        """基准时间常数 τ_base = 1/λ_base（帧）——就是 a=exp(−Δt/τ) 里的 τ，恒在区间内。"""
        return 1.0 / self.lam_base()

    def keep_gain(self) -> torch.Tensor:
        """保持项增益 κ ∈ (0, κ_max)（仅目标通道）。"""
        if not self.keep:
            return torch.zeros((), device=self.theta_lambda.device)
        return self.kappa_max * torch.sigmoid(self.theta_kappa)

    def step(self, h: torch.Tensor, x_in: torch.Tensor, scene: torch.Tensor,
             alpha_prev: torch.Tensor | None, dt: float = 1.0) -> torch.Tensor:
        """单步泄漏式更新。h/x_in: [B,C,H,W]；scene: [B,5]；alpha_prev: [B,1,H,W]。

        h' = a⊙h + (1−a)⊙cand,  a = exp(−Δt·λ),
        λ = clamp(λ_base·exp(s_max·tanh(f))·(1−κ·α_prev), λ_lo, λ_hi)
        """
        inp = torch.cat([h, x_in], dim=1) if self.state_dependent else x_in
        outs = []
        for i, head in enumerate((self.f_head, self.g_head, self.m_head)):
            o = head(inp)
            gamma, beta = self.film[i](scene).chunk(2, dim=1)
            o = o * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            outs.append(o)
        f, g, m = outs
        # ① 有界输入相关速率调制（"液态"）：tanh ∈ (−1,1)，f 零初始化 ⇒ 起步严格 1.0
        mod = torch.exp(self.s_max * torch.tanh(f))
        # ② 保持项（4.4）：α_prev 大 ⇒ λ 小 ⇒ a 大 ⇒ 记忆更长
        lam = self.lam_base()[None, :, None, None] * mod
        if self.keep and alpha_prev is not None:
            lam = lam * (1.0 - self.keep_gain() * alpha_prev)
        lam = lam.clamp(self.lam_lo, self.lam_hi)   # ★ 不变量 0：终 clamp（判据③前提）
        a = torch.exp(-dt * lam)                    # 保留系数 ∈ (0,1)，构造性夹住
        # ③ 有界写候选：tanh ⇒ 逐元素 |h| ≤ max(|h_prev|, 1)，结构上不可能发散
        cand = torch.tanh(g + self.m_scale * m)
        with torch.no_grad():  # 实现出来的动力学量（Gate 证据），逐帧刷新
            self.last_f_abs = float(f.abs().mean())
            self.last_tau_eff = float((1.0 / lam).mean())
            self.last_lam_min = float(lam.min())
            self.last_lam_max = float(lam.max())
            eps = 1e-6
            self.last_lam_frac_at_bound = float(
                ((lam <= self.lam_lo + eps) | (lam >= self.lam_hi - eps)).float().mean())
        return a * h + (1.0 - a) * cand


def scene_stats(x: torch.Tensor, quality: torch.Tensor | None,
                t: int, T: int) -> torch.Tensor:
    """外环场景统计 s(t)（3.2 表 M：均值/方差/亮斑密度估计/帧序归一化 [+ 清晰度]）。

    x: [B,C,H,W] 主尺度特征；quality: [B] 窗口清晰度（M1 quality.npy，缺省 0）。
    返回 [B,5] fp32。（逐帧 quality 消费属阶段 C1，当前为窗口标量直通。）
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
                        alpha_th: float = 0.6, m_max: float = 0.8,
                        softness: float = 0.0) -> torch.Tensor:
    """内环反馈掩码（4.5，返工 A3）：M(t) = min( max( dilate(peak_t, r), decay·M(t−1) ), m_max )。

    三条不变量：
      ① M ≤ m_max < 1 ⇒ (1−M) ≥ 1−m_max：双通道输入永不被清零，窒息在结构上不可达
        （评审 2.1 的 bias −2 只把故障推出启动区，本上界才把它排除出可行域）；
      ② softness=0 时 α ≤ α_th ⇒ peak ≡ 0："无证据不掩码"，无全局弱泄漏；
      ③ 阈值阶跃对 α 无梯度，α 的梯度仅经 seg_head([α⊙x_res, h_T]) 通路。
    softness>0 为软化版 σ((α−α_th)/softness)：梯度可达，但 α≈α_th 时全图 0.5 量级弱
    掩码、与门控可能成自反馈——仅在确认需要该梯度通路时启用（补丁草案 A-4 取舍表）。
    """
    if softness > 0:
        peaks = torch.sigmoid((alpha - alpha_th) / softness)
    else:
        peaks = (alpha > alpha_th).float()
    dil = F.max_pool2d(peaks, 2 * radius + 1, stride=1, padding=radius)
    return torch.clamp(torch.maximum(dil, decay * M_prev), max=m_max)


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
        tau_b: tuple[float, float, float] = (24.0, 192.0, 48.0),
        tau_t: tuple[float, float, float] = (2.0, 8.0, 6.0),
        mask_radius: int = 5,
        mask_decay: float = 0.9,
        alpha_th: float = 0.6,
        mask_m_max: float = 0.8,
        mask_softness: float = 0.0,
        detach_every: int = 0,
        use_checkpoint: bool = False,
        s_max: float = 0.7,
        kappa_max: float = 0.5,
        m_scale: float = 0.5,
        state_dependent: bool = False,
        bound_f: float = 0.0,   # 已废弃：>0 时告警（旧 config 兼容位）
        scene_dim: int = 5,
    ):
        super().__init__()
        if bound_f > 0:
            warnings.warn(
                "liquid.bound_f 已废弃：指数泄漏形式下 τ 直接约束衰减（终 clamp），"
                "f_bound 的'小则失效、大则不生效'两难不再存在。请从 config 移除。",
                DeprecationWarning, stacklevel=2)
        assert mode in ("dual", "single")
        self.mode = mode
        self.c_h = c_h
        self.mask_radius = mask_radius
        self.mask_decay = mask_decay
        # 4.5"高 α 峰值"阈值：0.6 与 gate_head 偏置 −2（α₀≈0.119）之间留足裕度（A3）
        self.alpha_th = alpha_th
        self.mask_m_max = mask_m_max
        self.mask_softness = mask_softness
        self.detach_every = detach_every
        # 梯度检查点（4.9-② 显存回退）：逐帧重算换显存
        self.use_checkpoint = use_checkpoint
        self.dw_ctx = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.bg_head = nn.Conv2d(c_h, c_in, 1)  # ŷ_B：背景一步预测（特征域）
        self.gate_head = nn.Conv2d(c_in + c_h, 1, 1)  # M3 临时门控（M4 换三重门控）
        nn.init.zeros_(self.gate_head.weight)
        # 偏置 −2：α 初始 ≈ σ(−2) = 0.119，与 α_th=0.6 阈值解耦。评审 2.1 教训：
        # 零初始化（α≡0.5）+ 严格 > 阈值判定是刀刃条件——训练一拍后 α 以 0.0003 裕度
        # 越阈（smoke_qe 实测 frac_high=0.94+），反馈掩码铺满全图 → 双通道输入 ≈0
        #（自我窒息）→ L_recon 有效像素为空静默归零。低置信起步 = "无证据不掩码"。
        nn.init.constant_(self.gate_head.bias, -2.0)
        # 分割头（7.1①）：输入 [α⊙(x−ŷ_B), h_T]（64ch @ 1.0×），监督直接施加在
        # 物理抑制通路上，梯度同时回传门控与双状态
        self.seg_head = nn.Conv2d(c_in + c_h, 1, 1)
        chan_kw = dict(scene_dim=scene_dim, s_max=s_max, kappa_max=kappa_max,
                       m_scale=m_scale, state_dependent=state_dependent)
        if mode == "dual":
            self.ch_bg = _LiquidChannel(c_in, c_h, *tau_b, keep=False, **chan_kw)
            self.ch_tg = _LiquidChannel(c_in, c_h, *tau_t, keep=True, **chan_kw)
        else:  # 消融 a：单状态（τ 区间并集，无结构分立）
            tau_single = (min(tau_b[0], tau_t[0]), max(tau_b[1], tau_t[1]),
                          math.sqrt(tau_b[2] * tau_t[2]))
            self.ch_bg = None
            self.ch_tg = _LiquidChannel(c_in, c_h, *tau_single, keep=True, **chan_kw)
        self.last_norms: dict[str, float] = {}  # 4.8-③ 隐状态范数 + 掩码/门控运行统计

    # ---- 单帧递推 ----------------------------------------------------------
    def _step_frame(self, h_t, h_b, x, M, alpha_prev, scene, t, T):
        """单帧递推（方案 4.6 伪代码）。返回 (h_t, h_b, ŷ_B, α, 残差 x−ŷ_B)。"""
        x_ctx = self.dw_ctx(x)
        if self.mode == "dual":
            x_b = x_ctx * (1.0 - M)              # 内环：背景通道输入打码（≤m_max 永不清零）
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
        with torch.no_grad():  # 4.8-③ 范数监控（有界候选下 |h|≤1，范数哨兵已结构性满足）
            self.last_norms = {
                "h_t_rms": float(h_t.pow(2).mean().sqrt()),
                "h_b_rms": float(h_b.pow(2).mean().sqrt()),
                "m_frac": float(M.mean()),        # 反馈掩码覆盖率（Gate 判据 <0.3）
                "alpha_mean": float(alpha.mean()),  # 门控基线（init ≈ 0.119）
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
                # 反馈掩码（4.5）：高 α 峰值膨胀 ∪ decay·M_prev，上界 m_max（返工 A3）
                M = update_feedback_mask(alpha, M, self.mask_radius,
                                         self.mask_decay, self.alpha_th,
                                         self.mask_m_max, self.mask_softness)
                outs["m_tgt"].append(M)
                alpha_prev = alpha
                if self.detach_every and (t + 1) % self.detach_every == 0 and t < T - 1:
                    h_t, h_b, M, alpha_prev = (h_t.detach(), h_b.detach(),
                                               M.detach(), alpha_prev.detach())
            for k in outs:
                outs[k] = torch.stack(outs[k], dim=1)  # [B,T,...]
            return outs

    def tau_report(self) -> dict[str, float]:
        """τ 监控（返工 A1/A6）。基准量 = 1/λ_base 分位（参数空间，光滑有界）；
        实现量 = τ_eff / 尺度比 / 边界占用率（动力学空间——Gate 判据以这些为准）。

        Gate（返工后）：tau_scale_ratio ≥ 3 为构造性质（只需确认 lam_at_bound 未把
        自由度吃死，<0.5 为健康）；tau_b/tau_t 分位降为必要非充分。
        """
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
            # 实现出来的时间常数与尺度分离（终 clamp 下比值 ≥3 为结构性质，
            # tau_scale_ratio 偏离 3× 区间比即调制顶边、lam_at_bound 定位原因）
            out["tau_eff_b"] = round(self.ch_bg.last_tau_eff, 2)
            out["tau_eff_t"] = round(self.ch_tg.last_tau_eff, 2)
            out["tau_scale_ratio"] = round(
                out["tau_eff_b"] / max(out["tau_eff_t"], 1e-3), 2)
            out["lam_at_bound_b"] = round(self.ch_bg.last_lam_frac_at_bound, 3)
            out["lam_at_bound_t"] = round(self.ch_tg.last_lam_frac_at_bound, 3)
        else:
            for k, v in qs(self.ch_tg.tau()).items():
                out[f"tau_single_{k}"] = v
            out["tau_eff_single"] = round(self.ch_tg.last_tau_eff, 2)
            out["lam_at_bound_single"] = round(self.ch_tg.last_lam_frac_at_bound, 3)
        return out
