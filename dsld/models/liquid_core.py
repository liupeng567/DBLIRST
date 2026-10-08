"""双状态液态核（M3 v2.0 方案 §4.3）：λ 域精确离散 + 保护式更新 + Δt 变步长 + 四臂。

动力学（§4.3.1，CfC 闭式单元的 λ 域精确离散化）
    x_in  = film(dw3x3(x ⊙ (1−M)), s(t))          # 邻域上下文 + 5 维场景条件
    mod   = 1 + s_max·tanh(f(x_in))               # 输入相关速率调制（f 零初始化 ⇒ ≡1）
    λ     = clamp(λ_base·mod, 1/τ_max, 1/τ_min)    # σ 重参数化 + ★终 clamp
    a     = exp(−Δt·λ)                            # Δt = 该帧槽距上一个新曝光的槽数
    a_eff = a + (1−a)·M                           # 保护式更新（ViBe 可微推广，仅背景通道）
    h'    = a_eff⊙h + (1−a_eff)⊙tanh(g + β·α_prev) # 候选有界；β·α 保持项仅目标通道

结构不变量（每条都有 tests/test_liquid_core.py 的可失败断言）：
  ① a ∈ [exp(−Δt·λ_hi), exp(−Δt·λ_lo)]，与权重/输入无关（终 clamp）；
  ② |h| ≤ 1：tanh 候选 + 系数凸组合 ⇒ 逐位有界（随机长序列 + 极端输入实测）；
  ③ 双通道 τ 区间不相交（B [24,192] / T [2,8]）⇒ scale_ratio = τ_B/τ_T ≥ 3 **对任意
     θ 与任意输入成立**（24/8=3）；
  ④ teacher 完美保护：M≡1 处 a_eff≡1 ⇒ h_B 跨帧**逐位不变**（判据 ② 的机制本体）；
  ⑤ gate 自举掩码 M ≤ m_max=0.8 ⇒ (1−M) ≥ 0.2，双通道输入永不被清零（L1 窒息排除出
     可行域）；α ≤ α_th 时本帧源 ≡0（"无证据不掩码"，无全局弱泄漏）；
  ⑥ α₀ = σ(bias_init) = σ(−2) = 0.1192 精确（末层零权重）——起步掩码分数低，窒息不可达；
  ⑦ EMA 臂：M>0.5 处 y_b 逐位等于上一帧值（被保护目标零渗入背景估计）。

三条**动力学级**偏差（与方案伪代码；其余实现级偏差 D-P1-4…D-P1-12 见行内注释与
reports/m3/P1_核心动力学_实施报告.md §4，理由都写在被改的那一行旁边）：
  D-P1-1 **Δt 变步长**（§4.3.1 的 Δt=1 特例化推广）：ITTD 是 ≈33⅓Hz 实采被补帧成
    50fps 容器（严格周期-3 字节级重复帧，P0 实测）。补帧槽喂给背景的是一张**过期测量**，
    既不该重写状态也不该衰减 ⇒ dt=0 ⇒ a=1、1−a=0（逐位冻结）；新曝光槽 dt∈{1,2} ⇒
    衰减按真实间隔。数据侧不变量 Σ_{s≤t} dt_s = t+1（容器时间逐槽守恒）由 IttdWindows 保证。
  D-P1-2 **保护式更新只作用于背景通道**：伪代码把 M 同时传进 step_bg/step_tg，字面执行
    会让 teacher 模式（M≡1 于 GT 区）把 h_T 也**永久冻结在零初始化**——目标通道恰好在
    目标位置失去可塑性，L_seg 的 h_T 输入支路变死支，判据 ⑥ 的 dual-vs-single 差异无法
    归因给"双时间尺度"。故保护 = 背景通道专属机制（它要防的是"目标被背景吸收"），
    目标通道靠 x_T = r⊙(1−M) 的输入侧打码 + β·α_prev 保持项（§4.3.1 明示"目标通道专属"）。
  D-P1-3 **single 臂保留保护式更新**：单流同时兼任背景记忆，若不冻结则 single 与 dual
    差两件事（结构分立 + 保护有无），判据 ⑥ 失焦；故 single 的 ch_tg.protect=True。

掩码三源（§4.3.4，逐帧递推 M_t = clamp(max(本帧源, decay·M_{t−1}), 0, cap)）：
  teacher — 本帧源 = GT 膨胀掩码（与 L_recon 剔除区同几何，register 同一对函数产出），
            cap = 1.0（"完美保护"是阳性对照的前提，不能被 m_max 截断）；
  gate    — 本帧源 = 上一帧 α 的峰值膨胀 radius px，cap = m_max（自举闭环；
            **核心层兜底忽略 teacher**，语义防护不依赖装配层）；
  off     — 本帧源 ≡0（保护关闭臂，判据 ②/④ 对照）。
  阈值膨胀对 α 无梯度（硬阈值）⇒ α 的梯度仅经 seg_head([α⊙r, h_T]) 与 L_gate 两条通路，
  这是设计意图（监督直接施加在物理抑制通路上，总方案 7.1①），不是遗漏。

四臂（§4.5，全部在本文件实现，装配层只透传）：
  bg_mode=ema（非学习 EMA 背景 + 掩码处保持旧值）、state_mode=single、tau_mode=swap、
  mask_source=off。EMA 动量按 dt 幂次换算（dt=0 ⇒ 系数 1 ⇒ 逐位保持），与 ① 同口径。

诚实边界：
  - 旧 ckpt 不兼容（θ_τ→θ_λ、候选头 3→2、门控 1×1→MLP），从零重训；
  - resid_scr / bg_resid_rms / recon_valid_frac **不在核心算**：它们需要 GT，而核心是
    纯状态机（§4.3.5 把它们列在核心诊断，实现挪到损失/快评层，口径不变）；
  - 掩码对静态热点杂波存在"打码→残差自持"锁定模式（ViBe 选择性更新两难的反面），
    由 M4 尺度选择性抑制对冲，判读 F_a 时须知；
  - 掩码在 dw 卷积**之前**施加（目标能量不进邻域上下文），代价是掩码区被填 0 = 归一化域
    的黑洞：M=1 处该洞因 a_eff≡1 完全不进状态；0<M<1 处洞的影响按 (1−a)(1−M) 加权，
    与保护强度严格反比——不会污染完美保护区，但在部分保护区引入向 0 的偏置（M4 可换
    邻域内插修复）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_EPS = 1e-6


def _logit(p: float) -> float:
    """σ 重参数化的逆：θ = ln(p/(1−p))，使 σ(θ)=p（σ 输出恒在 (0,1) ⇒ 无 clip 死区）。"""
    return math.log(p / (1.0 - p))


def scene_stats(x: torch.Tensor, quality: torch.Tensor | None, prog: torch.Tensor) -> torch.Tensor:
    """外环 5 维场景向量 s(t) = [帧均值, 帧方差, 亮斑密度, 时间进度, quality_med]（§4.3.1）。

    x [B,C,H,W]；quality [B] 或 None（窗口清晰度中位，与 IttdWindows 同源，消除训练/评测
    FiLM 偏移）；prog [B] ∈[0,1] 为该帧的**容器时间进度**（dt 前缀和 / 窗总时长，GPU 上算，
    不在循环里 float()）。返回 [B,5] fp32。
    """
    mean = x.mean(dim=(1, 2, 3))
    std = x.std(dim=(1, 2, 3))
    thr = mean[:, None] + 3.0 * std[:, None]
    blob = (x > thr[:, :, None, None]).float().mean(dim=(1, 2, 3))
    prog = prog.clamp(0.0, 1.0)
    q = quality if quality is not None else torch.zeros_like(mean)
    return torch.stack([mean, std, blob, prog, q], dim=1)


def alpha_peaks(alpha: torch.Tensor, radius: int, alpha_th: float) -> torch.Tensor:
    """门控峰值本帧源：α > α_th 的像素经方核膨胀 radius px（§4.3.4，硬阈值、对 α 无梯度）。"""
    peaks = (alpha > alpha_th).float()
    return F.max_pool2d(peaks, 2 * radius + 1, stride=1, padding=radius)


def feedback_mask(M_prev: torch.Tensor, source: torch.Tensor, decay: float,
                  cap: float) -> torch.Tensor:
    """内环掩码递推 M_t = clamp(max(source, decay·M_{t−1}), 0, cap)（§4.3.4）。

    cap 是 L1 窒息纪律的落点：gate 源用 m_max<1（输入永不被清零），teacher 源用 1.0
    （完美保护 = 判据 ② 阳性对照的前提）。decay=0.9 ⇒ 目标离开后打码约 10 帧，
    短于 τ_B≥24 的吸收窗口（"先保护后吸收"，§4.4 时序自洽）。
    """
    return torch.clamp(torch.maximum(source, decay * M_prev), 0.0, cap)


class _LiquidChannel(nn.Module):
    """单条液态通道：dw 上下文后的 f/g 两头 + σ 重参数化 λ_base + 终 clamp + 保护/保持项。

    protect=True ⇒ a_eff = a + (1−a)·M（背景通道；M=1 逐位冻结）；
    keep=True    ⇒ 候选 = tanh(g + β·α_prev)（目标通道；β = β_max·σ(θ_β)，θ_β=0 ⇒ β=β_max/2）。
    """

    def __init__(self, c_in: int, c_h: int, tau_min: float, tau_max: float,
                 tau_init: float, keep: bool = False, protect: bool = False,
                 scene_dim: int = 5, s_max: float = 0.7, beta_max: float = 1.0,
                 state_dependent: bool = False):
        super().__init__()
        assert tau_min < tau_init < tau_max, "tau_init 必须严格在 (tau_min, tau_max) 内"
        assert 0 < s_max < 1.0, "s_max<1 保证 mod=1+s_max·tanh(f)>0（乘性调制不得变号）"
        self.state_dependent = state_dependent
        self.s_max = float(s_max)
        self.beta_max = float(beta_max)
        self.keep = keep
        self.protect = protect
        head_in = c_in + c_h if state_dependent else c_in
        self.f_head = nn.Conv2d(head_in, c_h, 1)   # 速率调制头（§4.3.2 f）
        self.g_head = nn.Conv2d(head_in, c_h, 1)   # 候选头（§4.3.1 g）
        # f_head 零初始化 ⇒ 起步 mod ≡ 1 ⇒ tau_eff 精确等于 tau_init（单测 ⑤/test5）；
        # 权重非零时梯度仍可达（零权重不是冻结），无死路。
        nn.init.zeros_(self.f_head.weight)
        nn.init.zeros_(self.f_head.bias)
        self.lam_lo = 1.0 / tau_max
        self.lam_hi = 1.0 / tau_min
        p0 = (1.0 / tau_init - self.lam_lo) / (self.lam_hi - self.lam_lo)
        self.theta_lambda = nn.Parameter(torch.full((c_h,), _logit(p0)))
        if keep:
            self.theta_beta = nn.Parameter(torch.zeros(()))  # β_init = beta_max/2
        self.film = nn.ModuleList([nn.Linear(scene_dim, 2 * c_h) for _ in range(2)])
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)                        # 恒等起步（不注入场景偏置）
        # 诊断（仅末帧更新；float() 强制 GPU→CPU 同步，逐帧做 = L5 死开销）
        self.last_tau_eff = float(tau_init)
        self.last_lam_frac_at_bound = 0.0

    def lam_base(self) -> torch.Tensor:
        """基准泄漏速率 λ_base [C_h]：σ 重参数化，恒在 [λ_lo, λ_hi]（1/帧）。"""
        return self.lam_lo + (self.lam_hi - self.lam_lo) * torch.sigmoid(self.theta_lambda)

    def tau(self) -> torch.Tensor:
        """基准时间常数 τ_base = 1/λ_base（帧）——就是 a=exp(−Δt/τ) 里的 τ。"""
        return 1.0 / self.lam_base()

    def beta_gain(self) -> torch.Tensor:
        """保持项增益 β ∈ (0, β_max)（仅目标通道）。"""
        if not self.keep:
            return torch.zeros((), device=self.theta_lambda.device)
        return self.beta_max * torch.sigmoid(self.theta_beta)

    def step(self, h: torch.Tensor, u: torch.Tensor, scene: torch.Tensor,
             dt: torch.Tensor, alpha_prev: torch.Tensor | None,
             mask: torch.Tensor | None = None, collect_stats: bool = True) -> torch.Tensor:
        """单步更新。h/u: [B,C_h/C_in,H,W]；scene: [B,5]；dt: [B]（帧槽间隔）；
        alpha_prev: [B,1,H,W] 或 None；mask: [B,1,H,W]（仅 protect 通道消费）。
        """
        inp = torch.cat([h, u], dim=1) if self.state_dependent else u
        outs = []
        for i, head in enumerate((self.f_head, self.g_head)):
            o = head(inp)
            gamma, shift = self.film[i](scene).chunk(2, dim=1)
            o = o * (1.0 + gamma[:, :, None, None]) + shift[:, :, None, None]
            outs.append(o)
        f, g = outs
        mod = 1.0 + self.s_max * torch.tanh(f)              # ∈ (1−s_max, 1+s_max)，恒正
        lam = (self.lam_base()[None, :, None, None] * mod).clamp(self.lam_lo, self.lam_hi)
        a = torch.exp(-dt[:, None, None, None] * lam)       # Δt=0 ⇒ a≡1（补帧槽冻结）
        if self.protect and mask is not None:
            a = a + (1.0 - a) * mask                        # ★ 保护式：M=1 ⇒ a_eff≡1
        cand = torch.tanh(g + self.beta_gain() * alpha_prev) if (
            self.keep and alpha_prev is not None) else torch.tanh(g)
        if collect_stats:
            with torch.no_grad():
                self.last_tau_eff = float((1.0 / lam).mean())
                eps = 1e-6
                self.last_lam_frac_at_bound = float(
                    ((lam <= self.lam_lo + eps) | (lam >= self.lam_hi - eps))
                    .float().mean())
        return a * h + (1.0 - a) * cand


class DualStateLiquidCore(nn.Module):
    """主尺度双状态液态核（§4.3）。forward(feats [B,T,C,H,W]) → 逐帧堆叠输出 dict。

    输出键（全部 stride-2，上采样属 P4 推理档，此处不留无消费者开关）：
      logits [B,T,1,H,W]  seg_head([α⊙r, h_T]) 的原始 logit（损失侧过 BCE/Dice）
      alpha  [B,T,1,H,W]  门控概率图
      y_b    [B,T,C,H,W]  背景一步预测 ŷ_B（L_recon 目标；bg_mode=ema 时为 EMA 背景）
      r      [B,T,C,H,W]  残差 x − ŷ_B（L_recon 剔除区 / resid_scr 的同一张图）
      h_t/h_b [B,T,C_h,H,W] 两路隐状态（L_dec 用；single 臂 h_b ≡ h_t）
      m_tgt  [B,T,1,H,W]  本帧实际生效的内环掩码（teacher 模式即 GT，gate 模式即自举）

    全程 fp32（§4.3.7；编码器/头在装配层走 autocast bf16）。
    """

    def __init__(self, c_in: int, c_h: int = 24, state_mode: str = "dual",
                 tau_mode: str = "normal", mask_source: str = "teacher",
                 bg_mode: str = "learned", tau_b: tuple[float, float, float] = (24.0, 192.0, 48.0),
                 tau_t: tuple[float, float, float] = (2.0, 8.0, 6.0),
                 mask_decay: float = 0.9, mask_radius: int = 5, m_max: float = 0.8,
                 alpha_th: float = 0.5, ema_momentum: float = 0.9, scene_dim: int = 5,
                 s_max: float = 0.7, beta_max: float = 1.0, state_dependent: bool = False,
                 gate_hidden: int = 16, gate_bias_init: float = -2.0,
                 use_checkpoint: bool = False):
        super().__init__()
        for name, val, allowed in (
                ("state_mode", state_mode, ("dual", "single")),
                ("tau_mode", tau_mode, ("normal", "swap")),
                ("mask_source", mask_source, ("teacher", "gate", "off")),
                ("bg_mode", bg_mode, ("learned", "ema"))):
            assert val in allowed, f"{name}={val!r} 非法，可选 {allowed}"
        assert 0.0 < m_max < 1.0, "gate 源的 m_max 必须 <1（L1 窒息排除出可行域）"
        assert 0.0 < ema_momentum < 1.0
        self.state_mode = state_mode
        self.tau_mode = tau_mode
        self.mask_source = mask_source
        self.bg_mode = bg_mode
        self.c_h = c_h
        self.mask_decay = float(mask_decay)
        self.mask_radius = int(mask_radius)
        self.m_max = float(m_max)
        self.alpha_th = float(alpha_th)
        self.ema_momentum = float(ema_momentum)
        self.use_checkpoint = use_checkpoint
        kw = dict(scene_dim=scene_dim, s_max=s_max, beta_max=beta_max,
                  state_dependent=state_dependent)
        tb, tt = tau_b, tau_t
        if tau_mode == "swap":     # 判据 ④ 的因果臂：范围互换，其他一律不变
            tb, tt = tau_t, tau_b
        if state_mode == "dual":
            self.ch_bg = _LiquidChannel(c_in, c_h, *tb, keep=False, protect=True, **kw)
            self.ch_tg = _LiquidChannel(c_in, c_h, *tt, keep=True, protect=False, **kw)
            self.ch_single = None
        else:                      # 单状态臂：一个流兼任背景记忆与目标读出（D-P1-3）
            self.ch_bg = None
            self.ch_single = _LiquidChannel(
                c_in, c_h, min(tb[0], tt[0]), max(tb[1], tt[1]),
                math.sqrt(tb[2] * tt[2]), keep=True, protect=True, **kw)
            self.ch_tg = self.ch_single
        self.dw_ctx = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.bg_head = nn.Conv2d(c_h, c_in, 1)          # ŷ_B 学习式读出（bg_mode=ema 时不用）
        self.t_readout = nn.Conv2d(c_h, 1, 1)           # 门控输入③：h_T 的目标证据读出
        # 门控 α（§4.3.3）：MLP[4→gate_hidden→1] + sigmoid，逐位置（1×1 卷积即 MLP）。
        # 输入 4 通道 = [blur(|r|)/σ̂, 局部 SCR, h_T 读出, quality 广播]。
        self.gate = nn.Sequential(nn.Conv2d(4, gate_hidden, 1), nn.SiLU(),
                                  nn.Conv2d(gate_hidden, 1, 1))
        # 末层零权重 + 偏置 gate_bias_init ⇒ α ≡ σ(−2) = 0.1192 **精确**（不变量 ⑥）。
        # 代价：t=0 时 gate 第一层权重梯度为 0（链式穿过零权重），一次更新后恢复——
        # 起步"无证据不掩码"的确定性比第一层的初始噪声重要得多（L1 教训）。
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, float(gate_bias_init))
        self.seg_head = nn.Conv2d(c_in + c_h, 1, 1)     # logits = head([α⊙r, h_T])
        self.last_norms: dict[str, float] = {}

    # ---- 门控输入（无 GT，全部锚点系同帧量） --------------------------------
    def _alpha_map(self, r: torch.Tensor, h_t: torch.Tensor, quality_val: torch.Tensor,
                   shape: torch.Size) -> torch.Tensor:
        """α = sigmoid(MLP([z, scr, h_T 读出, quality]))（§4.3.3）。

        z = 3×3 均值(|r|) / σ̂_r，scr = (3×3 均值 − 环带 4..7 均值) / σ̂_r：前者是局部显著性、
        后者是"中心超出邻环"的目标性判据（小目标的经典对比度），σ̂_r 用本帧残差标准差归一，
        使两者对背景增益/亮度漂移近似不变（FiLM 的 quality 通道再补一个全局档）。
        """
        mag = r.abs().mean(dim=1, keepdim=True)
        sig = mag.std(dim=(2, 3), keepdim=True) + _EPS
        b3 = F.avg_pool2d(mag, 3, stride=1, padding=1)
        b7 = F.avg_pool2d(mag, 7, stride=1, padding=3)
        ring = (49.0 * b7 - 9.0 * b3) / 40.0            # 3×3 之外的环带均值
        q = quality_val.expand(-1, -1, *shape[-2:]) if quality_val is not None else \
            torch.zeros_like(mag)
        return torch.sigmoid(self.gate(torch.cat([b3 / sig, (b3 - ring) / sig,
                                                  self.t_readout(h_t), q], dim=1)))

    # ---- 单帧递推 ----------------------------------------------------------
    def _step_frame(self, h_t: torch.Tensor, h_b: torch.Tensor, y_b_ema: torch.Tensor,
                    x: torch.Tensor, M: torch.Tensor, alpha_prev: torch.Tensor,
                    scene: torch.Tensor, dt: torch.Tensor, q_val: torch.Tensor,
                    collect_stats: bool = True):
        """单帧递推（§4.3.6 伪代码）。返回 (h_t, h_b, y_b_ema, ŷ_B, α, r, M)。

        掩码在 dw 卷积**之前**施加：背景的邻域上下文必须看不到目标能量，否则保护期内
        目标仍会以光环形式进入 ŷ_B 并被吸收（判据 ② 的机制被自身实现抵消）。
        """
        x_masked = self.dw_ctx(x * (1.0 - M))
        if self.state_mode == "dual":
            h_b = self.ch_bg.step(h_b, x_masked, scene, dt, None, mask=M,
                                  collect_stats=collect_stats)
            if self.bg_mode == "ema":
                # 滑动对象是 **x 本身**，不是 dw 上下文：两臂的 ŷ_B 都必须预测同一个量
                # （L_recon 的重构目标就是 x），否则 EMA 臂的残差多一项系统性 dw 偏差、
                # 判据 ③ 的 learned-vs-ema 比较不对等。
                keep = torch.pow(self.ema_momentum, dt)[:, None, None, None]  # dt=0 ⇒ 1
                cand = keep * y_b_ema + (1.0 - keep) * x
                y_b_ema = torch.where((M > 0.5).expand_as(cand), y_b_ema, cand)  # ViBe 选择性更新
                y_b = y_b_ema
            else:
                y_b = self.bg_head(h_b)
            r = x - y_b
            u_t = self.dw_ctx(r * (1.0 - M))
            h_t = self.ch_tg.step(h_t, u_t, scene, dt, alpha_prev, collect_stats=collect_stats)
        else:
            h_t = self.ch_single.step(h_t, x_masked, scene, dt, alpha_prev, mask=M,
                                      collect_stats=collect_stats)
            h_b = h_t                                   # single 臂：cos²≡1（§4.5）
            if self.bg_mode == "ema":
                keep = torch.pow(self.ema_momentum, dt)[:, None, None, None]
                cand = keep * y_b_ema + (1.0 - keep) * x
                y_b_ema = torch.where((M > 0.5).expand_as(cand), y_b_ema, cand)
                y_b = y_b_ema
            else:
                y_b = self.bg_head(h_t)
            r = x - y_b
        alpha = self._alpha_map(r, h_t, q_val, r.shape)
        if collect_stats:                               # 末帧门控（L5：禁逐帧 GPU→CPU 同步）
            self.last_norms = {
                "h_t_rms": float(h_t.pow(2).mean().sqrt()),
                "h_b_rms": float(h_b.pow(2).mean().sqrt()),
                "m_frac": float(M.mean()),
                "alpha_mean": float(alpha.mean()),
                "alpha_p90": float(torch.quantile(alpha.detach().flatten().float(), 0.9)),
                "resid_rms": float(r.pow(2).mean().sqrt()),
            }
        return h_t, h_b, y_b_ema, y_b, alpha, r, M

    def forward(self, feats: torch.Tensor, dt: torch.Tensor | None = None,
                 quality: torch.Tensor | None = None,
                 teacher_mask: torch.Tensor | None = None) -> dict:
        """feats [B,T,C,H,W] → dict（逐帧堆叠）。状态/门控/λ/α 强制 fp32（§4.3.7）。

        dt [B,T]：帧槽间隔（IttdWindows 的 `dt`，0=补帧槽）；None ⇒ 全 1（等间隔合成数据
        /单测的常规口径）。teacher_mask [B,T,1,H,W]：stride-2、已膨胀下采样，仅
        mask_source=teacher 时消费；gate/off 模式在**核心层**忽略（语义防护不依赖装配层）。
        """
        if self.mask_source == "teacher" and teacher_mask is None:
            raise ValueError("mask_source=teacher 必须提供 teacher_mask（静默退回 gate 会把"
                             "判据 ② 的阳性对照降级成自举臂，装配错误必须显式失败）")
        B, T, C, H, W = feats.shape
        with torch.autocast(device_type=feats.device.type, enabled=False):
            feats = feats.float()
            if dt is None:
                dt_seq = feats.new_ones(B, T)
            else:
                dt_seq = dt.float()
                assert tuple(dt_seq.shape) == (B, T), "dt 形状必须为 [B,T]"
            quality = quality.float() if quality is not None else None
            q_val = None if quality is None else quality[:, None, None, None]
            if teacher_mask is not None:
                teacher_mask = teacher_mask.float()
            # 时间进度 = dt 前缀和 / 窗总时长（全程 GPU，循环内不得 float()——L5）
            dt_cum = torch.cumsum(dt_seq, dim=1)
            dt_total = dt_cum[:, -1:].clamp_min(1.0)
            h_t = feats.new_zeros(B, self.c_h, H, W)
            h_b = feats.new_zeros(B, self.c_h, H, W) if self.state_mode == "dual" else h_t
            y_b_ema = feats.new_zeros(B, C, H, W)
            M = feats.new_zeros(B, 1, H, W)
            alpha_prev = feats.new_zeros(B, 1, H, W)
            outs = {k: [] for k in ("logits", "y_b", "r", "alpha", "h_t", "h_b", "m_tgt")}
            for t in range(T):
                x = feats[:, t]
                dtt = dt_seq[:, t]
                scene = scene_stats(x, quality, dt_cum[:, t] / dt_total[:, 0])
                # ---- 本帧掩码（§4.3.4 三源；teacher 上限 1.0，gate 上限 m_max）----
                if self.mask_source == "teacher":
                    source, cap = teacher_mask[:, t], 1.0
                elif self.mask_source == "gate":
                    source, cap = alpha_peaks(alpha_prev, self.mask_radius, self.alpha_th), \
                        self.m_max
                else:
                    source, cap = M.new_zeros(M.shape), self.m_max
                M = feedback_mask(M, source, self.mask_decay, cap)
                collect = t == T - 1                    # 诊断量仅末帧（不变量 ② 之外的 L5）
                args = (h_t, h_b, y_b_ema, x, M, alpha_prev, scene, dtt, q_val, collect)
                if self.use_checkpoint and self.training:
                    h_t, h_b, y_b_ema, y_b, alpha, r, M = checkpoint(
                        self._step_frame, *args, use_reentrant=False)
                else:
                    h_t, h_b, y_b_ema, y_b, alpha, r, M = self._step_frame(*args)
                outs["logits"].append(self.seg_head(torch.cat([alpha * r, h_t], dim=1)))
                outs["y_b"].append(y_b)
                outs["r"].append(r)
                outs["alpha"].append(alpha)
                outs["h_t"].append(h_t)
                outs["h_b"].append(h_b)
                outs["m_tgt"].append(M)
                alpha_prev = alpha
            for k in outs:
                outs[k] = torch.stack(outs[k], dim=1)   # [B,T,...]
            return outs

    def tau_report(self) -> dict[str, float]:
        """τ 诊断（L3 报告口径：只认动力学量 tau_eff / scale_ratio / lam_at_bound）。

        基准量（1/λ_base 分位）是参数空间的必要非充分描述；scale_ratio 由终 clamp 构造
        保证 ≥3，运行期只需确认 lam_at_bound 没把自由度吃死（<0.5 健康）。
        """
        def qs(v: torch.Tensor) -> dict[str, float]:
            p = torch.quantile(v.detach().float(),
                               torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32,
                                            device=v.device))
            return {"p10": round(float(p[0]), 2), "median": round(float(p[1]), 2),
                    "p90": round(float(p[2]), 2)}

        out: dict[str, float] = {}
        if self.state_mode == "dual":
            for k, v in qs(self.ch_bg.tau()).items():
                out[f"tau_b_{k}"] = v
            for k, v in qs(self.ch_tg.tau()).items():
                out[f"tau_t_{k}"] = v
            out["tau_eff_b"] = round(self.ch_bg.last_tau_eff, 2)
            out["tau_eff_t"] = round(self.ch_tg.last_tau_eff, 2)
            out["tau_scale_ratio"] = round(out["tau_eff_b"] / max(out["tau_eff_t"], 1e-3), 2)
            out["lam_at_bound_b"] = round(self.ch_bg.last_lam_frac_at_bound, 3)
            out["lam_at_bound_t"] = round(self.ch_tg.last_lam_frac_at_bound, 3)
        else:
            for k, v in qs(self.ch_single.tau()).items():
                out[f"tau_single_{k}"] = v
            out["tau_eff_single"] = round(self.ch_single.last_tau_eff, 2)
            out["lam_at_bound_single"] = round(self.ch_single.last_lam_frac_at_bound, 3)
        return out
