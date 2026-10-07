# DSLD 项目当前状态：独立代码评审（静态阅读，未执行）

| 项 | 内容 |
| --- | --- |
| 评审对象 | 工作区当前快照（`D:\DBLIRST`），内容与 `reports/` 下 M0–M3 报告一致（最近报告日期 2026-10-06） |
| 评审方式 | 纯静态阅读：逐文件通读代码 + 交叉核对仓库内既有产物（`metrics.jsonl`、`eval_*.json`、缓存目录、报告）。**未执行任何代码** |
| 未执行的限制 | 本机 PowerShell 被 DSH 文件沙箱 ACL 故障阻断（`SetNamedSecurityInfoW failed (Win32 5): grantWrite(D:\DBLIRST)`），因此**没有跑测试、没有实测参数量/FLOPs/显存/速度、没有测量缓存体积** |
| 证据约定 | 所有路径相对仓库根目录；行号以工作区当前版本为准。凡属推断均显式标注 |

---

## 0. 结论摘要

这是一个**工程素养明显高于平均水平的科研脚手架**，但**目前还没有任何研究结论**。三个层次必须分开评价，混在一起会失真：

| 层次 | 实况 |
| --- | --- |
| M0 / M1（数据 + 评估基建） | **真交付**，证据链可核验，报告敢写自己的失败项 |
| M2（基线复现） | **只有代码，没有指标** —— Gate 未达成 |
| M3（双状态核心） | **只有代码 + 冒烟**，Gate 未验证；且已暴露一个可能让核心自我窒息的机制缺陷 |

方案的核心科学主张（P_d ≥ 0.90 约束下 F_a 降 1 个数量级 → 官方总分超最优多帧基线）**至今零证据**：仓库里唯一一次对 `dsld_core` 的端到端评测，`experiments/smoke_dsld_m3c/eval_smoke.json` 显示 `f1: 0.0`、`tp: 0`、官方检测分 `-4757`。

**一句话总评**：M0/M1 可以打高分；M2/M3 目前是"脚手架 + 冒烟"。报告的诚实度高，但若干 headline 数字（配准 RMSE、显存/速度、满分自测）属于自指或缺少仓库内出处。真正该警惕的不是它做错了什么，而是**核心机制在唯一一次有记录的运行里就已处于半失效状态，而 M3 的 Gate 恰好测不出这件事**。

---

## 1. 真正扎实、值得肯定的部分

1. **数据与缓存层是真的。** 逐一核对磁盘：87 个 `reg.npz`、87 个 `quality.npy`，每段 10 类文件齐备（见 `data/cache/ittd/seq_0001/`），`reports/m1/qc/seq_0001.png … seq_0087.png` 共 87 张逐段质检拼图确实存在。报告里的东西在磁盘上真能找到，这一比例在科研项目里并不常见。
2. **评估器写得讲究。** `dsld/eval/official_score.py` 把论文口径的歧义点（同框多点是罚还是不罚）显式登记为 `literal` / `strict` 双实现，并列入 R6 风险回溯条款，而不是偷偷挑一个对自己有利的口径。这是诚实工程的做法。
3. **有真正可证伪的测试，不是凑数。**
   - `tests/test_register.py:65-99`：用合成已知仿射变换反解，断言恢复矩阵的点误差（<1px / <0.5px / FM <2–4px）。
   - `tests/test_map_iou.py:106-125`：用 pycocotools 做独立实现交叉校验 AP50。
   - `tests/test_baselines.py:79-88`：手算 SoftIoU = 25/26 并断言。
   - "87 项单测"这个数字我核过：81 个测试函数 + `test_manifest.py` 的 parametrize 展开，**数字属实**。
4. **一处很容易做错的数学做对了。** 窗口对齐复用同一矩阵：输入侧用 `WARP_INVERSE_MAP`、输出侧用默认正向语义回投（`dsld/eval/infer_seq.py:94-110`）。按 OpenCV 两种 flag 的定义独立推导，**方向正确**（同矩阵两种语义的换算是自洽的）。
5. **报告敢写失败与缺口。**
   - M1 吞吐 Gate ⑥ 写 FAIL（183.4 fps vs ≥200 fps）。
   - M2 把基线冻结、§5 指标表留 `<!-- TRAINING_PENDING -->`。
   - 预训练语料缺失被列为"需用户决策"。**实地确认**：`D:\Datasets` 下确实没有 IRDST-Sim / SIATD / ITSDT-15K / IRDST-Real，而 ITTD（43,587 个文件）与 IRSTD-1k、seg_dataset 均存在 —— 披露准确。
6. **可复现性意图明确**：`environment.yml` 锁版本；每次启动打印配置树 + manifest MD5 + git commit；断点恢复含优化器状态。

---

## 2. 严重问题（按危害排序）

### 2.1 M3 核心机制有"自我窒息"的设计缺陷，且被自家指标文件证实

`gate_head` 零初始化使 α ≡ 0.5，而反馈掩码用**严格大于** 0.5 判定（`dsld/models/liquid_core.py:112` 与 `:151-152`）—— 一个刀刃条件。真实运行一拍就翻过去了：项目自己的 `experiments/smoke_qe/metrics.jsonl` 记录

```
alpha_mean = 0.5003   alpha_p99 = 0.502   alpha_frac_high = 0.94 ~ 0.95
```

即 **95% 的像素以 0.0003 的裕度越过了阈值**。于是掩码 M ≈ 1 几乎铺满全图，而 M 直接去乘输入：

```python
x_b = x_ctx * (1.0 - M)     # ≈ 0
x_t = x_res * (1.0 - M)     # ≈ 0
```

（`dsld/models/liquid_core.py:172-177`）——慢通道 h_B 与目标通道 h_T 在掩码覆盖区**收不到任何输入**。

后果在同一份文件里可见：`loss_recon: 0.0`。而 `dsld/train/losses.py:137-150` 在有效像素为空时**恰好返回 0 且不作任何告警**，背景通道唯一的监督信号静默消失。

M3 报告 §5 把这个现象当成"快评背景集合被 m_tgt 挤空"来修（改成只按 GT 排除）——**修的是症状，不是因**：h_B 拿不到输入这件事仍然在。

### 2.2 几个 Gate / 指标在结构上不可能失败

- **M1 Gate ②「RMSE P95 ≤ 0.5 px」是自指的。** KLT 结果只有在 `rmse <= 0.5` 时才被接受（`dsld/data/preprocess/register.py:262`），被拒的走 FM；而 FM 帧存进 `rmse` 字段的是 `1.0 - quality`，一个**无量纲相关系数代理**（`register.py:266`），随后与像素单位的 RMSE 混进同一个百分位（`scripts/preprocess_all.py:297,316`）。报告"逐段 P95 最大 0.493"正好贴着 0.5 上限——那是**接受门限本身**，不是精度成就。
  > 公平地说：注册算法本身在合成数据上被真测试过（见 §1.3）。弱的是这个**对外汇报的指标**，它无法证伪。
- **官方评分自测 22,418 / 22,418 是满分贯通测试**，验证格式与管道，**不是**对官方程序的保真度（GT 中心点当完美预测，任何"命中 +1、匹配 +1"的实现都会得到同一个数）。代码自己承认歧义未定（R6）。报告写成"严格一致"，语气强于证据。
- **M3 的显存 / 速度表没有代码出处。** 报告引用的 `scripts/bench_dsld_memory.py` 实际只做：在 `eval()` + `no_grad()` 下打印张量范围与 loss 值；**不测显存、不测时间**；硬编码 `sys.path.insert(0, 'D:/DBLIRST')`；其 `use_checkpoint=True` 因 `and self.training` 守卫在 eval 下完全失效。M3 报告的 `10.33 GB / 78 s`、`5.19 GB / 0.78 s` 两行，**在仓库里无法复算**——而"8GB 卡可训"这个结论决定了后续所有训练排期的可行性判断。

### 2.3 τ 不是结构性时间常数，而 Gate 恰恰只测 τ

更新式 `p = σ(−f·dt/τ)` 中 `f` 是自由的 1×1 卷积输出（`dsld/models/liquid_core.py:80`），有效时间尺度是 `f/τ`；网络完全可以学 `f ∝ τ` 来抵消 τ 的硬 clip。因此

> "双状态时间尺度分立假设由结构保证"（`liquid_core.py:5-6`）

是**过度声明**。真正分开两个状态的只有：① 级联结构（h_T 只吃残差 `x − ŷ_B`）；② τ 的**初始化值**（B=48 / T=6）。

而 M3 Gate 监控的正是 `theta_tau` 的分位数——**τ_B ∈ [24,128] 可以轻松 PASS，而动力学上的尺度分离是任意的**。单元测试也只断言 clip 范围，不验证有效速率。

### 2.4 缺失件（且未进 M2 的偏差登记表）

| 缺失项 | 证据 | 影响 |
| --- | --- | --- |
| IDF1 / IDsw / MOTA 从未实现 | 全库无匹配；`motmetrics` 在 `environment.yml` 声明却从未 import | 方案 8.x 把轨迹级指标列为跨论文可比的主表项；M1/M2 报告均承诺"在 crossing / long_occlusion 子集单列" |
| `scripts/eval.py`（方案 9.1⑤"唯一评估入口"）不存在 | `scripts/` 下只有 `eval_baseline.py` / `eval_irstd1k.py` | 审计把它"移交 M2"，M2 已关闭，无下文 |
| DDP 不存在 | grep 仅命中报告文字 | `reports/M3_双状态核心_启动报告.md` §3 给出 2×4090 启动命令，实际只能单卡 |
| EMA 从未计算 | `trainer.py:131` 只把 `ema` 当"附带状态"打印 | 方案 9.1④ 要求 EMA / raw 分开保存 |
| `ncps` 从未 import | `environment.yml` 声明为"CfC 参考实现" | CfC 实现从未与参考实现对照 |

### 2.5 工程卫生

- **工作区里没有 git 仓库**（无 `.git`），而报告把 commit hash（`5a658b4` / `9856f8c` / `84d16d9` / `e9359f9`）当作可复现证据反复引用；`.gitignore` 又排除了 `experiments/` 与 `data/cache/` —— 即**证据产物与缓存都不在版本控制内**。
- **无 README、无 requirements/pyproject、无 CI 配置**（方案 9.1② 明确承诺 CI 冒烟）。
- **硬编码 Windows 绝对路径**：`scripts/bench_dsld_memory.py:2`、`scripts/check_ittd.py:39`、`scripts/probe_distractors.py:23`、`scripts/selftest_official_score.py:23`、`scripts/eval_irstd1k.py:84`，以及 `configs/*.yaml` 中的 `D:/Datasets/...`。而 M3 的实操在 Linux 云端（torch 2.3.1，与锁定的 2.5.1+cu121 **已漂移**）。
- **文档漂移**：
  - `main.py` 的默认配置 `configs/ittd_finetune.yaml` 至今是 `model.type: dryrun` —— 文档里那条"推荐启动"命令不会训练真模型。
  - `environment.yml` 头部注释写"win32 + RTX 4090"，`reports/m0/M0_验收报告.md` 也写 4090，而 M1/M2/M3 报告一致描述实际机器是 RTX 4060 Laptop 8GB —— 同一仓库里硬件事实不一致。
  - `configs/base.yaml:33` 的 `data.T` 是死键（trainer 读 `train.window.T`，`trainer.py:214`）。
- **效率叙事不对称**：产物里 `fps_fullres = 3.1`（`experiments/smoke_dsld_m3c/eval_smoke.json`），报告只宣传"2.96 GFLOPs，是方案 lite 口径的 1/4"，**未提帧率**。对一个声称"机载边缘部署候选"的项目，3.1 FPS 才是决定性数字。成因是 T=32 的 Python 逐帧循环造成延迟受限；液态核本身只占约 2% FLOPs —— 动力学几乎免费，是循环吃掉了吞吐。

---

## 3. 次要缺陷清单（已核对到行）

| # | 缺陷 | 位置 | 后果 |
| --- | --- | --- | --- |
| 1 | 阈值扫描的行标签全部写成 `iou_thr: 0.5`，**变化的置信度阈值未被记录** | `scripts/eval_baseline.py:192` | `eval_smoke.json` 的 7 行 sweep 无法对应回各自阈值，运行点校准不可复现 |
| 2 | 先计算并打印 `[WARN] missing/unexpected`，随后 `load_state_dict(..., strict=True)` | `dsld/train/trainer.py:126` | 任何部分加载直接抛异常；Stage-A 预训练骨干 → ITTD 微调（方案核心流程）会踩 |
| 3 | 早停的 `break` 在断点保存之前 | `trainer.py:649-650` vs `:670` | 早停触发的那一轮**不存盘，最优模型丢失**。现默认 `early_stop=false` 故潜伏，一开启必踩 |
| 4 | 累积梯度的最后一个不满组仍除以 `accum` | `trainer.py:606-616` | 末组更新被系统性缩小 |
| 5 | `metrics.jsonl` 会出现 `steps: 0, loss_avg: 0.0` 的轮记录 | `experiments/smoke_qe/metrics.jsonl` 第 4、7 行 | 易被误读为"loss 收敛到 0" |
| 6 | `cudnn.benchmark=True` + 仅 `manual_seed` | `trainer.py:31`、`:515` | 审计报告"loss 逐位一致（0.3515）"的确定性声明**不被代码保证** |
| 7 | `torch.load(..., weights_only=False)` | `trainer.py:104` | 加载外部 ckpt 时可执行任意代码（自产 ckpt 风险低） |
| 8 | `amp` 用字符串匹配，未知值静默退化为"无 AMP" | `trainer.py:545-547` | 配置拼写错误不报错 |
| 9 | 基线（mshnet / msd3d）训练期无任何验证 | `trainer.py:619` 将 quick_eval 限定 `dsld_core` | 两个基线"盲训"，M2 报告中 MSHNet 仅 1 轮且无指标 |

> **另附两条我最初怀疑、核对后排除的点**，以免误导后续读者：
> 1. **梯度累积是正确的** —— 每次 `optim.step()` 之后都紧跟 `zero_grad()`（`trainer.py:616`），不存在跨整轮累积。
> 2. **LR 调度是正确的** —— `warmup: 0` 时 step 0 取 base lr（λ=1），`experiments/dryrun_m0/metrics.jsonl` 的 `3e-4 → 3e-5` 恰好对应 `iters_total = 16//2 × 1 = 8`，即余弦正确走到 `cosine_min`。

---

## 4. 测试质量评估

**真正的可证伪测试**：合成 GT 配准反解（`tests/test_register.py`）、pycocotools AP 交叉校验（`tests/test_map_iou.py:106-125`，容差 0.05 偏松但方向正确）、手算 SoftIoU（`tests/test_baselines.py:79-88`）、manifest 校验和篡改检测（`tests/test_manifest.py:108`）、掩码值域/膨胀恒等式（`tests/test_dsld_core.py:66-88`）、参数预算（`:202-210`）。

**弱测试 / 不可证伪测试**：

| 测试 | 问题 |
| --- | --- |
| keep 项（`test_dsld_core.py:55-61`） | 只断言 `softplus(θ_β) ≈ 2`，**从不验证作用方向**；改名后仍恒过 |
| "α≡0.5 ⇒ 掩码为 0"（`test_dsld_core.py:91-104`） | 只认证精确算术边角；真实产物显示 94% 像素会翻过阈值，故它不证明鲁棒性 |
| `decouple_loss` 边界（`:186-192`） | 对**它本要防止的坍缩**（h_T ≡ 0 → cos² = 0 → 无惩罚）恒过 |
| fp32 autocast（`:129-138`） | CUDA 分支是空的 `pass`，无实质断言 |
| quick_eval（`tests/test_quick_eval.py:44-81`） | 用未训练模型只验字段存在与确定性；**常量模型也能过** |
| "50k 步无 NaN" Gate | 代码里只是一个 `raise`（`trainer.py:596-598`），**全库没有任何 50k 步运行** |

没有任何测试覆盖：流水线级的配准指标、官方评分对官方程序的对拍、掩码饱和 / `recon_loss=0` 这一失效模式、断点续训的等值性。

---

## 5. 未能验证的事项（评审的边界）

因未执行代码，以下均为**转述而非复核**：

- 测试套件是否真过（"87 项全过"）；
- 参数量 0.111M 与组件拆分（`encoder 0.041 + neck 0.053 + core 0.016`）。总分与 `eval_smoke.json` 的 `params_m = 0.111` 一致，但按代码手工拆算 encoder 一项我算不拢 —— 无论取哪个值，≤5M 闸门都以 30 倍以上余量通过，故不改变结论，但**建议花 30 秒用 `report_model_params` 打一遍确认**；
- 2.96 GFLOPs/帧、3.1 FPS；
- 缓存体积（"13GB"）；
- 全部显存 / 时间数字（含 M3 报告的显存表）。

---

## 6. 建议行动顺序

**① 在砸 GPU-天之前，先修掩码 / 门控的刀刃条件。**
把 α_th 判定改成带死区，或让 gate 偏置在初始化时离开 0，使 α 的初始分布与阈值解耦；或对 M 加上限约束。同时给 `recon_loss` 加"有效像素过少即告警 / 报错"。否则背景通道会在训练早期被饿死，55 小时也是白跑。

**② 重做 τ 的参数化与 Gate 度量。**
让时间常数真正约束动力学（例如对 `f` 施加正定 / 上界，或直接以 `exp(−dt/τ)` 作衰减并对 `f` 做饱和），并把 Gate 从"读 θ_τ"改为"测有效速率 `f·dt/τ` 的分布 + 两通道尺度分离度"。当前 Gate 对核心假设没有鉴别力。

**③ 先把分母补上，再谈结论。**
至少把 MSHNet / T-MSD3D 训到能出 val-int 指标——否则"F_a 降低 X%"没有分母。同时**立刻统一主口径**：现在的中心点官方分与框级 IoU-F1 会讲出相反的故事（同一模型官方记 243 次命中，而 IoU≥0.5 的 `tp = 0`），这个矛盾拖到写论文时会很难看。

**④ 补齐失落的交付件并恢复可复现性**：`scripts/eval.py`、IDF1/IDsw（或明确从主表删除并登记偏差）、DDP 与 EMA（或明确降级为非目标）、把 `experiments/` 与缓存纳入版本控制 / 外部产物登记、补 README 与 CI、清理硬编码路径。

**⑤ 修次要缺陷**（§3 的 1、3、5 优先：sweep 标签、早停丢模型、监控记录可误读）。

---

## 附录：本次评审的证据索引

| 类别 | 文件 |
| --- | --- |
| 方案（承诺基准） | `DSLD双状态液态动力学_方案细化与落地实施方案.md` |
| 里程碑报告 | `reports/m0/M0_验收报告.md`、`reports/m1/M1_验收报告.md`、`reports/m1/qc_report.md`、`reports/M2_基线复现报告.md`、`reports/M3_双状态核心_启动报告.md` |
| 既有审计 | `reports/audit/M0_M1_落地核验报告.md`、`reports/m0/official_score_selftest.md` |
| 核心代码 | `dsld/models/liquid_core.py`、`dsld/models/dsld_core.py`、`dsld/models/encoder_ghostnetv2.py`、`dsld/train/trainer.py`、`dsld/train/losses.py`、`dsld/eval/official_score.py`、`dsld/eval/infer_seq.py`、`dsld/data/preprocess/register.py` |
| 关键产物 | `experiments/smoke_dsld_m3c/eval_smoke.json`、`experiments/smoke_qe/metrics.jsonl`、`experiments/dryrun_m0/metrics.jsonl`、`data/cache/ittd/seq_*/`、`reports/m1/qc/seq_*.png` |
