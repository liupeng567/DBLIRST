# ITTD 划分重平衡对照表（manifest v1 → v2）

- 生成时间：2026-10-05 21:37:32（scripts/build_manifest.py）
- 依据：方案 1.3 v1.6 条款——val-int（65–76）全为傍晚、train-int 无外场傍晚，
  天时×场地两轴同时失衡，直接使用会把早停与阈值校准带偏。

## 调整动作

- 调入 train-int（外场傍晚 ×6）：[71, 72, 73, 74, 75, 76]（取自原 val-int 的 67-76 外场傍晚段）
- 调入 val-int（白天 ×6）：外场 [21, 22, 23] + 内场 [38, 39, 40]
- val-official（77–87）不动，保持只读。

## scene×daytime 构成对照（段数）

| split | scene | daytime | v1 | v2 |
| --- | --- | --- | --- | --- |
| train-int | outdoor | day | 23 | 20 |
| train-int | outdoor | dusk | 0 | 6 |
| train-int | indoor | day | 17 | 14 |
| train-int | indoor | dusk | 24 | 24 |
| val-int | outdoor | day | 0 | 3 |
| val-int | outdoor | dusk | 10 | 4 |
| val-int | indoor | day | 0 | 3 |
| val-int | indoor | dusk | 2 | 2 |
| val-official | outdoor | day | 4 | 4 |
| val-official | outdoor | dusk | 1 | 1 |
| val-official | indoor | day | 3 | 3 |
| val-official | indoor | dusk | 3 | 3 |

## 难点标签覆盖对照（有该难点的段数/总段数）

| split | crossing | occlusion | static_target | distractor_present | motion_blur |
| --- | --- | --- | --- | --- | --- |
| v1 train-int | 9/64 | 39/64 | 56/64 | 36/64 | 21/64 |
| v2 train-int | 10/64 | 34/64 | 53/64 | 36/64 | 19/64 |
| v1 val-int | 1/12 | 4/12 | 6/12 | 6/12 | 1/12 |
| v2 val-int | 0/12 | 9/12 | 9/12 | 6/12 | 3/12 |
| v1 val-official | 1/11 | 7/11 | 8/11 | 6/11 | 3/11 |
| v2 val-official | 1/11 | 7/11 | 8/11 | 6/11 | 3/11 |

## 实例数对照

| split | v1 实例 | v2 实例 |
| --- | --- | --- |
| train-int | 70415 | 66432 |
| val-int | 7550 | 11533 |
| val-official | 11209 | 11209 |

## v2 三子集序列清单

- **train-int**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 71, 72, 73, 74, 75, 76]
- **val-int**: [21, 22, 23, 38, 39, 40, 65, 66, 67, 68, 69, 70]
- **val-official**: [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]
