# manifest v2 → v3 统计字段回填（M1-6）

- v3 checksum: `md5:f25e2dbaa469394e1fecf584a1bf6491`（splits 与 v2 完全一致，仅回填统计字段）
- strong_clutter 阈值（前 25%）：clutter_density ≥ 2.51 个/千像素
- mean SCR 中位数：2.6；reg RMSE 均值最大：0.350px

| 标签 | 覆盖段数 | 口径 |
| --- | --- | --- |
| strong_clutter | 21 | clutter_density 前 25% |
| long_occlusion | 13 | 轨迹连续消失 ≥20 帧 |
| small_target | 0 | 对角线<8px 实例占比 >50% |
| motion_blur(并集) | 25 | 官方晃动 ∪ 自动模糊占比>10% |

图表：reports/m1/manifest_v3_backfill.png