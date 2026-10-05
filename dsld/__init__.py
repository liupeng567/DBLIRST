"""DSLD: Dual-State Liquid Dynamics for infrared time-sensitive target detection.

包结构（方案 9.1）：
    dsld.data     数据解析 / manifest / 数据集
    dsld.models   编码器 / 液态核心 / 门控 / 头（M2 起填充）
    dsld.eval     官方评分复刻器 / 指标
    dsld.train    训练循环
    dsld.utils    日志 / 可视化 / 种子
"""

__version__ = "0.1.0"
