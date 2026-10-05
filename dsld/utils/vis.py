"""可视化公共设置：中文字体（Win: 微软雅黑/黑体）、Agg 后端。"""

from __future__ import annotations


def setup_cjk() -> None:
    """让 matplotlib 正常渲染中文（Windows 自带微软雅黑/黑体）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            import matplotlib.font_manager as fm

            if any(f.name == font for f in fm.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
                break
        except Exception:  # noqa: BLE001
            continue
    plt.rcParams["axes.unicode_minus"] = False
