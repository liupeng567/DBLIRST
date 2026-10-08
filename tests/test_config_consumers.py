"""config 全键消费者 AST 扫描（M3 v2.0 §8.3 第 13 项，硬约束 L5 的运行期版本）。

规则（写得比"扫一遍字符串"严，也比"逐个手写 allowlist"省）：
  · 取 configs/dsld_core.yaml 的**全部叶子键路径**（列表视为叶子）；
  · 用 AST 解析 dsld/**/*.py 与 scripts/*.py，收集所有**属性访问名**（`cfg.model.liquid` 的
    liquid）与**字符串下标键**（`weights["seg"]`、`node["min"]`）；
  · 叶子键 k（父路径 p）算被消费 ⇽ k 出现在名字集合里，且（p 是根 或 p 的末段也出现）。
    父链检查防的是"随便哪个文件里有个同名变量"造成的假消费——`tau_b.min` 必须真的
    经 `tau_b` 这个节点读进来，否则改名/挪层级就没人发现。

能红方式：往 yaml 加一个没人读的键（例如前版的 `infer.threshold`），或把 build.py 里
某次读取删掉。反向失配（代码读了 yaml 里没有的键）由 OmegaConf 的 struct 模式在
build_model 构造时抛错，本测试不重复管。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIG = REPO / "configs" / "dsld_core.yaml"
SRC_ROOTS = (REPO / "dsld", REPO / "scripts")


def leaf_paths(node, prefix=""):
    """dict 递归展开；list/标量算叶子（crop: [240,320] 整体被 tuple(...) 消费）。"""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out += leaf_paths(v, p)
    else:
        out.append(prefix)
    return out


def consumed_names():
    """生产代码里的属性名 + 字符串下标键（测试文件不算消费者：配置不能只被测试读）。"""
    names = set()
    for root in SRC_ROOTS:
        for f in root.rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if isinstance(n, ast.Attribute):
                    names.add(n.attr)
                elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                        and isinstance(n.slice.value, str):
                    names.add(n.slice.value)
    return names


def unconsumed(leaves, names):
    bad = []
    for path in leaves:
        parts = path.split(".")
        key = parts[-1]
        parent = parts[-2] if len(parts) > 1 else None
        if key not in names:
            bad.append(path)
        elif parent is not None and parent not in names:
            bad.append(f"{path}（键名有同名符号，但父节点 {parent} 从未被读取）")
    return bad


def test_every_config_key_has_a_consumer():
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    leaves = [p for p in leaf_paths(cfg) if p]
    assert leaves, "配置为空——扫描本身失去意义"
    bad = unconsumed(leaves, consumed_names())
    assert not bad, f"死配置键（L5）：{'; '.join(bad)}（共 {len(leaves)} 个叶子键）"


def test_scan_can_actually_fail():
    """自检（判据 ⑦：本测试自己必须可失败）：塞一个没人读的键必须被揪出，真键必须放过。"""
    names = consumed_names()
    assert unconsumed(["model.liquid.brand_new_dead_key"], names) == \
        ["model.liquid.brand_new_dead_key"]
    assert unconsumed(["model.liquid.c_h", "train.window.crop", "train.loss.seg"], names) == []
    assert "accum" not in names, "train.batch.accum 属 P2，本轮配置里不该出现（出现即漏检）"
