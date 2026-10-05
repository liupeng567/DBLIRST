"""直连训练脚本（方案 9.1：与 main.py 共用 dsld/train/trainer.py 同一训练循环）。

用法：conda activate Alirst && python scripts/train.py --config dryrun [key=value ...]
供流水线与调试直连；一键启动请用根目录 main.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from dsld.train.trainer import run_training

REPO = Path(__file__).resolve().parents[1]


def load_config(name: str):
    """手动解析 hydra defaults 继承（dryrun.yaml 的 defaults: [base, _self_]）。"""
    cfg_dir = REPO / "configs"
    cfg = OmegaConf.load(cfg_dir / f"{name}.yaml")
    for dep in cfg.get("defaults", []):
        dep_name = dep if isinstance(dep, str) else None
        if dep_name is None or dep_name == "_self_":
            continue
        cfg = OmegaConf.merge(OmegaConf.load(cfg_dir / f"{dep_name}.yaml"), cfg)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="DSLD 直连训练脚本")
    ap.add_argument("--config", default="dryrun", help="configs/ 下的 yaml 名（无后缀）")
    ap.add_argument("overrides", nargs="*", help="OmegaConf 点号覆盖，如 train.seed=1")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    run_training(cfg)


if __name__ == "__main__":
    main()
