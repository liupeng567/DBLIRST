"""DSLD 一键启动入口（方案 9.1 规范 / 附录 A.1）。

推荐启动：conda activate Alirst && python main.py --config-name ittd_finetune seed=0
M0 验收 dry-run：conda activate Alirst && python main.py --config-name dryrun

启动四步（任一步失败即终止并打印原因）：
  ① 打印生效参数配置（OmegaConf 树 + manifest MD5 + git commit + GPU/版本）
  ② 打印模型权重加载情况（逐模块参数量 / matched·missing·unexpected / 形状不符拒绝启动）
  ③④ 按轮训练（tqdm 双进度条）+ 轮摘要 + metrics.jsonl + 按轮断点
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dsld.train.trainer import run_training


@hydra.main(config_path="configs", config_name="ittd_finetune", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
