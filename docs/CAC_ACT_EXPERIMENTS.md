# CAC ACT 论文实验脚本说明

本文档说明 `scripts/run_cac_paper_experiments.py` 的实验矩阵和运行方式。该脚本只负责使用已训练好的 checkpoint 做 deploy sweep，不再负责训练。

## 结论先行

- 训练由 `3.train_finetune.py` 独立完成，runner 只做 deploy。
- 当前实验设计是在 FGDA 新训练 checkpoint 上比较不同 temporal ensemble 参数。
- 保留 `none` 作为无 temporal ensemble 对照；不再包含 Consistency-Aware Temporal Ensemble 或关键阶段重采样实验。

## 快速命令

查看实验矩阵：

```bash
python scripts/run_cac_paper_experiments.py --list
```

只打印 deploy 命令，不启动 MuJoCo：

```bash
python scripts/run_cac_paper_experiments.py --dry-run
```

用默认 FGDA checkpoint 跑 5 个 seed：

```bash
python scripts/run_cac_paper_experiments.py --deploy-seed-start 1 --deploy-trials 5
```

如果 FGDA checkpoint 或数据集路径不同：

```bash
python scripts/run_cac_paper_experiments.py \
  --ckpt-dir ckpt/v5_finetune_new_data \
  --failure-guided-dataset failure_seed_data
```

如果批量 deploy 中某些 seed 因 MuJoCo 初始化异常失败，可以只补跑指定 seed：

```bash
python scripts/rerun_cac_deploy_seeds.py --exp FGDA_TE_090 --deploy-seeds 3
python scripts/rerun_cac_deploy_seeds.py --exp FGDA_TE_090 --deploy-seeds 3,7,11 --continue-on-fail
python scripts/rerun_cac_deploy_seeds.py --exp FGDA_TE_001 FGDA_TE_090 --deploy-seeds 3 --dry-run
```

补跑脚本要求显式传 `--exp`，只运行 deploy；成功写出 metrics 的 seed 会增量更新对应实验目录下的 `seed_results.csv`，并刷新全局 `results.csv`。

## FGDA TE 实验矩阵

| 实验 | 含义 | 当前状态 |
| --- | --- | --- |
| `FGDA_TE_none` | 无 temporal ensemble | 验证 TE-off 对照 |
| `FGDA_TE_001` | 固定时间集成，系数 0.01 | 接近原始 ACT 推荐的轻量平滑 |
| `FGDA_TE_005` | 固定时间集成，系数 0.05 | 轻平滑 sweep |
| `FGDA_TE_010` | 固定时间集成，系数 0.10 | 中低强度平滑 |
| `FGDA_TE_015` | 固定时间集成，系数 0.15 | 细化 0.10 到 0.30 区间 |
| `FGDA_TE_020` | 固定时间集成，系数 0.20 | 细化 0.10 到 0.30 区间 |
| `FGDA_TE_030` | 固定时间集成，系数 0.30 | 低系数 sweep 上界 |
| `FGDA_TE_070` | 固定时间集成，系数 0.70 | 强平滑对照 |
| `FGDA_TE_090` | 固定时间集成，系数 0.90 | 当前强平滑基线 |

默认 checkpoint 为 `./ckpt/v5_finetune_new_data`，默认数据集标识为 `failure_seed_data`，均可通过命令行覆盖。

## 输出文件

所有 CAC 实验输出到：

```text
experiments/cac_act_paper/
```

主要文件包括：

- `<exp_id>/logs/`：单个实验的部署日志。
- `<exp_id>/metrics/`：单个实验内每个 seed 的部署指标。
- `<exp_id>/videos/`：单个实验的部署视频。
- `<exp_id>/seed_results.csv`：单个实验内按 seed 增量汇总的关键部署指标。
- `results.csv`：所有实验共享的论文实验汇总表。

示例结构：

```text
experiments/cac_act_paper/FGDA_TE_090/
  logs/
    deploy_seed1.log
  metrics/
    deploy_seed1.json
  seed_results.csv
  videos/
```

结果表字段包含：

```text
exp_id / suite / dataset / ckpt_dir / temporal_ensemble_coeff /
adaptive_te / stage_resampling / steps / mean_action_error / final_loss /
success_rate / placement_success_rate / strict_success_rate /
avg_steps / avg_success_steps / action_smoothness_mean /
prediction_inconsistency_mean / final_mug_plate_xy_dist /
min_mug_plate_xy_dist / failure_mode / video_path / notes
```

`adaptive_te` 和 `stage_resampling` 是兼容旧表格的保留列；当前 FGDA TE sweep 中两列均为 `0`。

`seed_results.csv` 每一行对应一个 deploy seed，字段包含：

```text
seed / executed_steps / success / error / action_smoothness_mean /
action_smoothness_max / prediction_inconsistency_mean /
prediction_inconsistency_max
```

`seed_results.csv` 只在实际 deploy 写出 metrics 后按 seed 增量更新：补跑单个 seed 时会替换该 seed 行，新 seed 会追加到表中，未重跑或未产生 metrics 的 seed 保持不变。全局 `results.csv` 仍会基于当前可读取的 metrics 汇总更新。

`--summarize-only` 只刷新全局 `results.csv`，不会改动 `seed_results.csv`，避免误刷新未重跑的 seed 行。

## 论文表述边界

- 当前 runner 不再执行训练；训练过程和 WandB 记录应引用 `3.train_finetune.py`。
- 当前 runner 不包含 Consistency-Aware Temporal Ensemble 或关键阶段重采样实验。
- FGDA deploy 结论应写成：同一个失败引导微调 checkpoint 在不同 temporal ensemble 参数下的闭环表现对比。
