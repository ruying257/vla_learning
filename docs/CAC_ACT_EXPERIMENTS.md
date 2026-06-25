# CAC ACT 论文实验脚本说明

本文档说明 `scripts/run_cac_paper_experiments.py` 的实验矩阵和运行方式。文档与脚本只负责编排实验，不实现新的 ACT 模型结构。

## 结论先行

- CATE 实验用于比较时间集成策略，必须包含原始 ACT 无时间集成对照。
- FGDA 实验以 `datasets/demo_v5_30demos_random` 为 baseline，主结论来自 `FGDA_E0 -> FGDA_E2 -> FGDA_E3`。
- 当前脚本会保护尚未实现的能力：如果 `4.deploy.py` 不支持关闭 temporal ensemble 或自适应时间集成，对应实验不会正式运行；如果 `3.train.py` 不支持关键阶段重采样，`FGDA_E3` 不会正式训练。

## 快速命令

查看实验矩阵：

```bash
python scripts/run_cac_paper_experiments.py --list
```

只打印 CATE 命令，不启动 MuJoCo：

```bash
python scripts/run_cac_paper_experiments.py --suite cate --dry-run
```

只运行无 temporal ensemble 对照：

```bash
python scripts/run_no_temporal_ensemble_experiment.py --deploy-seed-start 1 --deploy-trials 1
```

只打印 FGDA 命令，不启动训练或 MuJoCo：

```bash
python scripts/run_cac_paper_experiments.py --suite fgda --dry-run
```

运行 FGDA 主实验：

```bash
python scripts/run_cac_paper_experiments.py --suite fgda --phase both --deploy-trials 5
```

如果失败补数据数据集路径不同：

```bash
python scripts/run_cac_paper_experiments.py --suite fgda --failure-guided-dataset datasets/your_dataset
```

## CATE 实验矩阵

| 实验 | 含义 | 当前状态 |
| --- | --- | --- |
| `CATE_E0_no_ensemble` | 原始 ACT，无时间集成 | 通过 `ACT_TEMPORAL_ENSEMBLE_COEFF=none` 关闭时间集成 |
| `CATE_E1b_fixed_03` | 固定时间集成，系数 0.3 | 可编排 |
| `CATE_E1_fixed_07` | 固定时间集成，系数 0.7 | 可编排 |
| `CATE_E2_fixed_09` | 固定时间集成，系数 0.9 | 可编排 |
| `CATE_E3_adaptive_pending` | 自适应时间集成 | pending，需要后续接入 `ACT_ADAPTIVE_TE` |

CATE 默认使用 `./ckpt/act_y`，可通过 `--ckpt-dir` 覆盖。

`scripts/run_no_temporal_ensemble_experiment.py` 是 `CATE_E0_no_ensemble` 的专用入口。它固定 `ACT_TEMPORAL_ENSEMBLE_COEFF=none`，并默认令 `ACT_N_ACTION_STEPS=ACT_CHUNK_SIZE`，让无集成对照按完整 action chunk 执行；如需测试更短动作片段，可显式传 `--n-action-steps` 覆盖。

## FGDA 实验矩阵

| 实验 | 数据策略 | 论文作用 |
| --- | --- | --- |
| `FGDA_E0_v5_baseline` | `datasets/demo_v5_30demos_random` | 原始 baseline |
| `FGDA_E2_failure_guided` | `datasets/demo_v6_failure_guided` | 验证失败案例补数据效果 |
| `FGDA_E3_failure_guided_resampled` | 同 E2 数据集 + `ACT_STAGE_RESAMPLING=1` | 验证补数据 + 关键阶段重采样效果 |

`FGDA_E1_v5_extra_random_optional` 是普通补随机数据的可选对照，不进入主论文结论。需要时加 `--include-optional`。

## 输出文件

所有 CAC 实验输出到：

```text
experiments/cac_act_paper/
```

主要文件包括：

- `<exp_id>/logs/`：单个实验的训练和部署日志。
- `<exp_id>/metrics/`：单个实验的训练指标和每个 seed 的部署指标。
- `<exp_id>/videos/`：单个实验的部署视频。
- `<exp_id>/seed_results.csv`：单个实验内按 seed 增量汇总的关键部署指标。
- `results.csv`：所有实验共享的论文实验汇总表。

示例结构：

```text
experiments/cac_act_paper/CATE_E2_fixed_09/
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

`seed_results.csv` 每一行对应一个 deploy seed，字段包含：

```text
seed / executed_steps / success / error / action_smoothness_mean /
action_smoothness_max / prediction_inconsistency_mean /
prediction_inconsistency_max
```

`seed_results.csv` 只在实际 deploy 写出 metrics 后按 seed 增量更新：补跑单个 seed 时会替换该 seed 行，新 seed 会追加到表中，未重跑或未产生 metrics 的 seed 保持不变。全局 `results.csv` 仍会基于当前可读取的 metrics 汇总更新。

`--summarize-only` 只刷新全局 `results.csv`，不会改动 `seed_results.csv`，避免误刷新未重跑的 seed 行。

## 论文表述边界

- 在 `CATE_E3_adaptive_pending` 未真正接入部署逻辑前，不要声称已经完成自适应时间集成实验。
- 在 `FGDA_E3_failure_guided_resampled` 未真正接入采样器前，不要声称已经完成关键阶段重采样实验。
- FGDA 主对比应写成：原始 v5 baseline、失败案例补数据、失败案例补数据 + 关键阶段重采样。
