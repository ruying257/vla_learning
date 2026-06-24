# ACT 调参实验计划

这套实验面向具身智能算法求职与 CAC 论文准备，目标不是堆实验数量，而是证明能够把 ACT 策略从数据、模型训练、离线评估一路分析到闭环部署。

## 实验主线

核心问题：

- 数据质量：固定初始位置和随机初始位置会如何影响泛化。
- 动作块建模：`chunk_size` 太短会短视，太长会增加预测难度。
- 训练稳定性：学习率影响 loss 下降速度、动作误差和闭环抖动。
- 部署平滑：`temporal_ensemble_coeff` 能降低抖动，但过强可能响应变慢。
- 闭环指标：最终结论必须结合任务成功率、完成步数、动作平滑度和失败视频。

实验矩阵在 `experiments/act_tuning/experiments.json` 中维护，结果表模板在 `experiments/act_tuning/results_template.csv`。

## 快速使用

在项目环境中查看实验矩阵：

```bash
python scripts/run_act_experiment.py --list
```

先跑 smoke test：

```bash
python scripts/run_act_experiment.py --exp E0 --phase both --smoke
```

正式跑单组训练：

```bash
python scripts/run_act_experiment.py --exp E0 --phase train
```

正式跑多 seed 闭环部署：

```bash
python scripts/run_act_experiment.py --exp E0 --phase deploy --deploy-trials 5 --deploy-seed-start 0
```

只重建结果表，不重新启动训练或 MuJoCo：

```bash
python scripts/run_act_experiment.py --exp E0 --summarize-only
```

## 闭环评价指标

部署评估以 `4.deploy.py` 为主，`4.deploy_headless.py` 暂不作为本轮实验入口。单次部署可以用环境变量指定 seed：

```bash
ACT_DEPLOY_SEED=1 ACT_DEPLOY_MAX_STEPS=100 python 4.deploy.py
```

`success_rate` 仍沿用严格成功定义：杯子在盘子附近、夹爪打开、末端抬起。为了分析“杯子已经放到盘子上但夹爪没有松开”的情况，部署指标会额外记录 `placement_success_rate` 和 `strict_success_rate`。论文中建议同时报告这两个口径：前者说明放置是否完成，后者说明释放和撤离是否完整。

新增部署指标包括：

- `avg_steps`：所有 seed 的平均执行步数。
- `avg_success_steps`：成功 seed 的平均完成步数。
- `action_smoothness_mean`：相邻动作差分均值，用于衡量动作抖动。
- `prediction_inconsistency_mean`：多个 ACT action chunk 对当前动作预测的平均分歧。
- `final_mug_plate_xy_dist`：结束时马克杯和盘子的 XY 距离。
- `min_mug_plate_xy_dist`：部署过程中马克杯和盘子的最小 XY 距离。

如果出现“杯子已经在盘子上但夹爪未松开”，metrics 会记录：

```json
{
  "placement_success": true,
  "strict_success": false,
  "failure_mode": "placement_success_gripper_closed"
}
```

## 建议实验顺序

第一阶段先做中等规模筛选，观察变量趋势：

```bash
python scripts/run_act_experiment.py --exp E0 --phase train
python scripts/run_act_experiment.py --exp E4 --phase train
python scripts/run_act_experiment.py --exp E6 --phase train
python scripts/run_act_experiment.py --exp E9 --phase train
```

第二阶段对候选 checkpoint 做多 seed 闭环部署：

```bash
python scripts/run_act_experiment.py --exp E4 --phase deploy --deploy-trials 5 --deploy-seed-start 0 --continue-on-fail
```

第三阶段汇总结果并结合视频分析失败类型：

- 数据不足：随机初始位置后轨迹偏移。
- 动作块不合适：过短导致反复修正，过长导致后段漂移。
- 平滑不合适：弱平滑导致抖动，强平滑导致响应慢。
- 夹爪释放失败：`placement_success=true` 但 `strict_success=false`。

## CAC 论文方案

如果目标是基于本项目准备一篇创新性适中的 CAC 论文，可以参考 `CAC_ACT_PAPER_PROPOSAL.md`。当前优先方案包括：

- 预测一致性驱动的自适应时间集成 ACT：主要优化部署阶段动作抖动和纠偏能力。
- 失败样例引导的数据增强 ACT：主要优化训练数据分布覆盖和闭环任务成功率。
