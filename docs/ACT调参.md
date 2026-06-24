# ACT 调参实验计划：面向具身智能算法求职

## Summary

- 目标：用 8 到 12 组小规模实验，证明你不仅“跑过 ACT”，还理解数据质量、动作块建模、视觉输入、训练稳定性和闭环部署之间的关系。
- 主线：离线指标筛选参数，闭环部署验证真实效果，最后用失败样例分析解释为什么某些参数会导致轨迹偏移或动作抖动。
- 默认基线：`datasets/demo_v5_30demos_random`，`chunk_size=50`，`batch_size=64`，`lr=1e-4`，`training_steps=6000`，双视角输入，部署时 `temporal_ensemble_coeff=0.9`。

## Experiment Matrix

- Baseline：
  - E0：默认配置，作为所有实验的对照组。
- 动作块长度：
  - E1：`ACT_CHUNK_SIZE=25`
  - E2：`ACT_CHUNK_SIZE=50`
  - E3：`ACT_CHUNK_SIZE=100`
  - 目的：说明 action chunking 太短会短视，太长会增加预测难度。
- 学习率与训练稳定性：
  - E4：`ACT_LR=5e-5`
  - E5：`ACT_LR=1e-4`
  - E6：`ACT_LR=2e-4`
  - 目的：观察 loss 下降速度、动作误差和闭环抖动。
- 数据规模/数据分布：
  - E7：`ACT_DATASET_ROOT=datasets/demo_v1_3demos_fixed`
  - E8：`ACT_DATASET_ROOT=datasets/demo_v2_50demos_fixed`
  - E9：`ACT_DATASET_ROOT=datasets/demo_v5_30demos_random`
  - 目的：对比少量固定示范、大量固定示范、随机初始位置示范的泛化差异。
- 部署平滑：
  - E10：部署 `temporal_ensemble_coeff=0.7`
  - E11：部署 `temporal_ensemble_coeff=0.9`
  - 目的：解释时间集成如何降低动作抖动，以及过强平滑可能带来的响应滞后。

## Execution Plan

- 每组训练统一保存到独立 checkpoint：
  ```bash
  ACT_CKPT_DIR=./ckpt/exp_E0_baseline python 3.train.py
  ```
- 每组至少记录：
  - 训练配置：数据集、步数、batch size、lr、chunk size、是否双视角。
  - 离线指标：final loss、`Mean action error`、训练耗时。
  - 闭环指标：5 次部署成功次数、平均完成步数、是否明显抖动、是否轨迹偏移。
  - 证据材料：每组保留 1 个部署视频，失败组保留失败视频。
- 先跑 smoke，再跑正式实验：
  ```bash
  ACT_TRAINING_STEPS=1 ACT_BATCH_SIZE=2 ACT_NUM_WORKERS=0 ACT_CKPT_DIR=./ckpt/smoke python 3.train.py
  ACT_DEPLOY_MAX_STEPS=20 ACT_CKPT_DIR=./ckpt/v5 python 4.deploy.py
  ```
- 正式实验建议：
  - 每组先跑 `3000 steps` 做初筛。
  - 选最好的 2 到 3 组补跑到 `6000 steps`。
  - 最终只把表现最好、最差、最有解释价值的 3 组写进简历/项目文档。

## Evaluation And Deliverables

- 产出一个实验表格：
  - columns：`ExpID / Dataset / chunk_size / lr / batch_size / steps / mean_action_error / success_rate / avg_steps / failure_mode / video_path`
- 产出一张结论图：
  - x 轴为实验组，y 轴为 `Mean action error` 和闭环成功率。
- 产出 3 类失败分析：
  - 数据不足：轨迹靠近训练分布，换随机位置后偏移。
  - chunk 不合适：动作块过短导致反复修正，过长导致预测后段漂移。
  - 平滑不合适：平滑弱时抖动，平滑强时响应慢。
- 简历表述建议：
  - “围绕 ACT 策略完成 10+ 组超参数与数据分布对比实验，系统分析 action chunk size、学习率、示范数据随机性和 temporal ensemble 对离线动作误差与闭环成功率的影响。”
  - “通过部署视频和失败样例定位策略轨迹偏移、动作抖动与训练分布不足问题，并据此选择更稳定的训练配置。”

## Assumptions

- 求职方向按“具身智能算法”准备，重点突出 ACT/VLA、模仿学习、闭环评估和失败分析。
- 实验规模控制为小规模可完成，不做大规模自动调参。
- 不改 `mp1` 环境；所有实验使用 WSL2 的 `vla-code` 环境。
- 不把单纯 loss 下降作为最终结论，最终结论必须结合闭环成功率和部署视频。