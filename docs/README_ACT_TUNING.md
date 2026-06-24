# ACT 调参实验计划

这套实验面向具身智能算法求职，目标不是堆实验数量，而是证明你能把 ACT 策略从数据、模型训练、离线评估一路分析到闭环部署。

## 实验主线

核心问题：

- 数据质量：固定初始位置和随机初始位置会如何影响泛化。
- 动作块建模：`chunk_size` 太短会短视，太长会增加预测难度。
- 训练稳定性：学习率影响 loss 下降速度、动作误差和闭环抖动。
- 部署平滑：`temporal_ensemble_coeff` 能降低抖动，但过强可能响应变慢。

实验矩阵在 `experiments/act_tuning/experiments.json` 中维护，结果表模板在 `experiments/act_tuning/results_template.csv`。

## 快速使用

在 WSL2 中进入项目并激活环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla-code
cd /mnt/d/Desktop/vla_learning
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

查看实验矩阵：

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

正式跑单组闭环部署，默认 5 个 seed：

```bash
python scripts/run_act_experiment.py --exp E0 --phase deploy --deploy-trials 5
```

跑部署平滑对比：

```bash
python scripts/run_act_experiment.py --exp E10 --phase deploy --deploy-trials 5
python scripts/run_act_experiment.py --exp E11 --phase deploy --deploy-trials 5
```

## 输出文件

每次实验会生成：

- `experiments/act_tuning/logs/`：训练和部署日志
- `experiments/act_tuning/metrics/`：机器可读 JSON 指标
- `experiments/act_tuning/videos/`：部署视频证据
- `experiments/act_tuning/results.csv`：自动汇总表

结果表字段：

```text
ExpID / Dataset / chunk_size / lr / batch_size / steps / mean_action_error / final_loss / success_rate / avg_steps / failure_mode / video_path / notes
```

## 建议执行顺序

第一阶段先跑 3000 steps 初筛：

```bash
python scripts/run_act_experiment.py --exp E0 --phase train
python scripts/run_act_experiment.py --exp E1 --phase train
python scripts/run_act_experiment.py --exp E3 --phase train
python scripts/run_act_experiment.py --exp E4 --phase train
python scripts/run_act_experiment.py --exp E6 --phase train
python scripts/run_act_experiment.py --exp E7 --phase train
python scripts/run_act_experiment.py --exp E8 --phase train
python scripts/run_act_experiment.py --exp E9 --phase train
```

第二阶段选 2 到 3 组较好的 checkpoint 做 5 次闭环部署：

```bash
python scripts/run_act_experiment.py --exp E0 --phase deploy --deploy-trials 5
```

第三阶段把最有解释价值的 3 组写进简历：

- 最好组：展示最终成功率和部署视频。
- 最差组：展示失败样例和原因定位。
- 对照组：展示某个变量的单因素影响，例如 `chunk_size` 或数据分布。

## 简历表述

可以写成：

> 围绕 ACT 策略完成 10+ 组超参数与数据分布对比实验，系统分析 action chunk size、学习率、示范数据随机性和 temporal ensemble 对离线动作误差与闭环成功率的影响。

也可以写成：

> 通过部署视频和失败样例定位策略轨迹偏移、动作抖动与训练分布不足问题，并据此选择更稳定的训练配置。

面试时重点讲清楚：离线 `Mean action error` 只能说明模仿误差，最终判断必须回到闭环成功率、完成步数和失败轨迹。

## 结果汇总与异常续跑

如果某个部署 seed 因为 WSLg/GLFW 窗口资源中断，可以让后续 seed 继续跑，并保留已产生的 metrics：

```bash
python scripts/run_act_experiment.py --exp E11 --phase deploy --deploy-trials 5 --continue-on-fail
```

部署前 runner 会清理同一实验同一 seed 的旧 metrics，保证 `results.csv` 反映当前 checkpoint 的最新闭环结果。
部署默认继承训练阶段的数据集、`chunk_size` 和 `n_action_steps`；只有在 `experiments.json` 的 deploy 字段里显式设置时才覆盖，避免 checkpoint 结构或归一化统计不匹配。

如果已经有 `metrics/*.json`，但 `results.csv` 没有更新，可以只重建汇总表，不重新启动训练或 MuJoCo：

```bash
python scripts/run_act_experiment.py --exp E10 --summarize-only
```

生成当前阶段小结和结论图：

```bash
python scripts/summarize_act_results.py
```

本机交互式调参建议先跑中等筛选，避免默认 `batch_size=64` 的 3000-step 训练长时间占用机器：

```bash
python scripts/run_act_medium_screening.py --exps E0 E1 E2 E3 E4 E5 E6 E7 E8 E9 --steps 300 --batch-size 4
```

这轮结果适合回答“变量趋势是什么”，不适合直接宣称最终最优配置。正式结论仍建议选 2 到 3 组补跑更长训练和多 seed 闭环部署。

输出文件：

- `experiments/act_tuning/analysis.md`：求职导向阶段小结、结果表和下一轮实验建议
- `experiments/act_tuning/act_tuning_summary.png`：离线动作误差与闭环成功率对比图

`analysis.md` 会额外生成面试讲法和简历素材，重点区分三类证据：离线动作误差、闭环任务结果、以及 MuJoCo/GLFW 等实验环境异常。

当前建议把 E4 作为深跑主线：它已补到 6000 steps，可用来说明较小学习率带来的离线动作误差收益；再用 E6/E9 作为学习率和随机数据对照组补跑更长训练。

## 追加闭环复测

如果 seed0 的视频已经保留，不想被新一轮部署覆盖，可以从 seed1 开始追加闭环复测：

```bash
python scripts/run_act_experiment.py --exp E4 --phase deploy --deploy-seed-start 1 --deploy-trials 2 --deploy-max-steps 120 --continue-on-fail
```

这条命令会补跑 seed1 和 seed2；runner 仍然会清理同一实验同一 seed 的旧 metrics，保证表格统计的是当前 checkpoint 的复测结果。

## CAC 论文方案

如果目标是基于本项目准备一篇创新性适中的 CAC 论文，可以参考 `CAC_ACT_PAPER_PROPOSAL.md`。该文档给出了两条优先方案：

- 预测一致性驱动的自适应时间集成 ACT：主要优化部署阶段动作抖动和纠偏能力。
- 失败样例引导的数据增强 ACT：主要优化训练数据分布覆盖和闭环任务成功率。
