# CATE 部署侧轻量模块说明

## 1. 核心思想

CATE，全称 Consistency-Aware Temporal Ensemble，中文可写作“预测一致性驱动的自适应时间集成”。它不是重新训练 ACT，也不是修改网络结构，而是在 ACT policy 输出 action chunk 之后、MuJoCo 执行动作之前，增加一个轻量部署后处理模块。

从第一性原理看，ACT 每一步会预测未来一段 action chunk。不同历史 chunk 对同一个当前时刻都会给出动作预测。如果这些预测接近，说明策略对当前动作比较确定，可以更强地沿用历史规划，让动作更稳定；如果这些预测差异大，说明策略处在不确定或需要纠偏的状态，应降低历史惯性，让较新的预测更快参与控制。

## 2. 符号定义

设当前部署时刻为 `t`。

- `K_t`：当前缓存中仍覆盖时刻 `t` 的历史 action chunk 数量。
- `a_t^i`：第 `i` 个历史 action chunk 对当前时刻 `t` 的动作预测。
- `i=0`：最旧的 chunk 对当前时刻的预测。
- `i=K_t-1`：最新的 chunk 对当前时刻的预测。
- `u_t`：当前时刻的预测不一致性。
- `c_t`：由不一致性映射得到的置信度。
- `beta_t`：当前时刻的自适应 temporal ensemble 系数。
- `a_t`：最终送入 MuJoCo 的执行动作。

## 3. 预测不一致性

先收集所有仍覆盖当前时刻的预测：

```text
P_t = {a_t^0, a_t^1, ..., a_t^(K_t-1)}
```

计算这些预测的均值：

```text
mean(a_t) = (1 / K_t) * sum_i(a_t^i)
```

用平均 L2 距离衡量预测分歧：

```text
u_t = (1 / K_t) * sum_i(||a_t^i - mean(a_t)||_2)
```

含义：

- `u_t` 小：多个 chunk 对当前动作意见一致，策略输出稳定。
- `u_t` 大：多个 chunk 分歧明显，策略可能需要纠偏或处在不确定状态。

## 4. 自适应系数

将不一致性映射为置信度：

```text
c_t = exp(-lambda * u_t)
```

再映射到 temporal ensemble 系数：

```text
beta_t = beta_min + c_t * (beta_max - beta_min)
```

推荐默认值：

| 超参数 | 推荐值 | 说明 |
| --- | --- | --- |
| `beta_min` | `0.01` | 预测分歧大时降低历史惯性，避免动作过度锁死 |
| `beta_max` | `0.30` | 预测一致时加强历史规划权重，提高稳定性 |
| `lambda` | `2.0` | 控制 `u_t` 到置信度的映射强度 |
| `window_size` | `10` | 最多保留 10 个历史 chunk 参与融合 |

注意：这里的 `beta_t` 应接近 LeRobot 原生 temporal ensemble 的系数语义，不建议直接使用 `0.5 ~ 0.95` 这类较大的平滑范围。

## 5. 时间集成公式

历史预测按从旧到新排列：

```text
[a_t^0, a_t^1, ..., a_t^(K_t-1)]
```

使用指数权重：

```text
w_i = exp(-beta_t * i)
```

归一化：

```text
W_i = w_i / sum_j(w_j)
```

最终动作：

```text
a_t = sum_i(W_i * a_t^i)
```

该权重方向与 LeRobot 原生 temporal ensemble 保持一致：正的系数会让较旧 chunk 的规划拥有更高权重。这样可以保留 ACT action chunk 的跨步规划信息，避免每一步都过度偏向最新 chunk 的第 0 帧。

## 6. 为什么不直接偏向最新 chunk

当前任务的 action 是 7 维绝对控制量，主要是关节角和夹爪状态。最新 chunk 的第 0 帧通常接近当前姿态。如果自适应模块在预测分歧时过度偏向最新 chunk，第 0 帧会不断把目标拉回当前位置附近，表现为机械臂前几秒有动作，随后动作逐渐变小甚至停住。

因此 CATE 的目标不是“越不确定越完全相信最新 chunk”，而是“降低历史惯性，让新预测更多参与，但仍保留 action chunk 的跨步规划”。

## 7. 工程默认配置

推荐环境变量：

```bash
ACT_ADAPTIVE_TE=1
ACT_ADAPTIVE_ALPHA_MIN=0.01
ACT_ADAPTIVE_ALPHA_MAX=0.30
ACT_ADAPTIVE_LAMBDA=2.0
ACT_ADAPTIVE_WINDOW_SIZE=10
```

对应 runner 参数：

```bash
python scripts/run_cate_adaptive_experiments.py \
  --exp CATE_E3_adaptive \
  --deploy-seed-start 1 \
  --deploy-trials 1 \
  --adaptive-alpha-min 0.01 \
  --adaptive-alpha-max 0.30 \
  --adaptive-lambda 2.0 \
  --adaptive-window-size 10
```

## 8. 对照实验配置

建议固定 checkpoint 和数据集，只比较部署策略：

| 实验 | 部署策略 | 关键配置 |
| --- | --- | --- |
| `CATE_E0_no_ensemble` | 无 temporal ensemble | `ACT_TEMPORAL_ENSEMBLE_COEFF=none` |
| `CATE_E1b_fixed_03` | 固定弱平滑 | `ACT_TEMPORAL_ENSEMBLE_COEFF=0.3` |
| `CATE_E1_fixed_07` | 固定中等平滑 | `ACT_TEMPORAL_ENSEMBLE_COEFF=0.7` |
| `CATE_E2_fixed_09` | 固定强平滑 | `ACT_TEMPORAL_ENSEMBLE_COEFF=0.9` |
| `CATE_E3_adaptive` | CATE 自适应平滑 | `ACT_ADAPTIVE_TE=1` |

固定对照组应显式关闭自适应：

```bash
ACT_ADAPTIVE_TE=0
```

## 9. 评价指标

核心指标：

- 成功率：部署成功次数 / 总 seed 数。
- 平均完成步数：成功样本的平均执行 step。
- 动作平滑度：`mean(||a_t - a_(t-1)||_2)`。
- 预测不一致性：`mean(u_t)`。
- 自适应系数均值：`mean(beta_t)`。
- 末端或杯子到盘子的最终距离和最小距离。

建议在 metrics 中记录：

```text
prediction_inconsistency_mean
prediction_inconsistency_max
adaptive_alpha_mean
adaptive_alpha_min_observed
adaptive_alpha_max_observed
action_smoothness_mean
action_smoothness_max
```

## 10. 排查建议

如果出现“前几秒机械臂会动，后面突然不动”：

1. 检查 `action_smoothness_mean` 是否低到 `0.001` 量级。
2. 检查 `window_size` 是否过大，例如 50。
3. 检查权重是否错误地让最新 chunk 权重最大。
4. 先尝试 `window_size=10`、`lambda=2.0`。
5. 如果仍不动，再检查 `beta_min/beta_max` 是否过大。

推荐从以下配置开始：

```text
beta_min = 0.01
beta_max = 0.30
lambda = 2.0
window_size = 10
```

