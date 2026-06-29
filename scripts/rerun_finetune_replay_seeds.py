"""
按微调阶段补跑 finetune_replay_eval 中初始化失败的 deploy seed。

使用方式：
1. 在下方 RERUN_SEEDS_BY_EXP 中分别填写每个 step 需要补跑的 seed。
2. 某个 step 没有需要补跑的 seed 时保留空列表 []，脚本会安全跳过。
3. 只需执行一次脚本，会按 step_0250 -> step_1000 的顺序完成所有非空任务。

配置示例：
  RERUN_SEEDS_BY_EXP = {
      "step_0250": [1, 6],
      "step_0500": [],
      "step_0750": [2, 12],
      "step_1000": [8],
  }

常用命令：
  # 查看当前配置，不执行部署
  python scripts/rerun_finetune_replay_seeds.py --list

  # 检查全部补跑命令和输出路径
  python scripts/rerun_finetune_replay_seeds.py --dry-run

  # 默认使用 TE=0.30 和 EGL headless 模式执行全部非空任务
  python scripts/rerun_finetune_replay_seeds.py

  # 只补跑指定 step，并显示 MuJoCo 窗口
  python scripts/rerun_finetune_replay_seeds.py \
    --exp step_0250 step_0750 --viewer

  # 关闭录像并在 shell 后台执行
  nohup python scripts/rerun_finetune_replay_seeds.py --no-record-video \
    > rerun_finetune_replay.log 2>&1 &

输出规则：
- 补跑继续写入 experiments/finetune_replay_eval/<TE配置>/<step阶段>/。
- 同 step/seed 的 log 和 metrics 会被新结果覆盖，视频带时间戳。
- seed_results.csv 只更新本次实际写出 metrics 的 seed，其他 seed 不变。
- 默认单个 seed 失败后继续剩余任务；加 --stop-on-fail 可立即停止。
"""

import argparse

import run_finetune_replay_eval as eval_runner


# 在这里分别填写四个实验需要补跑的 seed；空列表表示跳过该实验。
RERUN_SEEDS_BY_EXP = {
    "step_0250": [6,10],
    "step_0500": [18],
    "step_0750": [2,12,14,20],
    "step_1000": [6,8],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerun configured deploy seeds for replay-finetuned checkpoint stages."
    )
    parser.add_argument("--list", action="store_true", help="List configured rerun seeds and exit.")
    parser.add_argument("--exp", nargs="+", help="Only rerun selected checkpoint stages.")
    parser.add_argument("--dataset-root", default=eval_runner.DEFAULT_DATASET_ROOT)
    parser.add_argument("--ckpt-root", default=eval_runner.DEFAULT_CKPT_ROOT)
    parser.add_argument(
        "--te",
        type=eval_runner.parse_te_coeff,
        default=eval_runner.parse_te_coeff(eval_runner.DEFAULT_TEMPORAL_ENSEMBLE_COEFF),
        help="Temporal ensemble coefficient, or 'none'.",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--deploy-max-steps", type=int, default=500)
    parser.add_argument("--deploy-cooldown", type=float, default=2.0)
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer; default is headless EGL.")
    parser.add_argument("--no-record-video", action="store_true", help="Do not encode rerun videos.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failed deploy process.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_seed_map(seed_map):
    """检查实验名和 seed 类型，空列表是合法配置。"""
    expected_ids = [stage_id for stage_id, _ in eval_runner.CHECKPOINT_STAGES]
    missing_ids = set(expected_ids) - set(seed_map)
    unknown_ids = set(seed_map) - set(expected_ids)
    if missing_ids or unknown_ids:
        details = []
        if missing_ids:
            details.append(f"missing={','.join(sorted(missing_ids))}")
        if unknown_ids:
            details.append(f"unknown={','.join(sorted(unknown_ids))}")
        raise SystemExit(f"RERUN_SEEDS_BY_EXP experiment ids are invalid: {'; '.join(details)}")

    for exp_id in expected_ids:
        seeds = seed_map[exp_id]
        if not isinstance(seeds, list):
            raise SystemExit(f"RERUN_SEEDS_BY_EXP[{exp_id!r}] must be a list")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise SystemExit(f"RERUN_SEEDS_BY_EXP[{exp_id!r}] only accepts non-negative integer seeds")
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"RERUN_SEEDS_BY_EXP[{exp_id!r}] contains duplicate seeds")


def selected_experiments(args):
    """使用主评估脚本的实验矩阵和 exp 合法性检查。"""
    return eval_runner.select_experiments(args)


def list_config(experiments):
    for exp in experiments:
        seeds = RERUN_SEEDS_BY_EXP[exp["id"]]
        seed_text = ",".join(str(seed) for seed in seeds) if seeds else "<empty, skip>"
        print(f"{exp['id']}: seeds={seed_text} ckpt={exp['ckpt_dir']}")


def build_run_args(args, seeds):
    """为单个实验构建独立参数，避免不同 step 共享 seed 列表。"""
    values = vars(args).copy()
    values["deploy_seeds"] = list(seeds)
    values["continue_on_fail"] = not args.stop_on_fail
    return argparse.Namespace(**values)


def main():
    args = parse_args()
    validate_seed_map(RERUN_SEEDS_BY_EXP)
    eval_runner.configure_base_module(args)
    experiments = selected_experiments(args)

    if args.list:
        list_config(experiments)
        return

    task_count = sum(len(RERUN_SEEDS_BY_EXP[exp["id"]]) for exp in experiments)
    if task_count == 0:
        print("No rerun seeds configured; nothing was changed.")
        return

    print(f"rerun_task_count: {task_count}")
    print(f"viewer: {args.viewer}")
    print(f"record_video: {not args.no_record_video}")
    for exp in experiments:
        seeds = RERUN_SEEDS_BY_EXP[exp["id"]]
        if not seeds:
            print(f"===== {exp['id']}: skip (empty seed list) =====")
            continue
        print(f"===== {exp['id']}: seeds={seeds} =====")
        eval_runner.run_one(exp, build_run_args(args, seeds))

    if not args.dry_run:
        print(f"results: {eval_runner.base.RESULT_PATH}")


if __name__ == "__main__":
    main()
