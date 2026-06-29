"""
批量评估 replay 微调过程中不同 step 阶段的 ACT checkpoint。

默认行为：
- 依次部署 ckpt/v5_finetune_replay 下的 step_0250、step_0500、
  step_0750 和 step_1000；不重复部署与 step_1000 相同的根目录最终模型。
- 默认 temporal ensemble 系数为 0.30，可修改下方常量，也可用 --te 覆盖。
- 默认运行 seed 1..20，但排除 seed 17，共 19 个 seed。
- 默认使用 EGL 离屏渲染，不打开 MuJoCo 窗口；加 --viewer 才显示窗口。
- 默认保存视频；加 --no-record-video 可关闭视频编码。

常用命令：
  # 查看可选的 checkpoint 阶段
  python scripts/run_finetune_replay_eval.py --list

  # Headless 运行全部四个阶段，使用默认 TE=0.30 和 19 个 seed
  python scripts/run_finetune_replay_eval.py

  # 只运行 250/500 step，关闭 TE，并只部署指定 seed
  python scripts/run_finetune_replay_eval.py \
    --exp step_0250 step_0500 --te none --deploy-seeds 1,5,18

  # 打开 MuJoCo 实时窗口
  python scripts/run_finetune_replay_eval.py --exp step_0250 --viewer

  # 关闭录像并在 shell 后台运行
  nohup python scripts/run_finetune_replay_eval.py --no-record-video \
    > finetune_replay_eval.log 2>&1 &

输出结构：
  experiments/finetune_replay_eval/<TE配置>/<step阶段>/
  每个阶段保存 logs/、metrics/、videos/、seed_results.csv 和 result.csv；
  同一 TE 配置目录下另有全阶段汇总 result.csv。
"""

import argparse
import os
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import run_te as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiments" / "finetune_replay_eval"
DEFAULT_DATASET_ROOT = "./datasets/demo_v5_30demos_random"
DEFAULT_CKPT_ROOT = "./ckpt/v5_finetune_replay"
DEFAULT_TEMPORAL_ENSEMBLE_COEFF = "0.30"
DEFAULT_DEPLOY_SEEDS = [seed for seed in range(1, 21) if seed != 17]
DEFAULT_FORCE_RELEASE_STREAK = 3
DEPLOY_SCRIPT = "deploy.py"

CHECKPOINT_STAGES = [
    ("step_0250", 250),
    ("step_0500", 500),
    ("step_0750", 750),
    ("step_1000", 1000),
]


def parse_te_coeff(value):
    """解析 TE 系数，保留 none 作为关闭 temporal ensemble 的显式值。"""
    value_text = str(value).strip().lower()
    if value_text in {"none", "null"}:
        return "none"
    try:
        coeff = Decimal(value_text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid TE coefficient: {value}") from exc
    if not coeff.is_finite() or coeff < 0:
        raise argparse.ArgumentTypeError("TE coefficient must be a finite non-negative number or 'none'")
    return value_text


def parse_seed_list(value):
    """解析逗号分隔的 seed，去重后保留用户给定的顺序。"""
    seeds = []
    seen = set()
    try:
        items = [item.strip() for item in value.split(",") if item.strip()]
        for item in items:
            seed = int(item)
            if seed < 0:
                raise ValueError
            if seed not in seen:
                seeds.append(seed)
                seen.add(seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--deploy-seeds must be comma-separated non-negative integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--deploy-seeds cannot be empty")
    return seeds


def te_output_id(value):
    """把 TE 系数转为稳定的目录名，例如 0.3 -> TE_030。"""
    if value == "none":
        return "TE_none"
    normalized = format(Decimal(value).normalize(), "f")
    whole, _, fraction = normalized.partition(".")
    fraction = (fraction.rstrip("0") or "0").ljust(2, "0")
    return f"TE_{whole}{fraction}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate replay-finetuned ACT checkpoints with one configurable TE coefficient."
    )
    parser.add_argument("--list", action="store_true", help="List checkpoint stages and exit.")
    parser.add_argument("--exp", nargs="+", help="Checkpoint stage ids to run; omit to run all stages.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument(
        "--te",
        type=parse_te_coeff,
        default=parse_te_coeff(DEFAULT_TEMPORAL_ENSEMBLE_COEFF),
        help="Temporal ensemble coefficient, or 'none'.",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument(
        "--deploy-seeds",
        type=parse_seed_list,
        default=DEFAULT_DEPLOY_SEEDS,
        help="Comma-separated deploy seeds; default is 1..20 excluding 17.",
    )
    parser.add_argument("--deploy-max-steps", type=int, default=500)
    parser.add_argument("--deploy-cooldown", type=float, default=2.0)
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer; default is headless EGL.")
    parser.add_argument("--no-record-video", action="store_true", help="Do not encode deploy videos.")
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def experiment_matrix(args):
    """返回四个微调阶段；每个阶段共用本次命令的 TE 系数。"""
    dataset_root = getattr(args, "dataset_root", DEFAULT_DATASET_ROOT)
    ckpt_root = getattr(args, "ckpt_root", DEFAULT_CKPT_ROOT)
    te_coeff = getattr(args, "te", parse_te_coeff(DEFAULT_TEMPORAL_ENSEMBLE_COEFF))
    experiments = []
    for stage_id, training_step in CHECKPOINT_STAGES:
        experiments.append(
            {
                "id": stage_id,
                "dataset_root": dataset_root,
                "ckpt_dir": str(Path(ckpt_root) / stage_id),
                "temporal_ensemble_coeff": te_coeff,
                "requires_no_ensemble_support": te_coeff == "none",
                "training_step": training_step,
                "notes": f"Replay 微调 {training_step} step checkpoint，TE={te_coeff}。",
            }
        )
    return experiments


def select_experiments(args):
    matrix = experiment_matrix(args)
    if not args.exp:
        return matrix
    wanted = set(args.exp)
    selected = [exp for exp in matrix if exp["id"] in wanted]
    missing = wanted - {exp["id"] for exp in selected}
    if missing:
        raise SystemExit(f"Unknown checkpoint stage: {', '.join(sorted(missing))}")
    return selected


def configure_base_module(args):
    """让通用 TE runner 的汇总函数写入本脚本的独立输出树。"""
    exp_dir = OUTPUT_ROOT / te_output_id(args.te)
    base.EXP_DIR = exp_dir
    base.RESULT_PATH = exp_dir / "result.csv"
    base.experiment_matrix = experiment_matrix
    base.DEPLOY_SCRIPT = DEPLOY_SCRIPT
    base.print_env_delta = print_env_delta


def deploy_env(base_env, exp, args, seed):
    env = dict(base_env)
    base.clean_proxy(env)
    exp_dir = base.experiment_output_dir(exp)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "ACT_DATASET_ROOT": exp["dataset_root"],
            "ACT_CKPT_DIR": exp["ckpt_dir"],
            "ACT_CHUNK_SIZE": str(args.chunk_size),
            "ACT_N_ACTION_STEPS": str(base.resolve_n_action_steps(exp, args)),
            "ACT_TEMPORAL_ENSEMBLE_COEFF": str(exp["temporal_ensemble_coeff"]),
            "ACT_ADAPTIVE_TE": "0",
            "ACT_DEPLOY_MAX_STEPS": str(args.deploy_max_steps),
            "ACT_DEPLOY_SEED": str(seed),
            "ACT_VIDEO_DIR": str(exp_dir / "videos"),
            "ACT_RECORD_VIDEO": "0" if args.no_record_video else "1",
            "ACT_DEPLOY_METRICS_PATH": str(exp_dir / "metrics" / f"deploy_seed{seed}.json"),
            "ACT_FORCE_RELEASE_ON_PLACEMENT": "1",
            "ACT_FORCE_RELEASE_STREAK": str(DEFAULT_FORCE_RELEASE_STREAK),
            "ACT_USE_VIEWER": "1" if args.viewer else "0",
        }
    )
    if args.viewer:
        # viewer 使用 GLFW 窗口，避免继承 headless EGL 设置。
        env.pop("MUJOCO_GL", None)
    else:
        env["MUJOCO_GL"] = "egl"
    return env


def print_env_delta(env):
    """打印会影响本次部署的关键环境变量。"""
    keys = [
        "ACT_DATASET_ROOT",
        "ACT_CKPT_DIR",
        "ACT_CHUNK_SIZE",
        "ACT_N_ACTION_STEPS",
        "ACT_TEMPORAL_ENSEMBLE_COEFF",
        "ACT_DEPLOY_SEED",
        "ACT_RECORD_VIDEO",
        "ACT_USE_VIEWER",
        "MUJOCO_GL",
        "ACT_VIDEO_DIR",
        "ACT_DEPLOY_METRICS_PATH",
    ]
    for key in keys:
        if key in env:
            print(f"  {key}={env[key]}")


def run_one(exp, args):
    """按显式 seed 列表部署单个 checkpoint 阶段。"""
    base.ensure_supported(exp, args.dry_run)
    base.ensure_dataset(exp, args.dry_run)
    base.ensure_checkpoint(exp, args.dry_run)

    base_env = os.environ.copy()
    python_cmd = [sys.executable]
    updated_metric_paths = []

    for index, seed in enumerate(args.deploy_seeds):
        env = deploy_env(base_env, exp, args, seed)
        metrics_path = Path(env["ACT_DEPLOY_METRICS_PATH"])
        before_signature = base.file_signature(metrics_path)
        code = base.run_command(
            python_cmd + [DEPLOY_SCRIPT],
            env,
            base.experiment_output_dir(exp) / "logs" / f"deploy_seed{seed}.log",
            args.dry_run,
        )
        after_signature = base.file_signature(metrics_path)
        if after_signature is not None and after_signature != before_signature:
            updated_metric_paths.append(metrics_path)
        if code != 0:
            print(f"deploy seed {seed} failed with exit code {code}")
            if not args.continue_on_fail:
                base.update_run_outputs(exp, args, updated_metric_paths)
                raise SystemExit(code)
        if index + 1 < len(args.deploy_seeds) and args.deploy_cooldown > 0:
            time.sleep(args.deploy_cooldown)

    base.update_run_outputs(exp, args, updated_metric_paths)


def list_experiments(args):
    for exp in experiment_matrix(args):
        print(f"{exp['id']}: {exp['ckpt_dir']} (TE={exp['temporal_ensemble_coeff']})")


def main():
    args = parse_args()
    configure_base_module(args)
    if args.list:
        list_experiments(args)
        return

    selected = select_experiments(args)
    if args.summarize_only:
        base.summarize_only(selected, args)
        return

    print(f"deploy_seeds: {','.join(str(seed) for seed in args.deploy_seeds)}")
    print(f"viewer: {args.viewer}")
    print(f"record_video: {not args.no_record_video}")
    for exp in selected:
        print(f"===== {exp['id']} =====")
        run_one(exp, args)

    if not args.dry_run:
        print(f"results: {base.RESULT_PATH}")


if __name__ == "__main__":
    main()
