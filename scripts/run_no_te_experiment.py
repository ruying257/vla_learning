# 运行无 temporal ensemble 的 ACT 对照实验

import argparse
from types import SimpleNamespace

import scripts.run_fgda_experiments as cac


def build_experiment(args):
    """构造固定的无时间集成实验配置。"""
    return {
        "id": "CATE_E0_no_ensemble",
        "suite": "cate",
        "phase_default": "deploy",
        "dataset_root": args.dataset_root,
        "ckpt_dir": args.ckpt_dir,
        "temporal_ensemble_coeff": "none",
        "adaptive_te": False,
        "stage_resampling": False,
        "deploy_only": True,
        "requires_no_ensemble_support": True,
        "notes": "原始 ACT，无 temporal ensemble；按 action chunk 执行无集成对照。",
    }


def runner_args(args):
    """补齐 CAC runner 需要的参数，并为无集成模式选择合理动作步数。"""
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else args.chunk_size
    return SimpleNamespace(
        phase="deploy",
        ckpt_dir=args.ckpt_dir,
        failure_guided_dataset=cac.DEFAULT_FAILURE_GUIDED_DATASET,
        extra_random_dataset=cac.DEFAULT_EXTRA_RANDOM_DATASET,
        training_steps=0,
        batch_size=64,
        chunk_size=args.chunk_size,
        n_action_steps=n_action_steps,
        learning_rate="1e-4",
        num_workers=0,
        log_freq=500,
        deploy_trials=args.deploy_trials,
        deploy_seed_start=args.deploy_seed_start,
        deploy_max_steps=args.deploy_max_steps,
        deploy_cooldown=args.deploy_cooldown,
        adaptive_alpha_min=0.5,
        adaptive_alpha_max=0.95,
        adaptive_lambda=10.0,
        continue_on_fail=args.continue_on_fail,
        summarize_only=args.summarize_only,
        dry_run=args.dry_run,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the no temporal ensemble ACT baseline.")
    parser.add_argument("--ckpt-dir", default=cac.DEFAULT_CATE_CKPT, help="Checkpoint used by the deploy run.")
    parser.add_argument("--dataset-root", default=cac.DEFAULT_BASELINE_DATASET)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Defaults to --chunk-size so the no-ensemble policy executes a full action chunk.",
    )
    parser.add_argument("--deploy-trials", type=int, default=20)
    parser.add_argument("--deploy-seed-start", type=int, default=1)
    parser.add_argument("--deploy-max-steps", type=int, default=400)
    parser.add_argument("--deploy-cooldown", type=float, default=2.0)
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_args = runner_args(args)
    exp = build_experiment(args)

    if not args.dry_run and not cac.supports_deploy_no_ensemble():
        raise SystemExit(f"{cac.DEPLOY_SCRIPT} must support ACT_TEMPORAL_ENSEMBLE_COEFF=none.")

    if args.summarize_only:
        cac.summarize_only([exp], run_args)
        return

    print(f"===== {exp['id']} ({exp['suite']}) =====")
    print(f"no temporal ensemble: ACT_N_ACTION_STEPS={run_args.n_action_steps}")
    cac.run_one(exp, run_args)
    if not args.dry_run:
        print(f"results: {cac.RESULTS_PATH}")


if __name__ == "__main__":
    main()
