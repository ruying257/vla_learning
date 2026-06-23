import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]


def run_one(exp_id, steps, batch_size, log_freq, num_workers):
    """按统一预算跑一组中等规模筛选，便于横向比较超参数影响。"""
    cmd = [
        sys.executable,
        "scripts/run_act_experiment.py",
        "--exp",
        exp_id,
        "--phase",
        "train",
        "--training-steps",
        str(steps),
        "--log-freq",
        str(log_freq),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    print(f"===== {exp_id} medium screening =====", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description="Run local-friendly ACT medium screening.")
    parser.add_argument("--exps", nargs="+", default=DEFAULT_EXPERIMENTS, help="Experiment ids to run.")
    parser.add_argument("--steps", type=int, default=300, help="Training steps per experiment.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per experiment.")
    parser.add_argument("--log-freq", type=int, default=50, help="Training log interval.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    args = parser.parse_args()

    for exp_id in args.exps:
        code = run_one(exp_id, args.steps, args.batch_size, args.log_freq, args.num_workers)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
