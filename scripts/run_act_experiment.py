import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "act_tuning"
CONFIG_PATH = EXP_DIR / "experiments.json"
RESULTS_PATH = EXP_DIR / "results.csv"
RESULT_FIELDS = [
    "exp_id",
    "dataset",
    "chunk_size",
    "lr",
    "batch_size",
    "steps",
    "mean_action_error",
    "final_loss",
    "success_rate",
    "placement_success_rate",
    "strict_success_rate",
    "avg_steps",
    "avg_success_steps",
    "action_smoothness_mean",
    "prediction_inconsistency_mean",
    "final_mug_plate_xy_dist",
    "min_mug_plate_xy_dist",
    "failure_mode",
    "video_path",
    "notes",
]


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_experiment(config, exp_id):
    for exp in config["experiments"]:
        if exp["id"] == exp_id:
            return exp
    raise SystemExit(f"Unknown experiment id: {exp_id}")


def merged_config(defaults, exp, phase):
    # 每组实验只覆盖自己关心的变量，其余继承 defaults，保证对照组清晰。
    merged = dict(defaults)
    merged.update(exp.get(phase, {}))
    return merged


def checkpoint_dir(exp):
    return f"./ckpt/exp_{exp['id']}_{exp['name']}"


def clean_proxy(env):
    # 调参实验默认不使用代理，优先走国内镜像或本地缓存。
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)


def run_command(cmd, env, log_path, dry_run=False):
    print(" ".join(cmd))
    print(f"log: {log_path}")
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def train_env(base_env, exp, train_cfg, smoke=False):
    env = dict(base_env)
    clean_proxy(env)
    ckpt_dir = train_cfg.get("ckpt_dir", checkpoint_dir(exp))
    metrics_path = EXP_DIR / "metrics" / f"{exp['id']}_train.json"

    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "ACT_DATASET_ROOT": str(train_cfg["dataset_root"]),
            "ACT_CKPT_DIR": str(ckpt_dir),
            "ACT_CHUNK_SIZE": str(train_cfg["chunk_size"]),
            "ACT_N_ACTION_STEPS": str(train_cfg["n_action_steps"]),
            "ACT_LR": str(train_cfg["learning_rate"]),
            "ACT_BATCH_SIZE": str(train_cfg["batch_size"]),
            "ACT_NUM_WORKERS": str(train_cfg["num_workers"]),
            "ACT_LOG_FREQ": str(train_cfg["log_freq"]),
            "ACT_TRAINING_STEPS": str(train_cfg["training_steps"]),
            "ACT_METRICS_PATH": str(metrics_path),
        }
    )
    if smoke:
        env.update(
            {
                "ACT_TRAINING_STEPS": "1",
                "ACT_LOG_FREQ": "1",
                "ACT_BATCH_SIZE": "2",
                "ACT_NUM_WORKERS": "0",
            }
        )
    return env


def deploy_env(base_env, exp, deploy_cfg, train_cfg, seed, smoke=False):
    env = dict(base_env)
    clean_proxy(env)

    ckpt_dir = deploy_cfg.get("ckpt_dir") or train_cfg.get("ckpt_dir") or checkpoint_dir(exp)
    metrics_path = EXP_DIR / "metrics" / f"{exp['id']}_deploy_seed{seed}.json"
    video_dir = EXP_DIR / "videos" / exp["id"]
    deploy_overrides = exp.get("deploy", {})

    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "ACT_DATASET_ROOT": str(deploy_overrides.get("dataset_root", train_cfg["dataset_root"])),
            "ACT_CKPT_DIR": str(ckpt_dir),
            "ACT_CHUNK_SIZE": str(deploy_overrides.get("chunk_size", train_cfg["chunk_size"])),
            "ACT_N_ACTION_STEPS": str(deploy_overrides.get("n_action_steps", train_cfg["n_action_steps"])),
            "ACT_TEMPORAL_ENSEMBLE_COEFF": str(deploy_cfg["temporal_ensemble_coeff"]),
            "ACT_DEPLOY_MAX_STEPS": str(deploy_cfg["deploy_max_steps"]),
            "ACT_DEPLOY_SEED": str(seed),
            "ACT_VIDEO_DIR": str(video_dir),
            "ACT_RECORD_VIDEO": "1",
            "ACT_DEPLOY_METRICS_PATH": str(metrics_path),
        }
    )
    if smoke:
        env.update({"ACT_DEPLOY_MAX_STEPS": "20", "ACT_RECORD_VIDEO": "0"})
    return env


def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_failed_deploy_metrics(env, return_code):
    """部署进程 native 崩溃时由 runner 兜底写指标，避免结果表丢失失败原因。"""
    metrics_path = Path(env["ACT_DEPLOY_METRICS_PATH"])
    if metrics_path.exists():
        return
    failure_mode = "native_crash" if return_code < 0 else "command_failed"
    metrics = {
        "dataset_root": env["ACT_DATASET_ROOT"],
        "ckpt_dir": env["ACT_CKPT_DIR"],
        "xml_path": env.get("ACT_XML_PATH", "./mode/demo_scene.xml"),
        "video_path": "",
        "record_video": env.get("ACT_RECORD_VIDEO") == "1",
        "deploy_seed": int(env["ACT_DEPLOY_SEED"]),
        "max_steps": int(env["ACT_DEPLOY_MAX_STEPS"]),
        "executed_steps": 0,
        "success": False,
        "strict_success": False,
        "placement_success": False,
        "failure_mode": failure_mode,
        "status": "command_failed",
        "return_code": return_code,
        "chunk_size": int(env["ACT_CHUNK_SIZE"]),
        "n_action_steps": int(env["ACT_N_ACTION_STEPS"]),
        "temporal_ensemble_coeff": float(env["ACT_TEMPORAL_ENSEMBLE_COEFF"]),
        "action_smoothness_mean": None,
        "prediction_inconsistency_mean": None,
        "final_mug_plate_xy_dist": None,
        "min_mug_plate_xy_dist": None,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def summarize_result(config, exp):
    defaults = config["defaults"]
    train_cfg = merged_config(defaults, exp, "train")
    deploy_cfg = merged_config(defaults, exp, "deploy")
    train_metrics = read_json(EXP_DIR / "metrics" / f"{exp['id']}_train.json")
    deploy_metrics = sorted((EXP_DIR / "metrics").glob(f"{exp['id']}_deploy_seed*.json"))
    deploy_payloads = [read_json(path) for path in deploy_metrics]
    deploy_payloads = [payload for payload in deploy_payloads if payload]

    successes = [payload["success"] for payload in deploy_payloads]
    success_rate = "" if not successes else f"{sum(successes) / len(successes):.2f}"
    placement_successes = [payload.get("placement_success", payload.get("success", False)) for payload in deploy_payloads]
    strict_successes = [payload.get("strict_success", payload.get("success", False)) for payload in deploy_payloads]
    placement_success_rate = "" if not placement_successes else f"{sum(placement_successes) / len(placement_successes):.2f}"
    strict_success_rate = "" if not strict_successes else f"{sum(strict_successes) / len(strict_successes):.2f}"
    avg_steps = "" if not deploy_payloads else f"{sum(p['executed_steps'] for p in deploy_payloads) / len(deploy_payloads):.1f}"
    success_step_payloads = [
        payload
        for payload in deploy_payloads
        if payload.get("strict_success", payload.get("success", False))
        or payload.get("placement_success", False)
    ]
    avg_success_steps = "" if not success_step_payloads else f"{sum(p['executed_steps'] for p in success_step_payloads) / len(success_step_payloads):.1f}"
    action_smoothness_values = [
        payload["action_smoothness_mean"]
        for payload in deploy_payloads
        if payload.get("action_smoothness_mean") is not None
    ]
    prediction_inconsistency_values = [
        payload["prediction_inconsistency_mean"]
        for payload in deploy_payloads
        if payload.get("prediction_inconsistency_mean") is not None
    ]
    final_xy_values = [
        payload["final_mug_plate_xy_dist"]
        for payload in deploy_payloads
        if payload.get("final_mug_plate_xy_dist") is not None
    ]
    min_xy_values = [
        payload["min_mug_plate_xy_dist"]
        for payload in deploy_payloads
        if payload.get("min_mug_plate_xy_dist") is not None
    ]
    video_path = ""
    for payload in deploy_payloads:
        if payload.get("video_path"):
            video_path = payload["video_path"]
            break
    failure_modes = []
    for payload in deploy_payloads:
        mode = payload.get("failure_mode", "")
        if not mode and payload.get("success") is False and payload.get("max_steps") == payload.get("executed_steps"):
            mode = "max_steps"
        if not mode and payload.get("success") is True:
            mode = "success"
        if mode:
            failure_modes.append(mode)
    failure_modes = sorted(set(failure_modes))
    failure_mode = "+".join(failure_modes) if failure_modes else "manual_review"

    dataset = train_metrics.get("dataset_root", train_cfg["dataset_root"]) if train_metrics else train_cfg["dataset_root"]
    chunk_size = train_metrics.get("chunk_size", train_cfg["chunk_size"]) if train_metrics else train_cfg["chunk_size"]
    lr = train_metrics.get("learning_rate", train_cfg["learning_rate"]) if train_metrics else train_cfg["learning_rate"]
    batch_size = train_metrics.get("batch_size", train_cfg["batch_size"]) if train_metrics else train_cfg["batch_size"]

    return {
        "exp_id": exp["id"],
        "dataset": dataset,
        "chunk_size": chunk_size,
        "lr": lr,
        "batch_size": batch_size,
        "steps": train_metrics.get("training_steps", train_cfg["training_steps"]) if train_metrics else ("" if exp.get("deploy_only") else train_cfg["training_steps"]),
        "mean_action_error": "" if not train_metrics else f"{train_metrics['mean_action_error']:.4f}",
        "final_loss": "" if not train_metrics else f"{train_metrics['final_loss']:.4f}",
        "success_rate": success_rate,
        "placement_success_rate": placement_success_rate,
        "strict_success_rate": strict_success_rate,
        "avg_steps": avg_steps,
        "avg_success_steps": avg_success_steps,
        "action_smoothness_mean": "" if not action_smoothness_values else f"{sum(action_smoothness_values) / len(action_smoothness_values):.4f}",
        "prediction_inconsistency_mean": "" if not prediction_inconsistency_values else f"{sum(prediction_inconsistency_values) / len(prediction_inconsistency_values):.4f}",
        "final_mug_plate_xy_dist": "" if not final_xy_values else f"{sum(final_xy_values) / len(final_xy_values):.4f}",
        "min_mug_plate_xy_dist": "" if not min_xy_values else f"{sum(min_xy_values) / len(min_xy_values):.4f}",
        "failure_mode": failure_mode,
        "video_path": video_path,
        "notes": exp["question"],
    }


def upsert_result(row):
    rows = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    replaced = False
    for idx, old in enumerate(rows):
        if old["exp_id"] == row["exp_id"]:
            rows[idx] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    rows.sort(key=lambda item: int(item["exp_id"].lstrip("E")) if item.get("exp_id", "").startswith("E") else 999)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def list_experiments(config):
    for exp in config["experiments"]:
        phase = "deploy" if exp.get("deploy_only") else "train"
        print(f"{exp['id']}: {exp['name']} [{phase}] - {exp['question']}")


def main():
    parser = argparse.ArgumentParser(description="Run ACT tuning experiments.")
    parser.add_argument("--list", action="store_true", help="List experiment matrix and exit.")
    parser.add_argument("--exp", help="Experiment id, e.g. E0.")
    parser.add_argument("--phase", choices=["train", "deploy", "both"], default="train")
    parser.add_argument("--training-steps", type=int, help="Override training steps for quick screening.")
    parser.add_argument("--batch-size", type=int, help="Override training batch size.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--log-freq", type=int, help="Override training log frequency.")
    parser.add_argument("--deploy-max-steps", type=int, help="Override deploy max steps.")
    parser.add_argument("--deploy-trials", type=int, default=None)
    parser.add_argument("--deploy-seed-start", type=int, default=0, help="First deploy seed to run.")
    parser.add_argument("--deploy-cooldown", type=float, default=2.0, help="Seconds to wait between deploy trials.")
    parser.add_argument("--continue-on-fail", action="store_true", help="Keep summarizing later trials after one deploy command fails.")
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild results.csv from existing metrics.")
    parser.add_argument("--smoke", action="store_true", help="Override to 1 train step and 20 deploy steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    config = load_config()
    if args.list:
        list_experiments(config)
        return
    if not args.exp:
        raise SystemExit("Please pass --exp E0, or use --list.")

    exp = find_experiment(config, args.exp)
    defaults = config["defaults"]
    train_cfg = merged_config(defaults, exp, "train")
    deploy_cfg = merged_config(defaults, exp, "deploy")
    if args.summarize_only:
        # 只重建实验表，不重新启动训练或 MuJoCo 部署。
        row = summarize_result(config, exp)
        upsert_result(row)
        print(f"updated: {RESULTS_PATH}")
        return
    if args.training_steps is not None:
        train_cfg["training_steps"] = args.training_steps
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.log_freq is not None:
        train_cfg["log_freq"] = args.log_freq
    if args.deploy_max_steps is not None:
        deploy_cfg["deploy_max_steps"] = args.deploy_max_steps
    base_env = os.environ.copy()
    python_cmd = [sys.executable]

    phases = [args.phase]
    if args.phase == "both":
        phases = ["train", "deploy"]
    if exp.get("deploy_only") and "train" in phases:
        phases = ["deploy"]

    if "train" in phases:
        env = train_env(base_env, exp, train_cfg, smoke=args.smoke)
        code = run_command(
            python_cmd + ["3.train.py"],
            env,
            EXP_DIR / "logs" / f"{exp['id']}_train.log",
            dry_run=args.dry_run,
        )
        if code != 0:
            raise SystemExit(code)

    if "deploy" in phases:
        trials = args.deploy_trials or int(deploy_cfg["deploy_trials"])
        if args.smoke:
            trials = 1
        # 允许从指定 seed 追加闭环复测，避免覆盖已经保留的视频和指标。
        for seed in range(args.deploy_seed_start, args.deploy_seed_start + trials):
            env = deploy_env(base_env, exp, deploy_cfg, train_cfg, seed=seed, smoke=args.smoke)
            # 每次部署前清理同 seed 的旧指标，避免新 checkpoint 的结果被旧 JSON 污染。
            metrics_path = Path(env["ACT_DEPLOY_METRICS_PATH"])
            if metrics_path.exists() and not args.dry_run:
                metrics_path.unlink()
            code = run_command(
                python_cmd + ["4.deploy.py"],
                env,
                EXP_DIR / "logs" / f"{exp['id']}_deploy_seed{seed}.log",
                dry_run=args.dry_run,
            )
            if code != 0:
                write_failed_deploy_metrics(env, code)
                print(f"deploy seed {seed} failed with exit code {code}")
                if not args.continue_on_fail:
                    raise SystemExit(code)
            # WSLg/GLFW 连续打开窗口时需要一点释放时间，避免下一组部署被图形资源影响。
            if seed + 1 < args.deploy_seed_start + trials and args.deploy_cooldown > 0:
                time.sleep(args.deploy_cooldown)

    if not args.dry_run:
        row = summarize_result(config, exp)
        upsert_result(row)
        print(f"updated: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

