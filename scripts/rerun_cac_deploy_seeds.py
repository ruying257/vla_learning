# 按指定 seed 补跑 CAC deploy 实验(跑None TE实验有问题，卡着不动)

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "cac_act_paper"
RESULTS_PATH = EXP_DIR / "results.csv"
DEFAULT_FAILURE_GUIDED_DATASET = "failure_seed_data"
DEFAULT_FGDA_CKPT = "./ckpt/v5_finetune_new_data"
DEFAULT_FORCE_RELEASE_STREAK = 3
DEPLOY_SCRIPT = "4.deploy_test.py"

TE_SWEEP = [
    ("FGDA_TE_none", "none", "FGDA 新 checkpoint，无 temporal ensemble 对照。"),
    ("FGDA_TE_001", "0.01", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.01。"),
    ("FGDA_TE_005", "0.05", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.05。"),
    ("FGDA_TE_010", "0.10", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.10。"),
    ("FGDA_TE_015", "0.15", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.15。"),
    ("FGDA_TE_020", "0.20", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.20。"),
    ("FGDA_TE_030", "0.30", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.30。"),
    ("FGDA_TE_070", "0.70", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.70。"),
    ("FGDA_TE_090", "0.90", "FGDA 新 checkpoint，固定 temporal ensemble 系数 0.90。"),
]

# 在这里修改默认补跑 seed；命令行 --deploy-seeds 会覆盖该默认值
DEFAULT_DEPLOY_SEEDS = [1, 4, 7, 8, 11, 13, 15]

RESULT_FIELDS = [
    "exp_id",
    "suite",
    "dataset",
    "ckpt_dir",
    "temporal_ensemble_coeff",
    "adaptive_te",
    "stage_resampling",
    "steps",
    "mean_action_error",
    "final_loss",
    "success_rate",
    "placement_success_rate",
    "release_success_rate",
    "strict_success_rate",
    "avg_steps",
    "avg_release_steps",
    "avg_success_steps",
    "action_smoothness_mean",
    "prediction_inconsistency_mean",
    "final_mug_plate_xy_dist",
    "min_mug_plate_xy_dist",
    "failure_mode",
    "video_path",
    "notes",
]

SEED_RESULT_FIELDS = [
    "seed",
    "executed_steps",
    "success",
    "error",
    "action_smoothness_mean",
    "action_smoothness_max",
    "prediction_inconsistency_mean",
    "prediction_inconsistency_max",
]


def experiment_output_dir(exp):
    """返回单个实验的独立输出目录。"""
    return EXP_DIR / exp["id"]


def seed_from_metric_path(path):
    """从 deploy metrics 文件名中解析 seed，用于稳定排序。"""
    match = re.search(r"deploy_seed(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def deploy_metric_paths(exp):
    """合并新旧 metrics 路径；同 seed 时优先使用实验独立目录。"""
    exp_dir = experiment_output_dir(exp)
    paths_by_seed = {}
    legacy_paths = (EXP_DIR / "metrics").glob(f"{exp['id']}_deploy_seed*.json")
    new_paths = (exp_dir / "metrics").glob("deploy_seed*.json")
    for path in legacy_paths:
        seed = seed_from_metric_path(path)
        paths_by_seed[seed if seed != 10**9 else path.name] = path
    for path in new_paths:
        seed = seed_from_metric_path(path)
        paths_by_seed[seed if seed != 10**9 else path.name] = path
    return sorted(paths_by_seed.values(), key=seed_from_metric_path)


def clean_proxy(env):
    """实验脚本默认移除代理，避免本地/国内网络环境下出现不稳定下载。"""
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)


def supports_deploy_no_ensemble():
    """检查 deploy 脚本是否支持无 temporal ensemble。"""
    deploy_text = (ROOT / DEPLOY_SCRIPT).read_text(encoding="utf-8", errors="ignore").lower()
    has_temporal_env = "act_temporal_ensemble_coeff" in deploy_text
    parses_none_literal = ('"none"' in deploy_text or "'none'" in deploy_text) and "lower()" in deploy_text
    return has_temporal_env and parses_none_literal


def experiment_matrix(args):
    """返回 FGDA 新 checkpoint 在不同 TE 参数下的 deploy sweep。"""
    ckpt_dir = args.ckpt_dir
    failure_dataset = args.failure_guided_dataset
    experiments = []
    for exp_id, te_coeff, notes in TE_SWEEP:
        experiments.append(
            {
                "id": exp_id,
                "suite": "fgda",
                "dataset_root": failure_dataset,
                "ckpt_dir": ckpt_dir,
                "temporal_ensemble_coeff": te_coeff,
                "adaptive_te": False,
                "stage_resampling": False,
                "requires_no_ensemble_support": te_coeff == "none",
                "notes": notes,
            }
        )
    return experiments


def select_experiments(args):
    """根据命令行参数选择 FGDA TE deploy 实验。"""
    matrix = experiment_matrix(args)
    if args.exp:
        wanted = set(args.exp)
        selected = [exp for exp in matrix if exp["id"] in wanted]
        missing = wanted - {exp["id"] for exp in selected}
        if missing:
            raise SystemExit(f"Unknown CAC experiment id: {', '.join(sorted(missing))}")
    else:
        selected = matrix
    return selected


def list_experiments(args):
    for exp in experiment_matrix(args):
        print(f"{exp['id']} ({exp['suite']}): {exp['notes']}")


def ensure_supported(exp, phases, dry_run):
    """pending 能 dry-run 展示命令，但正式运行前必须确认底层脚本支持。"""
    if dry_run:
        return
    if "deploy" in phases and exp.get("requires_no_ensemble_support") and not supports_deploy_no_ensemble():
        raise SystemExit(
            f"{exp['id']} needs {DEPLOY_SCRIPT} to support ACT_TEMPORAL_ENSEMBLE_COEFF=none before formal deploy."
        )


def ensure_dataset(exp, phases, dry_run):
    if dry_run:
        return
    if "train" not in phases and "deploy" not in phases:
        return
    dataset_path = ROOT / exp["dataset_root"]
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset missing for {exp['id']}: {dataset_path}\n"
            "Please collect/sync the dataset first, or pass --failure-guided-dataset."
        )


def ensure_checkpoint(exp, phases, dry_run):
    if dry_run or "deploy" not in phases or "train" in phases:
        return
    ckpt_path = ROOT / exp["ckpt_dir"]
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint missing for deploy-only {exp['id']}: {ckpt_path}")


def deploy_env(base_env, exp, args, seed):
    env = dict(base_env)
    clean_proxy(env)
    exp_dir = experiment_output_dir(exp)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "ACT_DATASET_ROOT": exp["dataset_root"],
            "ACT_CKPT_DIR": exp["ckpt_dir"],
            "ACT_CHUNK_SIZE": str(args.chunk_size),
            "ACT_N_ACTION_STEPS": str(args.n_action_steps),
            "ACT_TEMPORAL_ENSEMBLE_COEFF": str(exp["temporal_ensemble_coeff"]),
            "ACT_DEPLOY_MAX_STEPS": str(args.deploy_max_steps),
            "ACT_DEPLOY_SEED": str(seed),
            "ACT_VIDEO_DIR": str(exp_dir / "videos"),
            "ACT_RECORD_VIDEO": "1",
            "ACT_DEPLOY_METRICS_PATH": str(exp_dir / "metrics" / f"deploy_seed{seed}.json"),
            "ACT_FORCE_RELEASE_ON_PLACEMENT": "1",
            "ACT_FORCE_RELEASE_STREAK": str(DEFAULT_FORCE_RELEASE_STREAK),
        }
    )
    return env


def print_env_delta(env):
    keys = [
        "ACT_DATASET_ROOT",
        "ACT_CKPT_DIR",
        "ACT_CHUNK_SIZE",
        "ACT_N_ACTION_STEPS",
        "ACT_TEMPORAL_ENSEMBLE_COEFF",
        "ACT_DEPLOY_SEED",
        "ACT_VIDEO_DIR",
        "ACT_DEPLOY_METRICS_PATH",
        "ACT_FORCE_RELEASE_ON_PLACEMENT",
        "ACT_FORCE_RELEASE_STREAK",
    ]
    for key in keys:
        if key in env:
            print(f"  {key}={env[key]}")


def run_command(cmd, env, log_path, dry_run):
    print(" ".join(cmd))
    print(f"log: {log_path}")
    print_env_delta(env)
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


def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_signature(path):
    """记录文件状态，用于判断本次 deploy 是否真的写了新 metrics。"""
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def rate(payloads, key):
    values = [bool(payload.get(key, payload.get("success", False))) for payload in payloads]
    return "" if not values else f"{sum(values) / len(values):.2f}"


def mean_metric(payloads, key, digits=4):
    values = [payload[key] for payload in payloads if payload.get(key) is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.{digits}f}"


def is_release_success(payload):
    return bool(payload.get("placement_success", False)) and payload.get("final_gripper_qpos") is not None and payload.get("final_gripper_qpos", 1.0) < 0.1


def default_args_for_order():
    class Defaults:
        ckpt_dir = DEFAULT_FGDA_CKPT
        failure_guided_dataset = DEFAULT_FAILURE_GUIDED_DATASET

    return Defaults()


def summarize_experiment(exp, args):
    exp_dir = experiment_output_dir(exp)
    train_metrics = read_json(exp_dir / "metrics" / "train.json")
    if train_metrics is None:
        train_metrics = read_json(EXP_DIR / "metrics" / f"{exp['id']}_train.json")

    deploy_paths = deploy_metric_paths(exp)
    deploy_payloads = [read_json(path) for path in deploy_paths]
    deploy_payloads = [payload for payload in deploy_payloads if payload]

    release_success_payloads = [
        payload
        for payload in deploy_payloads
        if is_release_success(payload)
    ]
    strict_success_payloads = [
        payload
        for payload in deploy_payloads
        if payload.get("strict_success", payload.get("success", False))
    ]
    failure_modes = sorted({payload.get("failure_mode", "") for payload in deploy_payloads if payload.get("failure_mode")})
    video_path = ""
    for payload in deploy_payloads:
        if payload.get("video_path"):
            video_path = payload["video_path"]
            break

    return {
        "exp_id": exp["id"],
        "suite": exp["suite"],
        "dataset": exp["dataset_root"],
        "ckpt_dir": exp["ckpt_dir"],
        "temporal_ensemble_coeff": exp["temporal_ensemble_coeff"],
        "adaptive_te": int(bool(exp.get("adaptive_te"))),
        "stage_resampling": int(bool(exp.get("stage_resampling"))),
        "steps": train_metrics.get("training_steps", "") if train_metrics else "",
        "mean_action_error": "" if not train_metrics else f"{train_metrics['mean_action_error']:.4f}",
        "final_loss": "" if not train_metrics else f"{train_metrics['final_loss']:.4f}",
        "success_rate": rate(deploy_payloads, "success"),
        "placement_success_rate": rate(deploy_payloads, "placement_success"),
        "release_success_rate": "" if not deploy_payloads else f"{sum(is_release_success(payload) for payload in deploy_payloads) / len(deploy_payloads):.2f}",
        "strict_success_rate": rate(deploy_payloads, "strict_success"),
        "avg_steps": mean_metric(deploy_payloads, "executed_steps", digits=1),
        "avg_release_steps": mean_metric(release_success_payloads, "executed_steps", digits=1),
        "avg_success_steps": mean_metric(strict_success_payloads, "executed_steps", digits=1),
        "action_smoothness_mean": mean_metric(deploy_payloads, "action_smoothness_mean"),
        "prediction_inconsistency_mean": mean_metric(deploy_payloads, "prediction_inconsistency_mean"),
        "final_mug_plate_xy_dist": mean_metric(deploy_payloads, "final_mug_plate_xy_dist"),
        "min_mug_plate_xy_dist": mean_metric(deploy_payloads, "min_mug_plate_xy_dist"),
        "failure_mode": "+".join(failure_modes),
        "video_path": video_path,
        "notes": exp["notes"],
    }


def seed_result_row(path, payload):
    """把单个 seed 的部署 JSON 压缩成论文排查用的关键指标行。"""
    seed = payload.get("deploy_seed")
    if seed is None:
        seed = seed_from_metric_path(path)
        if seed == 10**9:
            seed = ""
    return {
        "seed": seed,
        "executed_steps": payload.get("executed_steps", ""),
        "success": payload.get("success", ""),
        "error": payload.get("error", ""),
        "action_smoothness_mean": payload.get("action_smoothness_mean", ""),
        "action_smoothness_max": payload.get("action_smoothness_max", ""),
        "prediction_inconsistency_mean": payload.get("prediction_inconsistency_mean", ""),
        "prediction_inconsistency_max": payload.get("prediction_inconsistency_max", ""),
    }


def seed_result_key(seed):
    """统一 seed 键，避免 CSV 字符串和 JSON 数字导致同一 seed 重复。"""
    if seed is None:
        return ""
    seed_text = str(seed).strip()
    if not seed_text:
        return ""
    try:
        return str(int(seed_text))
    except ValueError:
        return seed_text


def seed_result_sort_key(row):
    seed = seed_result_key(row.get("seed"))
    try:
        return 0, int(seed)
    except ValueError:
        return 1, seed


def upsert_seed_results(exp, metric_paths):
    """只用本次 deploy 生成的 metrics 增量更新 seed_results.csv。"""
    metric_paths = list(metric_paths)
    if not metric_paths:
        return None

    output_path = experiment_output_dir(exp) / "seed_results.csv"
    rows_by_seed = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = seed_result_key(row.get("seed"))
                if key:
                    rows_by_seed[key] = {field: row.get(field, "") for field in SEED_RESULT_FIELDS}

    for path in metric_paths:
        payload = read_json(path)
        if payload:
            row = seed_result_row(path, payload)
            key = seed_result_key(row.get("seed"))
            if key:
                row["seed"] = key
                rows_by_seed[key] = row

    rows = sorted(rows_by_seed.values(), key=seed_result_sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def upsert_result(row):
    rows = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    rows = [old for old in rows if old.get("exp_id") != row["exp_id"]]
    rows.append(row)
    order = {exp["id"]: idx for idx, exp in enumerate(experiment_matrix(default_args_for_order()))}
    rows.sort(key=lambda item: order.get(item.get("exp_id"), 999))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_run_outputs(exp, args, updated_metric_paths):
    """部署中途失败也要落盘已产生的有效 metrics。"""
    if args.dry_run:
        return
    seed_results_path = upsert_seed_results(exp, updated_metric_paths)
    upsert_result(summarize_experiment(exp, args))
    if seed_results_path:
        print(f"updated: {seed_results_path}")


def run_one(exp, args):
    phases = ["deploy"]
    ensure_supported(exp, phases, args.dry_run)
    ensure_dataset(exp, phases, args.dry_run)
    ensure_checkpoint(exp, phases, args.dry_run)

    base_env = os.environ.copy()
    python_cmd = [sys.executable]
    updated_metric_paths = []

    for seed_index, seed in enumerate(args.deploy_seeds):
        env = deploy_env(base_env, exp, args, seed)
        metrics_path = Path(env["ACT_DEPLOY_METRICS_PATH"])
        before_signature = file_signature(metrics_path)
        code = run_command(
            python_cmd + [DEPLOY_SCRIPT],
            env,
            experiment_output_dir(exp) / "logs" / f"deploy_seed{seed}.log",
            args.dry_run,
        )
        after_signature = file_signature(metrics_path)
        if after_signature is not None and after_signature != before_signature:
            updated_metric_paths.append(metrics_path)
        if code != 0:
            print(f"deploy seed {seed} failed with exit code {code}")
            if not args.continue_on_fail:
                update_run_outputs(exp, args, updated_metric_paths)
                raise SystemExit(code)
        if seed_index + 1 < len(args.deploy_seeds) and args.deploy_cooldown > 0:
            time.sleep(args.deploy_cooldown)

    update_run_outputs(exp, args, updated_metric_paths)


def summarize_only(exps, args):
    for exp in exps:
        upsert_result(summarize_experiment(exp, args))
    print(f"updated: {RESULTS_PATH}")
    print("seed_results.csv is only incrementally updated after deploy writes metrics.")


def parse_seed_list(seed_text):
    """解析逗号分隔的 deploy seed 列表，例如 3 或 3,7,11。"""
    try:
        seeds = [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--deploy-seeds 只支持整数或逗号分隔的整数列表") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--deploy-seeds 至少需要指定一个整数 seed")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun CAC deploy for explicit experiment ids and seed list.")
    parser.add_argument("--list", action="store_true", help="List CAC paper experiment matrix and exit.")
    parser.add_argument("--exp", nargs="+", help="Required experiment ids to rerun, for example FGDA_TE_090.")
    parser.add_argument("--failure-guided-dataset", default=DEFAULT_FAILURE_GUIDED_DATASET)
    parser.add_argument("--ckpt-dir", default=DEFAULT_FGDA_CKPT, help="Finetuned FGDA checkpoint used for deploy.")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument(
        "--deploy-seeds",
        type=parse_seed_list,
        default=DEFAULT_DEPLOY_SEEDS,
        help="Seed list to rerun, for example 3 or 3,7,11.",
    )
    parser.add_argument("--deploy-max-steps", type=int, default=500)
    parser.add_argument("--deploy-cooldown", type=float, default=2.0)
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.list and not args.exp:
        parser.error("--exp is required when rerunning deploy seeds")
    return args


def main():
    args = parse_args()
    if args.list:
        list_experiments(args)
        return

    selected = select_experiments(args)
    if not selected:
        print("No CAC experiments selected.")
        return

    for exp in selected:
        print(f"===== {exp['id']} ({exp['suite']}) =====")
        print(f"rerun deploy seeds: {args.deploy_seeds}")
        run_one(exp, args)

    if not args.dry_run:
        print(f"results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
