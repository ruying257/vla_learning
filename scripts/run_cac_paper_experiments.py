# 运行CAC论文实验脚本

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
DEFAULT_BASELINE_DATASET = "datasets/demo_v5_30demos_random"
DEFAULT_FAILURE_GUIDED_DATASET = "datasets/demo_v6_failure_guided"
DEFAULT_EXTRA_RANDOM_DATASET = "datasets/demo_v6_extra_random"
DEFAULT_CATE_CKPT = "./ckpt/v5"
DEFAULT_FORCE_RELEASE_STREAK = 3
DEPLOY_SCRIPT = "4.deploy_test.py"

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

def supports_adaptive_temporal_ensemble():
    """检查 deploy 脚本是否支持自适应 temporal ensemble。"""
    deploy_text = (ROOT / DEPLOY_SCRIPT).read_text(encoding="utf-8", errors="ignore")
    return "ACT_ADAPTIVE_TE" in deploy_text




def supports_stage_resampling():
    """检查 train.py 是否支持阶段采样。"""
    """检查 train.py 是否支持阶段采样。"""
    train_text = (ROOT / "3.train.py").read_text(encoding="utf-8", errors="ignore")
    return "ACT_STAGE_RESAMPLING" in train_text


def experiment_matrix(args):
    """返回 CAC 论文实验矩阵；FGDA 主结论只使用 E0/E2/E3。"""
    cate_ckpt = args.ckpt_dir
    failure_dataset = args.failure_guided_dataset
    extra_random_dataset = args.extra_random_dataset

    return [
        {
            "id": "CATE_E0_no_ensemble",
            "suite": "cate",
            "phase_default": "deploy",
            "dataset_root": DEFAULT_BASELINE_DATASET,
            "ckpt_dir": cate_ckpt,
            "temporal_ensemble_coeff": "none",
            "adaptive_te": False,
            "stage_resampling": False,
            "deploy_only": True,
            "requires_no_ensemble_support": True,
            "notes": "原始 ACT，无 temporal ensemble；用于验证不平滑时的动作抖动。",
        },
        {
            "id": "CATE_E1_fixed_07",
            "suite": "cate",
            "phase_default": "deploy",
            "dataset_root": DEFAULT_BASELINE_DATASET,
            "ckpt_dir": cate_ckpt,
            "temporal_ensemble_coeff": "0.7",
            "adaptive_te": False,
            "stage_resampling": False,
            "deploy_only": True,
            "notes": "固定 temporal ensemble 系数 0.7，响应较快但可能更抖。",
        },
        {
            "id": "CATE_E2_fixed_09",
            "suite": "cate",
            "phase_default": "deploy",
            "dataset_root": DEFAULT_BASELINE_DATASET,
            "ckpt_dir": cate_ckpt,
            "temporal_ensemble_coeff": "0.9",
            "adaptive_te": False,
            "stage_resampling": False,
            "deploy_only": True,
            "notes": "固定 temporal ensemble 系数 0.9，当前强平滑基线。",
        },
        {
            "id": "CATE_E3_adaptive_pending",
            "suite": "cate",
            "phase_default": "deploy",
            "dataset_root": DEFAULT_BASELINE_DATASET,
            "ckpt_dir": cate_ckpt,
            "temporal_ensemble_coeff": "0.9",
            "adaptive_te": True,
            "stage_resampling": False,
            "deploy_only": True,
            "pending": True,
            "requires_adaptive_support": True,
            "notes": "自适应 temporal ensemble 预留项；当前 deploy 未接入时不能正式运行。",
        },
        {
            "id": "FGDA_E0_v5_baseline",
            "suite": "fgda",
            "phase_default": "both",
            "dataset_root": DEFAULT_BASELINE_DATASET,
            "ckpt_dir": "./ckpt/cac_FGDA_E0_v5_baseline",
            "temporal_ensemble_coeff": "0.9",
            "adaptive_te": False,
            "stage_resampling": False,
            "notes": "v5 随机初始位置数据作为 FGDA baseline。",
        },
        {
            "id": "FGDA_E1_v5_extra_random_optional",
            "suite": "fgda",
            "phase_default": "both",
            "dataset_root": extra_random_dataset,
            "ckpt_dir": "./ckpt/cac_FGDA_E1_v5_extra_random_optional",
            "temporal_ensemble_coeff": "0.9",
            "adaptive_te": False,
            "stage_resampling": False,
            "optional": True,
            "notes": "普通补随机数据的可选对照；无该数据集时不纳入主论文结论。",
        },
        {
            "id": "FGDA_E2_failure_guided",
            "suite": "fgda",
            "phase_default": "both",
            "dataset_root": failure_dataset,
            "ckpt_dir": "./ckpt/cac_FGDA_E2_failure_guided",
            "temporal_ensemble_coeff": "0.9",
            "adaptive_te": False,
            "stage_resampling": False,
            "notes": "失败案例补数据；与 E0 对比补数据效果。",
        },
        {
            "id": "FGDA_E3_failure_guided_resampled",
            "suite": "fgda",
            "phase_default": "both",
            "dataset_root": failure_dataset,
            "ckpt_dir": "./ckpt/cac_FGDA_E3_failure_guided_resampled",
            "temporal_ensemble_coeff": "0.9",
            "adaptive_te": False,
            "stage_resampling": True,
            "requires_stage_resampling_support": True,
            "notes": "失败案例补数据 + 关键阶段重采样；与 E2 对比重采样效果。",
        },
    ]


def select_experiments(args):
    """
    根据命令行参数选择实验
    示例：
    python run_cac_paper_experiments.py --exp CATE_E0_no_ensemble
    """
    """
    根据命令行参数选择实验
    示例：
    python run_cac_paper_experiments.py --exp CATE_E0_no_ensemble
    """
    matrix = experiment_matrix(args)
    if args.exp:
        wanted = set(args.exp)
        selected = [exp for exp in matrix if exp["id"] in wanted]
        missing = wanted - {exp["id"] for exp in selected}
        if missing:
            raise SystemExit(f"Unknown CAC experiment id: {', '.join(sorted(missing))}")
    else:
        if args.suite == "combined":
            selected = [exp for exp in matrix if exp["suite"] in {"cate", "fgda"}]
        else:
            selected = [exp for exp in matrix if exp["suite"] == args.suite]

    if not args.exp and not args.include_pending:
        selected = [exp for exp in selected if not exp.get("pending")]
    if not args.exp and not args.include_optional:
        selected = [exp for exp in selected if not exp.get("optional")]
    return selected


def list_experiments(args):
    for exp in experiment_matrix(args):
        tags = []
        if exp.get("pending"):
            tags.append("pending")
        if exp.get("optional"):
            tags.append("optional")
        if exp.get("stage_resampling"):
            tags.append("stage_resampling")
        tag_text = f" [{' / '.join(tags)}]" if tags else ""
        print(f"{exp['id']} ({exp['suite']}){tag_text}: {exp['notes']}")


def phase_for(exp, requested_phase):
    if requested_phase != "auto":
        if exp.get("deploy_only") and requested_phase == "train":
            return []
        if exp.get("deploy_only") and requested_phase == "both":
            return ["deploy"]
        return ["train", "deploy"] if requested_phase == "both" else [requested_phase]
    default_phase = exp.get("phase_default", "both")
    return ["train", "deploy"] if default_phase == "both" else [default_phase]


def ensure_supported(exp, phases, dry_run):
    """pending 能 dry-run 展示命令，但正式运行前必须确认底层脚本支持。"""
    if dry_run:
        return
    if "deploy" in phases and exp.get("requires_no_ensemble_support") and not supports_deploy_no_ensemble():
        raise SystemExit(
            f"{exp['id']} needs {DEPLOY_SCRIPT} to support ACT_TEMPORAL_ENSEMBLE_COEFF=none before formal deploy."
        )
    if "deploy" in phases and exp.get("requires_adaptive_support") and not supports_adaptive_temporal_ensemble():
        raise SystemExit(f"{exp['id']} needs ACT_ADAPTIVE_TE support in {DEPLOY_SCRIPT} before formal deploy.")
    if "train" in phases and exp.get("requires_stage_resampling_support") and not supports_stage_resampling():
        raise SystemExit(f"{exp['id']} needs ACT_STAGE_RESAMPLING support in 3.train.py before formal train.")


def ensure_dataset(exp, phases, dry_run):
    if dry_run:
        return
    if "train" not in phases and "deploy" not in phases:
        return
    dataset_path = ROOT / exp["dataset_root"]
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset missing for {exp['id']}: {dataset_path}\n"
            "Please collect/sync the dataset first, or pass --failure-guided-dataset for FGDA E2/E3."
        )


def ensure_checkpoint(exp, phases, dry_run):
    if dry_run or "deploy" not in phases or "train" in phases:
        return
    ckpt_path = ROOT / exp["ckpt_dir"]
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint missing for deploy-only {exp['id']}: {ckpt_path}")


def train_env(base_env, exp, args):
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
            "ACT_LR": args.learning_rate,
            "ACT_BATCH_SIZE": str(args.batch_size),
            "ACT_NUM_WORKERS": str(args.num_workers),
            "ACT_LOG_FREQ": str(args.log_freq),
            "ACT_TRAINING_STEPS": str(args.training_steps),
            "ACT_METRICS_PATH": str(exp_dir / "metrics" / "train.json"),
        }
    )
    if exp.get("stage_resampling"):
        env["ACT_STAGE_RESAMPLING"] = "1"
    return env


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
    if exp.get("adaptive_te"):
        env.update(
            {
                "ACT_ADAPTIVE_TE": "1",
                "ACT_ADAPTIVE_ALPHA_MIN": str(args.adaptive_alpha_min),
                "ACT_ADAPTIVE_ALPHA_MAX": str(args.adaptive_alpha_max),
                "ACT_ADAPTIVE_LAMBDA": str(args.adaptive_lambda),
            }
        )
    return env


def print_env_delta(env):
    keys = [
        "ACT_DATASET_ROOT",
        "ACT_CKPT_DIR",
        "ACT_TEMPORAL_ENSEMBLE_COEFF",
        "ACT_ADAPTIVE_TE",
        "ACT_STAGE_RESAMPLING",
        "ACT_TRAINING_STEPS",
        "ACT_METRICS_PATH",
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
        ckpt_dir = DEFAULT_CATE_CKPT
        failure_guided_dataset = DEFAULT_FAILURE_GUIDED_DATASET
        extra_random_dataset = DEFAULT_EXTRA_RANDOM_DATASET

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
        "steps": train_metrics.get("training_steps", args.training_steps) if train_metrics else "",
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
    phases = phase_for(exp, args.phase)
    ensure_supported(exp, phases, args.dry_run)
    ensure_dataset(exp, phases, args.dry_run)
    ensure_checkpoint(exp, phases, args.dry_run)

    base_env = os.environ.copy()
    python_cmd = [sys.executable]
    updated_metric_paths = []

    if "train" in phases:
        env = train_env(base_env, exp, args)
        code = run_command(
            python_cmd + ["3.train.py"],
            env,
            experiment_output_dir(exp) / "logs" / "train.log",
            args.dry_run,
        )
        if code != 0:
            raise SystemExit(code)

    if "deploy" in phases:
        for seed in range(args.deploy_seed_start, args.deploy_seed_start + args.deploy_trials):
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
            if seed + 1 < args.deploy_seed_start + args.deploy_trials and args.deploy_cooldown > 0:
                time.sleep(args.deploy_cooldown)

    update_run_outputs(exp, args, updated_metric_paths)


def summarize_only(exps, args):
    for exp in exps:
        upsert_result(summarize_experiment(exp, args))
    print(f"updated: {RESULTS_PATH}")
    print("seed_results.csv is only incrementally updated after deploy writes metrics.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CAC ACT paper experiments.")
    parser.add_argument("--list", action="store_true", help="List CAC paper experiment matrix and exit.")
    parser.add_argument("--suite", choices=["cate", "fgda", "combined"], default="combined")
    parser.add_argument("--exp", nargs="+", help="Specific experiment ids to run.")
    parser.add_argument("--phase", choices=["auto", "train", "deploy", "both"], default="auto")
    parser.add_argument("--include-pending", action="store_true", help="Include pending experiments in run selection.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional non-main-paper experiments.")
    parser.add_argument("--failure-guided-dataset", default=DEFAULT_FAILURE_GUIDED_DATASET)
    parser.add_argument("--extra-random-dataset", default=DEFAULT_EXTRA_RANDOM_DATASET)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CATE_CKPT, help="Checkpoint used by CATE deploy-only experiments.")
    parser.add_argument("--training-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--learning-rate", default="1e-4")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=500)
    parser.add_argument("--deploy-trials", type=int, default=20)
    parser.add_argument("--deploy-seed-start", type=int, default=1)
    parser.add_argument("--deploy-max-steps", type=int, default=400)
    parser.add_argument("--deploy-cooldown", type=float, default=2.0)
    parser.add_argument("--adaptive-alpha-min", type=float, default=0.5)
    parser.add_argument("--adaptive-alpha-max", type=float, default=0.95)
    parser.add_argument("--adaptive-lambda", type=float, default=10.0)
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_experiments(args)
        return

    selected = select_experiments(args)
    if not selected:
        print("No CAC experiments selected.")
        return
    if args.summarize_only:
        summarize_only(selected, args)
        return

    for exp in selected:
        print(f"===== {exp['id']} ({exp['suite']}) =====")
        run_one(exp, args)

    if not args.dry_run:
        print(f"results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
