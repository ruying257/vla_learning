#!/usr/bin/env python3
"""从已有 TE sweep 中提取指定 seed 范围，并重新生成汇总表。"""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "experiments" / "te_sweep_v5"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "te_sweep_v5_20seeds"

RESULT_FIELDS = [
    "exp_id",
    "dataset",
    "ckpt_dir",
    "temporal_ensemble_coeff",
    "success_rate",
    "avg_steps",
    "avg_success_steps",
    "action_smoothness_mean",
    "prediction_inconsistency_mean",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a seed subset from an existing temporal ensemble sweep."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=20)
    parser.add_argument(
        "--exclude-seeds",
        default="",
        help="Comma-separated seeds excluded from the selected range, for example: 17,23.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_excluded_seeds(value):
    """解析逗号分隔的排除 seed，并拒绝无法识别的输入。"""
    excluded = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            excluded.add(int(item))
        except ValueError as exc:
            raise ValueError(f"无法识别的排除 seed: {item!r}") from exc
    return excluded


def read_csv_rows(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_error_annotations(path):
    """读取人工填写的非空 error，并按整数 seed 建立索引。"""
    annotations = {}
    for row in read_csv_rows(path):
        error = row.get("error", "")
        if not error.strip():
            continue
        try:
            seed = int(row.get("seed", ""))
        except (TypeError, ValueError):
            continue
        annotations[seed] = error
    return annotations


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def experiment_sort_key(exp_id):
    """保持无 TE 对照在首位，其余实验按系数编号排序。"""
    if exp_id == "TE_none":
        return 0, -1
    suffix = exp_id.removeprefix("TE_")
    try:
        return 1, int(suffix)
    except ValueError:
        return 2, exp_id


def discover_experiments(source_dir):
    """优先沿用源全局汇总表顺序，再补充尚未出现在表中的 TE 目录。"""
    directory_ids = {
        path.name
        for path in source_dir.glob("TE_*")
        if path.is_dir()
    }
    if not directory_ids:
        raise ValueError(f"源目录中没有 TE 实验目录: {source_dir}")

    ordered_ids = []
    for row in read_csv_rows(source_dir / "result.csv"):
        exp_id = row.get("exp_id", "")
        if exp_id in directory_ids and exp_id not in ordered_ids:
            ordered_ids.append(exp_id)
    ordered_ids.extend(sorted(directory_ids - set(ordered_ids), key=experiment_sort_key))
    return ordered_ids


def source_metadata(source_dir, exp_id):
    """读取源汇总中的实验配置字段，指标字段会基于 seed 子集重新计算。"""
    rows = read_csv_rows(source_dir / exp_id / "result.csv")
    if rows:
        return rows[0]
    for row in read_csv_rows(source_dir / "result.csv"):
        if row.get("exp_id") == exp_id:
            return row
    return {}


def mean_metric(payloads, key, digits=4):
    values = [payload[key] for payload in payloads if payload.get(key) is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.{digits}f}"


def success_rate(payloads):
    if not payloads:
        return ""
    successes = sum(bool(payload.get("success", False)) for payload in payloads)
    return f"{successes / len(payloads):.2f}"


def seed_result_row(payload, error_override=""):
    return {
        "seed": payload.get("deploy_seed", ""),
        "executed_steps": payload.get("executed_steps", ""),
        "success": payload.get("success", ""),
        # error 可能是人工复核结论，非空标注不能被 metrics 中的空值覆盖。
        "error": error_override or payload.get("error", ""),
        "action_smoothness_mean": payload.get("action_smoothness_mean", ""),
        "action_smoothness_max": payload.get("action_smoothness_max", ""),
        "prediction_inconsistency_mean": payload.get("prediction_inconsistency_mean", ""),
        "prediction_inconsistency_max": payload.get("prediction_inconsistency_max", ""),
    }


def summarize_experiment(exp_id, payloads, metadata):
    """使用与 run_te.py 相同的定义汇总选中 seed。"""
    successful_payloads = [
        payload
        for payload in payloads
        if payload.get("strict_success", payload.get("success", False))
    ]
    video_path = next(
        (payload["video_path"] for payload in payloads if payload.get("video_path")),
        "",
    )
    first_payload = payloads[0]
    return {
        "exp_id": exp_id,
        "dataset": metadata.get("dataset", first_payload.get("dataset_root", "")),
        "ckpt_dir": metadata.get("ckpt_dir", first_payload.get("ckpt_dir", "")),
        "temporal_ensemble_coeff": metadata.get(
            "temporal_ensemble_coeff",
            first_payload.get("temporal_ensemble_coeff", ""),
        ),
        "success_rate": success_rate(payloads),
        "avg_steps": mean_metric(payloads, "executed_steps", digits=1),
        "avg_success_steps": mean_metric(successful_payloads, "executed_steps", digits=1),
        "action_smoothness_mean": mean_metric(payloads, "action_smoothness_mean"),
        "prediction_inconsistency_mean": mean_metric(
            payloads, "prediction_inconsistency_mean"
        ),
        # 不复制视频；这里保留源视频路径，便于需要时回溯原始轨迹。
        "video_path": video_path,
        "notes": metadata.get("notes", ""),
    }


def collect_source_data(source_dir, experiment_ids, seeds):
    """先完整校验所有输入，避免生成只有部分实验的目标目录。"""
    collected = {}
    for exp_id in experiment_ids:
        records = []
        for seed in seeds:
            metric_path = source_dir / exp_id / "metrics" / f"deploy_seed{seed}.json"
            if not metric_path.is_file():
                raise FileNotFoundError(f"缺少 seed 指标文件: {metric_path}")
            payload = read_json(metric_path)
            if payload.get("deploy_seed") != seed:
                raise ValueError(
                    f"seed 与文件名不一致: {metric_path} 中为 {payload.get('deploy_seed')!r}"
                )
            records.append((seed, metric_path, payload))
        collected[exp_id] = records
    return collected


def collect_error_annotations(source_dir, output_dir, experiment_ids):
    """合并人工标注；目标目录已有标注优先于源目录标注。"""
    annotations = {}
    for exp_id in experiment_ids:
        source_errors = read_error_annotations(source_dir / exp_id / "seed_results.csv")
        target_errors = read_error_annotations(output_dir / exp_id / "seed_results.csv")
        source_errors.update(target_errors)
        annotations[exp_id] = source_errors
    return annotations


def build_output(
    staging_dir,
    source_dir,
    experiment_ids,
    collected,
    error_annotations,
):
    global_rows = []
    for exp_id in experiment_ids:
        records = collected[exp_id]
        output_exp_dir = staging_dir / exp_id
        output_metrics_dir = output_exp_dir / "metrics"
        output_metrics_dir.mkdir(parents=True, exist_ok=True)

        for seed, source_path, _ in records:
            shutil.copy2(source_path, output_metrics_dir / f"deploy_seed{seed}.json")

        payloads = [payload for _, _, payload in records]
        write_csv(
            output_exp_dir / "seed_results.csv",
            SEED_RESULT_FIELDS,
            [
                seed_result_row(payload, error_annotations[exp_id].get(seed, ""))
                for seed, _, payload in records
            ],
        )
        summary = summarize_experiment(
            exp_id,
            payloads,
            source_metadata(source_dir, exp_id),
        )
        write_csv(output_exp_dir / "result.csv", RESULT_FIELDS, [summary])
        global_rows.append(summary)

    write_csv(staging_dir / "result.csv", RESULT_FIELDS, global_rows)


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.seed_start > args.seed_end:
        raise SystemExit("--seed-start 不能大于 --seed-end")
    if not source_dir.is_dir():
        raise SystemExit(f"源目录不存在: {source_dir}")
    if source_dir == output_dir:
        raise SystemExit("源目录和输出目录不能相同")
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"输出目录已存在；如需重建请添加 --overwrite: {output_dir}")

    try:
        excluded_seeds = parse_excluded_seeds(args.exclude_seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    seeds = [
        seed
        for seed in range(args.seed_start, args.seed_end + 1)
        if seed not in excluded_seeds
    ]
    if not seeds:
        raise SystemExit("排除后没有可提取的 seed")
    experiment_ids = discover_experiments(source_dir)
    collected = collect_source_data(source_dir, experiment_ids, seeds)
    error_annotations = collect_error_annotations(source_dir, output_dir, experiment_ids)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        build_output(
            staging_dir,
            source_dir,
            experiment_ids,
            collected,
            error_annotations,
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        f"已提取 {len(experiment_ids)} 个实验，每个实验 {len(seeds)} 个 seed: "
        f"{output_dir}"
    )
    if excluded_seeds:
        print(f"已排除 seed: {', '.join(str(seed) for seed in sorted(excluded_seeds))}")


if __name__ == "__main__":
    main()
