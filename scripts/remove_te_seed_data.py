#!/usr/bin/env python3
"""从已有 TE sweep 中完整删除指定 seed，并更新现有格式的汇总表。"""

import argparse
import csv
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_DIR = ROOT / "experiments" / "te_sweep_v5_finetune"


@dataclass
class ExperimentRemoval:
    exp_id: str
    exp_dir: Path
    seed_results_path: Path
    filtered_seed_results: bytes
    metric_path: Path
    log_path: Path
    video_paths: list[Path]
    summary: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove one seed from every experiment in an existing TE sweep."
    )
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_from_metric_path(path):
    match = re.fullmatch(r"deploy_seed(\d+)\.json", path.name)
    return int(match.group(1)) if match else 10**9


def rate(payloads, key):
    values = [bool(payload.get(key, payload.get("success", False))) for payload in payloads]
    return "" if not values else f"{sum(values) / len(values):.2f}"


def mean_metric(payloads, key, digits=4):
    values = [payload[key] for payload in payloads if payload.get(key) is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.{digits}f}"


def is_release_success(payload):
    """沿用原 finetune 汇总中的释放成功定义。"""
    return (
        bool(payload.get("placement_success", False))
        and payload.get("final_gripper_qpos") is not None
        and payload.get("final_gripper_qpos", 1.0) < 0.1
    )


def summarize_payloads(payloads):
    """计算新旧 result.csv 表头中可能存在的全部部署汇总字段。"""
    release_success_payloads = [payload for payload in payloads if is_release_success(payload)]
    strict_success_payloads = [
        payload
        for payload in payloads
        if payload.get("strict_success", payload.get("success", False))
    ]
    failure_modes = sorted(
        {
            payload.get("failure_mode", "")
            for payload in payloads
            if payload.get("failure_mode")
        }
    )
    video_path = next(
        (payload["video_path"] for payload in payloads if payload.get("video_path")),
        "",
    )
    return {
        "success_rate": rate(payloads, "success"),
        "placement_success_rate": rate(payloads, "placement_success"),
        "release_success_rate": (
            ""
            if not payloads
            else f"{sum(is_release_success(payload) for payload in payloads) / len(payloads):.2f}"
        ),
        "strict_success_rate": rate(payloads, "strict_success"),
        "avg_steps": mean_metric(payloads, "executed_steps", digits=1),
        "avg_release_steps": mean_metric(
            release_success_payloads, "executed_steps", digits=1
        ),
        "avg_success_steps": mean_metric(
            strict_success_payloads, "executed_steps", digits=1
        ),
        "action_smoothness_mean": mean_metric(payloads, "action_smoothness_mean"),
        "prediction_inconsistency_mean": mean_metric(
            payloads, "prediction_inconsistency_mean"
        ),
        "final_mug_plate_xy_dist": mean_metric(payloads, "final_mug_plate_xy_dist"),
        "min_mug_plate_xy_dist": mean_metric(payloads, "min_mug_plate_xy_dist"),
        "failure_mode": "+".join(failure_modes),
        "video_path": video_path,
    }


def filter_seed_results_bytes(path, seed):
    """只移除目标 seed 的原始物理行，其余字节保持完全不变。"""
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    matched_indexes = []

    for index, raw_line in enumerate(lines):
        text = raw_line.decode("utf-8-sig")
        try:
            row = next(csv.reader([text]))
        except csv.Error as exc:
            raise ValueError(f"无法解析 CSV 物理行: {path}:{index + 1}") from exc
        if row and row[0].strip() == str(seed):
            matched_indexes.append(index)

    if len(matched_indexes) != 1:
        raise ValueError(
            f"{path} 中应恰好有一行 seed={seed}，实际为 {len(matched_indexes)} 行"
        )

    target_index = matched_indexes[0]
    return b"".join(
        raw_line for index, raw_line in enumerate(lines) if index != target_index
    )


def collect_remaining_payloads(exp_dir, removed_seed):
    paths = sorted(
        exp_dir.glob("metrics/deploy_seed*.json"),
        key=seed_from_metric_path,
    )
    payloads = []
    seen_seeds = set()
    for path in paths:
        payload = read_json(path)
        seed = payload.get("deploy_seed", seed_from_metric_path(path))
        if seed == removed_seed:
            continue
        if seed in seen_seeds:
            raise ValueError(f"{exp_dir} 中存在重复 metrics seed: {seed}")
        seen_seeds.add(seed)
        payloads.append(payload)
    if not payloads:
        raise ValueError(f"删除 seed={removed_seed} 后没有剩余 metrics: {exp_dir}")
    return payloads


def preflight(root_dir, seed):
    """在任何修改前检查全部实验，避免产生部分删除结果。"""
    exp_dirs = sorted(path for path in root_dir.glob("TE_*") if path.is_dir())
    if not exp_dirs:
        raise ValueError(f"没有找到 TE 实验目录: {root_dir}")

    removals = []
    for exp_dir in exp_dirs:
        seed_results_path = exp_dir / "seed_results.csv"
        metric_path = exp_dir / "metrics" / f"deploy_seed{seed}.json"
        log_path = exp_dir / "logs" / f"deploy_seed{seed}.log"
        video_paths = sorted((exp_dir / "videos").glob(f"*seed{seed}_*"))

        for required_path in [seed_results_path, metric_path, log_path, exp_dir / "result.csv"]:
            if not required_path.is_file():
                raise FileNotFoundError(f"缺少待清理实验文件: {required_path}")
        if not video_paths:
            raise FileNotFoundError(f"没有找到 seed={seed} 视频: {exp_dir / 'videos'}")

        metric_seed = read_json(metric_path).get("deploy_seed")
        if metric_seed != seed:
            raise ValueError(f"metrics 内容与文件名 seed 不一致: {metric_path}")

        filtered_seed_results = filter_seed_results_bytes(seed_results_path, seed)
        payloads = collect_remaining_payloads(exp_dir, seed)
        removals.append(
            ExperimentRemoval(
                exp_id=exp_dir.name,
                exp_dir=exp_dir,
                seed_results_path=seed_results_path,
                filtered_seed_results=filtered_seed_results,
                metric_path=metric_path,
                log_path=log_path,
                video_paths=video_paths,
                summary=summarize_payloads(payloads),
            )
        )

    global_result_path = root_dir / "result.csv"
    if not global_result_path.is_file():
        raise FileNotFoundError(f"缺少全局汇总表: {global_result_path}")
    return removals, global_result_path


def read_csv_table(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头: {path}")
        return reader.fieldnames, list(reader)


def csv_bytes(fieldnames, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {field: row.get(field, "") for field in fieldnames}
        for row in rows
    )
    return buffer.getvalue().encode("utf-8")


def updated_result_bytes(path, summaries):
    """保留现有表头和非汇总字段，只更新表头中已有的汇总指标。"""
    fieldnames, rows = read_csv_table(path)
    for row in rows:
        summary = summaries.get(row.get("exp_id", ""))
        if summary is None:
            continue
        for field, value in summary.items():
            if field in fieldnames:
                row[field] = value
    return csv_bytes(fieldnames, rows)


def write_bytes_atomic(path, content):
    """在同目录原子替换文件，避免写入中断留下半个 CSV。"""
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        temp_path = Path(f.name)
        f.write(content)
    try:
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def print_plan(removals, seed, dry_run):
    prefix = "[dry-run] " if dry_run else ""
    video_count = sum(len(removal.video_paths) for removal in removals)
    print(
        f"{prefix}seed={seed}: {len(removals)} 行 seed_results.csv, "
        f"{len(removals)} metrics, {len(removals)} logs, {video_count} videos"
    )
    for removal in removals:
        print(f"{prefix}{removal.exp_id}: videos={len(removal.video_paths)}")


def execute_removal(removals, global_result_path):
    summaries = {removal.exp_id: removal.summary for removal in removals}
    experiment_result_updates = {
        removal.exp_dir / "result.csv": updated_result_bytes(
            removal.exp_dir / "result.csv", summaries
        )
        for removal in removals
    }
    global_result_update = updated_result_bytes(global_result_path, summaries)

    for removal in removals:
        write_bytes_atomic(removal.seed_results_path, removal.filtered_seed_results)
        # 写入后立即校验，确保其它 seed 的原始字节没有被 CSV 序列化改变。
        if removal.seed_results_path.read_bytes() != removal.filtered_seed_results:
            raise RuntimeError(f"seed_results.csv 字节校验失败: {removal.seed_results_path}")

    for removal in removals:
        removal.metric_path.unlink()
        removal.log_path.unlink()
        for video_path in removal.video_paths:
            video_path.unlink()

    for path, content in experiment_result_updates.items():
        write_bytes_atomic(path, content)
    write_bytes_atomic(global_result_path, global_result_update)


def main():
    args = parse_args()
    root_dir = args.root_dir.resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"实验根目录不存在: {root_dir}")

    try:
        removals, global_result_path = preflight(root_dir, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print_plan(removals, args.seed, args.dry_run)
    if args.dry_run:
        return

    execute_removal(removals, global_result_path)
    print(f"已从 {len(removals)} 个实验删除 seed={args.seed} 并更新汇总表: {root_dir}")


if __name__ == "__main__":
    main()
