import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "act_tuning"
RESULTS_PATH = EXP_DIR / "results.csv"
ANALYSIS_PATH = EXP_DIR / "analysis.md"
FIGURE_PATH = EXP_DIR / "act_tuning_summary.png"


def load_rows(results_path):
    with open(results_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda row: int(row["exp_id"].lstrip("E")) if row["exp_id"].startswith("E") else 999)


def as_float(value):
    if value in (None, ""):
        return None
    return float(value)


def make_plot(rows, figure_path):
    """生成离线动作误差和闭环成功率对比图，便于放进项目文档或面试材料。"""
    exp_ids = [row["exp_id"] for row in rows]
    exp_labels = [f"{row['exp_id']}\n{row['steps']}s" if row.get("steps") else row["exp_id"] for row in rows]
    action_errors = [as_float(row.get("mean_action_error")) for row in rows]
    success_rates = [as_float(row.get("success_rate")) for row in rows]
    x = list(range(len(rows)))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(max(8, len(rows) * 0.75), 4.8))
    ax2 = ax1.twinx()

    ax1.bar(
        [idx - width / 2 for idx in x],
        [value if value is not None else 0 for value in action_errors],
        width=width,
        label="Mean action error",
        color="#4C78A8",
        alpha=0.88,
    )
    ax2.bar(
        [idx + width / 2 for idx in x],
        [value if value is not None else 0 for value in success_rates],
        width=width,
        label="Closed-loop success rate",
        color="#59A14F",
        alpha=0.82,
    )

    for idx, value in enumerate(action_errors):
        if value is not None:
            ax1.text(idx - width / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(success_rates):
        if value is not None:
            ax2.text(idx + width / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_labels)
    ax1.set_ylabel("Mean action error")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("ACT tuning preliminary screening")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def compact_dataset(path):
    if "3demos_fixed" in path:
        return "3 fixed demos"
    if "50demos_fixed" in path:
        return "50 fixed demos"
    if "30demos_random" in path:
        return "30 random demos"
    return path


def make_analysis(rows, figure_path):
    """根据当前 CSV 生成求职导向的实验小结。"""
    train_rows = [row for row in rows if row.get("mean_action_error")]
    deploy_rows = [row for row in rows if row.get("success_rate")]
    best_offline = min(train_rows, key=lambda row: float(row["mean_action_error"])) if train_rows else None
    train_steps = sorted({row["steps"] for row in train_rows if row.get("steps")})
    train_batches = sorted({row["batch_size"] for row in train_rows if row.get("batch_size")})
    train_budget = "/".join(train_steps) if train_steps else "unknown"
    batch_budget = "/".join(train_batches) if train_batches else "unknown"
    deploy_modes = sorted({row["failure_mode"] for row in deploy_rows if row.get("failure_mode")})
    native_crashes = [row for row in deploy_rows if "native_crash" in row.get("failure_mode", "")]
    random_rows = [
        row
        for row in train_rows
        if "30demos_random" in row["dataset"]
    ]
    best_random = min(random_rows, key=lambda row: float(row["mean_action_error"])) if random_rows else None
    # 选择一个“loss 很低但动作误差仍高”的反例，强调不能只看训练 loss。
    low_loss_high_error = None
    high_error_rows = [row for row in train_rows if float(row["mean_action_error"]) >= 0.15]
    if high_error_rows:
        low_loss_high_error = min(high_error_rows, key=lambda row: float(row["final_loss"]))
    max_step_deploys = [
        row
        for row in deploy_rows
        if "max_steps" in row.get("failure_mode", "") and row.get("video_path")
    ]
    representative_deploy = max_step_deploys[0] if max_step_deploys else None
    deep_rows = [row for row in train_rows if row.get("steps") == "6000"]

    lines = [
        "# ACT 调参实验阶段小结",
        "",
        "## 结论先行",
        "",
        f"- 当前已经跑通 {len(train_rows)} 组离线 {train_budget}-step、batch {batch_budget} 的本机中等筛选，以及 {len(deploy_rows)} 组闭环部署记录。",
        "- 这些结果已经能做超参数趋势分析，但仍不能替代 3000/6000 steps、5 seeds 的最终结论。",
        "- 从第一性原理看，离线动作误差只衡量数据分布内的模仿距离；闭环成功率还会暴露误差累积、视觉状态偏移和动作平滑滞后。",
    ]

    if best_offline:
        if "30demos_random" in best_offline["dataset"]:
            best_note = "这是随机初始位置数据上的当前最优离线结果，优先进入闭环复测。"
        else:
            best_note = "这更可能说明固定分布更容易拟合，而不是泛化更好。"
        lines.append(
            f"- 当前最低离线动作误差是 {best_offline['exp_id']} ({float(best_offline['mean_action_error']):.4f})，"
            f"数据集为 {compact_dataset(best_offline['dataset'])}；{best_note}"
        )
    if best_random:
        lines.append(
            f"- 在随机初始位置数据上，当前离线最优是 {best_random['exp_id']} "
            f"({float(best_random['mean_action_error']):.4f}, {best_random['steps']} steps)，可作为下一轮闭环验证优先候选。"
        )
    if low_loss_high_error:
        lines.append(
            f"- {low_loss_high_error['exp_id']} 的 final loss 为 {float(low_loss_high_error['final_loss']):.4f}，"
            f"但动作误差为 {float(low_loss_high_error['mean_action_error']):.4f}；这说明 loss 下降不能单独代表控制质量。"
        )
    if representative_deploy:
        lines.append(
            f"- {representative_deploy['exp_id']} 保留了 120-step 闭环失败视频，可作为轨迹偏移和误差累积的失败样例。"
        )
    random_3000 = [
        row
        for row in train_rows
        if "30demos_random" in row["dataset"] and row.get("steps") == "3000"
    ]
    if random_3000:
        best_random_3000 = min(random_3000, key=lambda row: float(row["mean_action_error"]))
        worst_random_3000 = max(random_3000, key=lambda row: float(row["mean_action_error"]))
        lines.append(
            f"- 3000-step 随机数据对照中，{best_random_3000['exp_id']} 最优 "
            f"({float(best_random_3000['mean_action_error']):.4f})，{worst_random_3000['exp_id']} 最差 "
            f"({float(worst_random_3000['mean_action_error']):.4f})。"
        )
    for row in deep_rows:
        deploy_note = ""
        if row.get("success_rate"):
            deploy_note = f"，闭环结果为 {row['failure_mode']} ({row['avg_steps']} steps)"
        lines.append(
            f"- {row['exp_id']} 已补跑到 6000 steps，动作误差为 {float(row['mean_action_error']):.4f}{deploy_note}。"
        )
    deep_deploy_rows = [row for row in deep_rows if row.get("success_rate")]
    if deep_deploy_rows:
        deep_modes = sorted({row["failure_mode"] for row in deep_deploy_rows if row.get("failure_mode")})
        lines.append(
            f"- 6000-step 深跑组已有 {len(deep_deploy_rows)} 组闭环记录，失败模式为 {', '.join(deep_modes)}，说明当前瓶颈已从离线拟合转向闭环控制。"
        )
    if deploy_rows:
        lines.append(
            f"- 闭环部署已有 {len(deploy_rows)} 组记录；当前失败模式包括 "
            f"{', '.join(deploy_modes) if deploy_modes else 'success'}。"
        )
    if native_crashes:
        lines.append("- `native_crash` 属于 MuJoCo/GLFW native 层异常，不能直接当作策略失败；需要优先复现实验窗口稳定性。")

    lines.extend(
        [
            "",
            f"![ACT tuning summary]({figure_path.name})",
            "",
            "## 当前结果表",
            "",
            "| ExpID | Dataset | chunk | lr | steps | mean_action_error | success_rate | avg_steps | failure_mode | video |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )

    for row in rows:
        video = row.get("video_path") or ""
        video_cell = Path(video).name if video else ""
        lines.append(
            "| {exp_id} | {dataset} | {chunk_size} | {lr} | {steps} | {mean_action_error} | {success_rate} | {avg_steps} | {failure_mode} | {video} |".format(
                exp_id=row["exp_id"],
                dataset=compact_dataset(row["dataset"]),
                chunk_size=row["chunk_size"],
                lr=row["lr"],
                steps=row["steps"],
                mean_action_error=row.get("mean_action_error", ""),
                success_rate=row.get("success_rate", ""),
                avg_steps=row.get("avg_steps", ""),
                failure_mode=row.get("failure_mode", ""),
                video=video_cell,
            )
        )

    lines.extend(
        [
            "",
            "## 面试讲法",
            "",
            "- 离线到闭环：离线误差降低后仍要做 MuJoCo 闭环部署；max_steps 失败更接近策略问题，native crash 则要归因到实验平台稳定性。",
            "- 数据分布：固定初始位置的离线误差更低，优先解释为训练/验证更贴近同一分布；是否真能泛化，必须看随机位置闭环部署。",
            "- Action chunk：3000-step 下 chunk25 和 chunk100 的动作误差都高于 E4，说明 chunk 长短不能只看训练 loss，要结合动作误差与闭环轨迹。",
            "- 学习率：E6 的 lr=2e-4 让 loss 很低，但动作误差最高之一，说明过快收敛可能没有带来更好的控制动作。",
            "- 深跑验证：E4/E6/E9 都已有 6000-step 多 seed 闭环记录；其中 max_steps 更接近策略闭环失败，native_crash 要单独归因为 MuJoCo/GLFW 平台稳定性。",
            "- Temporal ensemble：平滑本质是在多个时间步预测之间做加权平均，能降低高频动作噪声，但权重过强可能带来响应滞后。",
            "- 环境稳定性：native crash 和 max_steps 要分开记录；前者是实验平台可靠性问题，后者才更接近策略闭环表现问题。",
            "",
            "## 简历素材",
            "",
            "- 围绕 ACT 策略完成 10 组离线超参数/数据分布筛选，并用 MuJoCo 闭环部署验证离线误差与任务成功率之间的不一致。",
            "- 对比 action chunk size、learning rate 和示范数据分布对动作误差、训练稳定性与闭环失败模式的影响，发现低 loss 与低动作误差并不总一致，并定位 max_steps、native crash 等不同失败来源。",
            "",
            "## 下一轮正式实验",
            "",
            "- E1/E3/E4/E6/E7/E8/E9 已经完成 3000 steps，E1/E6/E9 也有 max_steps 闭环失败视频。",
            "- 下一步不再优先加长单组训练，而是继续补足 E4/E6/E9 的稳定闭环 seed，并复盘已有视频中的轨迹偏移模式。",
            "- 复盘 E4/E6/E9 视频，优先定位抓取前对齐、接触后抖动、放置阶段偏移三类失败。最终只把最有解释价值的 3 组写入简历。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize ACT tuning results.")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args()

    rows = load_rows(args.results)
    make_plot(rows, args.figure)
    args.analysis.write_text(make_analysis(rows, args.figure), encoding="utf-8")
    print(f"analysis: {args.analysis}")
    print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
