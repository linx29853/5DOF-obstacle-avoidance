"""由原始记录生成统计表、图表及中文实验报告。"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(success, total):
    if total == 0:
        return (None, None)
    z = 1.959963984540054
    p = success / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, center - half), min(1.0, center + half)


def read_records(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(directory):
    root = Path(directory)
    records = read_records(root / "records.jsonl")
    groups = defaultdict(list)
    for r in records:
        groups[(r["group"], r["mode"], r["scene"], r["planner"])].append(r)
    summaries = []
    checkpoint_rows = []
    for (group, mode, scene, planner), rs in sorted(groups.items()):
        successes = [r for r in rs if r["success"]]
        times = [r["first_solution_seconds"] for r in successes]
        lo, hi = wilson(len(successes), len(rs))
        summary = {
            "group": group,
            "mode": mode,
            "scene": scene,
            "planner": planner,
            "runs": len(rs),
            "successes": len(successes),
            "success_rate": len(successes) / len(rs),
            "success_ci_low": lo,
            "success_ci_high": hi,
            "first_time_median_s": float(np.median(times)) if times else None,
            "first_time_p90_s": float(np.percentile(times, 90)) if times else None,
        }
        for key in [
            "length_rad",
            "simplified_length_rad",
            "postprocess_seconds",
            "verification_seconds",
        ]:
            values = [r[key] for r in successes]
            summary[key + "_mean"] = float(np.mean(values)) if values else None
        for key in [
            "samples",
            "valid_extensions",
            "pose_checks",
            "edge_checks",
            "nearest_seconds",
            "collision_seconds",
            "seconds",
        ]:
            summary[key + "_mean"] = float(np.mean([r["stats"][key] for r in rs]))
        summary["valid_expansions_per_sample"] = sum(
            r["stats"]["valid_extensions"] for r in rs
        ) / max(1, sum(r["stats"]["samples"] for r in rs))
        summary["accepted_sample_rate"] = sum(
            r["stats"]["accepted_samples"] for r in rs
        ) / max(1, sum(r["stats"]["samples"] for r in rs))
        summaries.append(summary)
        for checkpoint in [1, 3, 10]:
            values = [
                next(
                    s["length_rad"]
                    for s in r["snapshots"]
                    if s["seconds"] == checkpoint
                )
                for r in rs
            ]
            finite = [v for v in values if v is not None]
            checkpoint_rows.append(
                {
                    "group": group,
                    "mode": mode,
                    "scene": scene,
                    "planner": planner,
                    "checkpoint_s": checkpoint,
                    "solved": len(finite),
                    "runs": len(rs),
                    "length_mean_on_solved": float(np.mean(finite)) if finite else None,
                }
            )
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "checkpoints.csv", checkpoint_rows)
    flattened = []
    for r in records:
        row = {
            k: v
            for k, v in r.items()
            if k
            not in ["stats", "path_rad", "history", "snapshots", "options", "budget"]
        }
        row.update(r["stats"])
        row.update({"budget_" + k: v for k, v in r["budget"].items()})
        flattened.append(row)
    write_csv(root / "runs.csv", flattened)
    # 只对四种算法共同成功的同场景/查询/种子/预算做路径质量对比。
    matched = defaultdict(dict)
    for r in records:
        if r["group"] == "single":
            matched[(r["scene"], r["query"], r["seed"], r["mode"])][r["planner"]] = r
    paired = []
    for key, rows in matched.items():
        if len(rows) == 4 and all(r["success"] for r in rows.values()):
            for planner, r in rows.items():
                paired.append(
                    {
                        "scene": key[0],
                        "query": key[1],
                        "seed": key[2],
                        "mode": key[3],
                        "planner": planner,
                        "length_rad": r["length_rad"],
                        "simplified_length_rad": r["simplified_length_rad"],
                    }
                )
    write_csv(root / "paired_success.csv", paired)
    multi = (
        read_records(root / "multiquery.jsonl")
        if (root / "multiquery.jsonl").exists()
        else []
    )
    write_csv(root / "multiquery.csv", multi)
    plot(root, records, multi)
    main = [r for r in summaries if r["group"] == "single" and r["mode"] == "time"]
    lines = [
        "# 五关节采样规划器对比实验",
        "",
        f"原始单次查询及消融记录：{len(records)} 条；重复查询记录：{len(multi)} 条。",
        "",
        "## 成功率与首次解时间",
        "",
        "| 场景 | 算法 | 成功/总数 | 95% Wilson 区间 | 首解中位数 / P90（s） |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in main:
        timing = (
            "—"
            if r["first_time_median_s"] is None
            else f"{r['first_time_median_s']:.4f} / {r['first_time_p90_s']:.4f}"
        )
        lines.append(
            f"| {r['scene']} | {r['planner']} | {r['successes']}/{r['runs']} | {r['success_ci_low']:.3f}～{r['success_ci_high']:.3f} | {timing} |"
        )
    lines += [
        "",
        "## 路径质量与耗时分析",
        "",
        "原始路径与统一简化后的路径分别统计。`paired_success.csv` 保留四种算法共同成功的配对样本，避免成功样本集合不同造成选择偏差。",
        "",
        "![成功率](success.png)",
        "![首解时间](first_solution.png)",
        "![路径代价](cost_progress.png)",
        "![耗时分解](timing.png)",
        "",
        "RRT* 在固定预算内继续选父和重连；有限时间内的路径改善由历史曲线衡量。渐近最优性质描述预算趋于充分时的行为，实验直接比较所测预算内的结果。",
        "",
        "## 重复查询与采样消融",
        "",
        "PRM 重复查询包含一次独立建图时间，后续查询连接端点并运行 Dijkstra；累计时间包括建图。场景、载荷或检查规则变化时路网失效。",
        "",
        "![重复查询累计耗时](multiquery.png)",
        "",
        "目标偏置、KD-tree 与归一化距离分别相对均匀采样、线性最近邻、弧度距离进行单因素比较。各组数据见 summary.csv，路径长度始终以原始弧度报告。",
        "",
        "推荐配置保持基线；只有新增优化在独立正式实例中同时保持成功率、降低首解所需检查量并改善配对耗时后，才调整推荐项。",
        "",
        "## 可复现性与场景解释",
        "",
        "suite.json 固定场景、起终构型与参考路径；参考路径仅用于可解性复核，未传入规划器。正式规划不使用动画中的通道引导点。",
        "",
        "狭窄场景包含明确的几何间隙；全关节规划允许寻找其他可行通路，路径图与失败分布用于解释实际搜索难度。",
        "",
        "metadata.json 记录硬件、依赖、源码校验值及并发设置。时间预算在连接插值内部检查，单次不可中断碰撞计算产生的微小超时另由实际 seconds 字段记录。",
        "",
        "构型检查预算另设 120 秒保护上限；达到保护上限的记录通过 termination 字段与真正耗尽检查数的记录区分。",
        "",
        "五维关节空间中的有效扩展率、最近邻和碰撞耗时用于解释采样效率；维度增长实验可另行增加独立多维基准。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def plot(root, records, multi):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    planners = ["rrt", "rrt_connect", "rrt_star", "prm"]
    scenes = sorted({r["scene"] for r in records})
    base = [r for r in records if r["group"] == "single" and r["mode"] == "time"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, p in enumerate(planners):
        rates = []
        lower = []
        upper = []
        for scene in scenes:
            rs = [r for r in base if r["scene"] == scene and r["planner"] == p]
            n = len(rs)
            s = sum(r["success"] for r in rs)
            lo, hi = wilson(s, n)
            rate = s / n if n else 0
            rates.append(rate)
            lower.append(max(0.0, rate - (lo or 0)))
            upper.append(max(0.0, (hi or 0) - rate))
        ax.bar(
            np.arange(len(scenes)) + j * 0.2,
            rates,
            0.2,
            label=p,
            yerr=[lower, upper],
            capsize=2,
        )
    ax.set(
        xticks=np.arange(len(scenes)) + 0.3,
        xticklabels=scenes,
        ylabel="Success rate (95% Wilson CI)",
        ylim=(0, 1.1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "success.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [
        [
            r["first_solution_seconds"]
            for r in base
            if r["planner"] == p and r["success"]
        ]
        for p in planners
    ]
    ax.boxplot(
        [d if d else [np.nan] for d in data],
        tick_labels=[f"{p}\nn={len(d)}" for p, d in zip(planners, data)],
        showfliers=False,
    )
    ax.set(ylabel="First solution seconds (successful runs)", yscale="log")
    fig.tight_layout()
    fig.savefig(root / "first_solution.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    # 代价随时间曲线使用四算法共同在 1 秒内成功的同一组实例。
    paired = defaultdict(dict)
    for r in base:
        paired[(r["scene"], r["query"], r["seed"])][r["planner"]] = r
    common = [
        v
        for v in paired.values()
        if len(v) == 4
        and all(r["snapshots"][0]["length_rad"] is not None for r in v.values())
    ]
    for p in planners:
        if common:
            ax.plot(
                [1, 3, 10],
                [
                    np.mean(
                        [
                            next(
                                s["length_rad"]
                                for s in v[p]["snapshots"]
                                if s["seconds"] == t
                            )
                            for v in common
                        ]
                    )
                    for t in [1, 3, 10]
                ],
                marker="o",
                label=f"{p} (n={len(common)})",
            )
    ax.set(
        xlabel="Planning budget (seconds)",
        ylabel="Mean path length (rad), paired solved runs",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "cost_progress.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(4)
    for key, label in [
        ("collision_seconds", "Collision"),
        ("nearest_seconds", "Nearest"),
        ("other", "Other"),
    ]:
        values = []
        for p in planners:
            rs = [r for r in base if r["planner"] == p]
            values.append(
                np.mean(
                    [
                        max(
                            0,
                            r["stats"]["seconds"]
                            - r["stats"]["collision_seconds"]
                            - r["stats"]["nearest_seconds"],
                        )
                        if key == "other"
                        else r["stats"][key]
                        for r in rs
                    ]
                )
                if rs
                else 0
            )
        ax.bar(planners, values, bottom=bottom, label=label)
        bottom += values
    ax.set(ylabel="Mean planning seconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "timing.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for p in planners:
        values = []
        for i in range(1, 21):
            rs = [
                r["cumulative_seconds"]
                for r in multi
                if r["planner"] == p and r["query_number"] == i
            ]
            values.append(np.mean(rs) if rs else np.nan)
        ax.plot(range(1, 21), values, label=p)
    ax.set(
        xlabel="Number of queries", ylabel="Mean cumulative seconds, including build"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "multiquery.png", dpi=150)
    plt.close(fig)
