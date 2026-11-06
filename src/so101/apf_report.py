"""从固定场景记录生成 APF 与采样规划器对比报告。"""

import json
from pathlib import Path

import numpy as np


def generate(directory):
    directory = Path(directory)
    rows = [
        json.loads(line)
        for line in (directory / "records.jsonl").read_text().splitlines()
    ]
    labels = {
        "empty": "空场",
        "single": "单障碍",
        "dense": "密集障碍",
        "narrow": "狭窄通道",
        "staggered": "交错挡板",
    }
    names = {"apf": "APF", "rrt": "单树 RRT", "rrt_connect": "RRT-Connect"}
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    planners = [p for p in names if any(r["planner"] == p for r in rows)]
    values = {p: {"time": [], "length": [], "success": []} for p in planners}
    table = [
        "| 场景 | 算法 | 成功次数 / 运行次数 | 首解时间中位数（秒） | 原始长度中位数（rad） | 简化长度中位数（rad） |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def display(x):
        return "—" if not np.isfinite(x) else f"{x:.4f}"

    for scene in scenes:
        for planner in planners:
            subset = [
                r for r in rows if r["scene"] == scene and r["planner"] == planner
            ]
            good = [r for r in subset if r["success"]]
            values[planner]["success"].append(len(good) / len(subset))
            medians = {}
            for key, field in [
                ("time", "first_solution_seconds"),
                ("length", "length_rad"),
                ("simplified", "simplified_length_rad"),
            ]:
                medians[key] = (
                    float(np.median([r[field] for r in good])) if good else float("nan")
                )
                if key in values[planner]:
                    values[planner][key].append(medians[key])
            table.append(
                f"| {labels[scene]} | {names[planner]} | {len(good)}/{len(subset)} | {display(medians['time'])} | {display(medians['length'])} | {display(medians['simplified'])} |"
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(scenes))
    width = 0.8 / len(planners)
    for ax, key, title in zip(
        axes,
        ["success", "time", "length"],
        ["Success fraction", "First solution (s)", "Raw joint path length (rad)"],
    ):
        for index, planner in enumerate(planners):
            ax.bar(
                x + (index - (len(planners) - 1) / 2) * width,
                values[planner][key],
                width=width,
                label=planner,
            )
        ax.set_xticks(x, scenes, rotation=25)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)
    totals = []
    for planner in planners:
        subset = [r for r in rows if r["planner"] == planner]
        totals.append(
            f"{names[planner]}：{sum(r['success'] for r in subset)}/{len(subset)} 成功"
        )
    text = f"""# SO-101：{"、".join(names[p] for p in planners)} 对比

## 实验设置

每类场景选取第一个 formal 实例，共 {len(scenes)} 个任务。APF 为确定性算法，每任务运行一次；单树 RRT 与 RRT-Connect 每任务使用 0、1、2 三个种子。共 {len(rows)} 次运行。

所有算法共享五关节模型、起终构型、关节限位、弧度欧氏距离、15° 最大扩展步长及碰撞有效性接口。每次规划上限 {rows[0]["budget"]["seconds"]} 秒，找到首解后停止。规划连接及独立路径复核使用 0.5° 插值与 5 mm 裕量。路径简化采用同一过程；规划、后处理和复核分别计时。参考路径只用于实例可解性验证。

单树 RRT 从起点生长，默认目标偏置 5%；RRT-Connect 双树交替扩展，采用均匀采样。此设置比较两种算法的默认策略，目标偏置差异单独记录在 options 中。

APF 使用关节空间二次吸引势能和运动连杆到桌面、环境障碍的表面距离斥势能。斥力影响范围 0.08 m、增益 0.00002，距离扣除 5 mm 裕量；梯度差分间隔 0.001 rad，回溯最多 14 次。自碰撞通过共享有效性检查拒绝。无随机重启、引导点或采样算法兜底。

APF 势能评估计入构型资源统计，单列 potential_evaluations 和 potential_seconds。不同查询成本以墙钟时间反映。

## 实测结果

{"；".join(totals)}。

表中时间和长度统计只包含成功运行，失败样本保留在成功次数分母中。APF 每项 n=1，采样算法每项最多 n=3。

{chr(10).join(table)}

![对比图](comparison.png)

## 分析与扩展

原始路径反映首解质量，简化后长度反映相同后处理下的质量。两者分别比较；采样算法的首解路径会随种子改变。

APF 的 local_minimum_or_blocked 表示回溯无法找到同时降低势能且通过碰撞检查的下一步。空场及本轮交错挡板任务接近直连；后续增加必须穿越挡板的起终构型，并扩展每类任务数量。成功率按任务和种子分别记录。

## 复现与原始数据

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/so101/compare_apf.py --output outputs/so101/apf-rrt-comparison
```

单进程执行，可用内存低于 3 GiB 或进程 RSS 超过 1.5 GiB 时停止。每条记录完成后写盘。

- records.jsonl：路径、种子、预算、失败原因、耗时与计数。
- summary.csv：逐次运行结果。
- metadata.json：依赖、硬件、源码提交与文件哈希。
"""
    (directory / "report.md").write_text(text)
