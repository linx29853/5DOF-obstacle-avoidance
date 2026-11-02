"""在固定场景和随机种子下评测五关节规划。"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np

from so101.collision import has_clearance, is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene
from so101.edge_check import is_edge_valid
from so101.model import PROJECT_ROOT
from so101.planning import rrt_connect


def run_case(name, model, config, joint_ids, margin, start, goal, seed, budget):
    checks = 0

    def pose_valid(angles):
        nonlocal checks
        checks += 1
        _, contacts = inspect_contacts(model, config, joint_ids, angles)
        return is_pose_valid(angles, config["joint_limits_rad"], contacts) and has_clearance(contacts, margin)

    def edge_valid(first, second):
        return is_edge_valid(first, second, pose_valid)

    began = perf_counter()
    path = rrt_connect(
        start, goal, config["joint_limits_rad"], pose_valid, edge_valid,
        step_size=np.deg2rad(15), max_iterations=budget, seed=seed,
    )
    duration = perf_counter() - began
    planning_checks = checks
    record = {
        "case": name, "seed": seed, "budget": budget,
        "start_rad": start.tolist(), "goal_rad": goal.tolist(),
        "found": path is not None, "seconds": duration, "planning_pose_checks": planning_checks,
        "verified": False, "nodes": 0, "length_rad": None,
    }
    if path is not None:
        path = np.asarray(path)
        if path.ndim != 2 or path.shape[1] != 5 or len(path) == 0 or not np.all(np.isfinite(path)):
            raise ValueError(f"{name}/{seed}: 路径格式错误")
        verified = np.allclose(path[0], start) and np.allclose(path[-1], goal)
        verified = verified and all(pose_valid(angles) for angles in path)
        verified = verified and all(
            is_edge_valid(first, second, pose_valid, max_joint_step=np.deg2rad(0.5))
            for first, second in zip(path[:-1], path[1:])
        )
        if not verified:
            raise ValueError(f"{name}/{seed}: 路径复核失败")
        record.update(
            verified=True, nodes=len(path),
            length_rad=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        )
    record["verification_pose_checks"] = checks - planning_checks
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="使用 0 到 seeds-1 的种子")
    parser.add_argument("--margin-mm", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/so101/evaluation.json"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("种子数量至少为 1")
    margin = args.margin_mm / 1000.0
    start, goal = np.deg2rad([-60, 0, 0, 0, 0]), np.deg2rad([60, 0, 0, 0, 0])
    scenes = [
        ("empty", False, None),
        ("sphere", True, None),
        ("table_multi", True, "configs/so101/scene_table_multi.json"),
    ]
    records = []
    for name, with_obstacle, scene_path in scenes:
        model, config, joint_ids = load_collision_scene(with_obstacle, scene_path, query_margin=margin)
        scene_records = []
        for seed in range(args.seeds):
            record = run_case(name, model, config, joint_ids, margin, start, goal, seed, 1000)
            scene_records.append(record)
        records.extend(scene_records)
        success = sum(record["verified"] for record in scene_records)
        mean_seconds = np.mean([record["seconds"] for record in scene_records])
        print(f"{name}: {success}/{args.seeds} 成功并通过复核；平均 {mean_seconds:.3f} s", flush=True)

    model, config, joint_ids = load_collision_scene(query_margin=margin)
    invalid = np.deg2rad([180, 0, 0, 0, 0])
    boundaries = [
        ("invalid_start", invalid, goal, 1000, False),
        ("invalid_goal", start, invalid, 1000, False),
        ("same_pose", start, start, 1000, True),
        ("zero_budget", start, goal, 0, False),
    ]
    for name, first, second, budget, expected_found in boundaries:
        record = run_case(name, model, config, joint_ids, margin, first, second, 42, budget)
        record["expected_found"] = expected_found
        if record["found"] != expected_found:
            raise ValueError(f"{name}: 结果与预期不符")
        if name == "same_pose" and record["nodes"] != 1:
            raise ValueError("相同起终点应返回单节点路径")
        records.append(record)
        print(f"{name}: 通过", flush=True)

    sources = sorted((PROJECT_ROOT / "src/so101").glob("*.py"))
    sources += sorted((PROJECT_ROOT / "configs/so101").glob("*.json"))
    sources += [PROJECT_ROOT / "models/so101/manifest.json", Path(__file__).resolve()]
    report = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "margin_m": margin, "step_size_rad": float(np.deg2rad(15)),
        "edge_resolution_rad": float(np.deg2rad(1)),
        "verification_resolution_rad": float(np.deg2rad(0.5)),
        "versions": {name: version(name) for name in ("numpy", "mujoco")},
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("评测记录已保存:", args.output)


if __name__ == "__main__":
    main()
