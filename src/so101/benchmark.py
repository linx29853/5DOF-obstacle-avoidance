"""SO-101 固定问题集、独立路径复核与规划器评测。"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from so101.benchmark_resources import check_resources, enable_guard
from so101.collision_scene import load_collision_scene
from so101.model import PROJECT_ROOT
from so101.path_simplification import simplify_path
from so101.planners import (
    Budget,
    Context,
    Exhausted,
    Options,
    Problem,
    Roadmap,
    plan,
    roadmap_key,
)

CONFIG = PROJECT_ROOT / "configs/so101/benchmark.json"
PLANNERS = ["rrt", "rrt_connect", "rrt_star", "prm"]
_ORACLES = OrderedDict()
ORACLE_CACHE_LIMIT = 1


class CollisionOracle:
    """复用 MjData；每次构型检查均重新计算碰撞，不缓存有效性结果。"""

    def __init__(self, scene, margin=0.005, payload=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(scene))
            self.model, self.config, joints = load_collision_scene(
                scene_path=path, query_margin=margin, payload=payload
            )
        self.data = mujoco.MjData(self.model)
        self.addresses = [
            self.model.jnt_qposadr[joints[n]] for n in self.config["arm_joint_names"]
        ]
        self.data.qpos[
            self.model.jnt_qposadr[joints[self.config["gripper_joint_name"]]]
        ] = self.config["gripper_position_rad"]
        self.limits = np.array(self.config["joint_limits_rad"])
        self.margin = margin
        self.payload = payload
        self.environment = [
            self.model.geom(g["name"]).id
            for g in ([scene["table"]] if "table" in scene else [])
            + scene.get("obstacles", [])
        ]
        self.site = self.model.site("gripperframe").id
        moving_bodies = set()
        joint_bodies = {
            int(self.model.jnt_bodyid[joints[n]])
            for n in self.config["arm_joint_names"]
        }
        for body in range(self.model.nbody):
            ancestor = body
            while ancestor:
                if ancestor in joint_bodies:
                    moving_bodies.add(body)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        self.arm_geoms = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] in moving_bodies
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        ]

    def repulsive_potential(self, q, influence=0.08, gain=0.00002):
        """环境表面距离势能；距离扣除统一的 5 mm 裕量。"""
        check_resources()
        self.data.qpos[self.addresses] = q
        mujoco.mj_forward(self.model, self.data)
        value = 0.0
        for arm in self.arm_geoms:
            for obstacle in self.environment:
                distance = (
                    mujoco.mj_geomDistance(
                        self.model,
                        self.data,
                        arm,
                        obstacle,
                        influence + self.margin,
                        None,
                    )
                    - self.margin
                )
                if distance < influence:
                    distance = max(float(distance), 0.0001)
                    value += 0.5 * gain * (1 / distance - 1 / influence) ** 2
        return value

    def __call__(self, q):
        check_resources()
        if (
            not np.all(np.isfinite(q))
            or np.any(q < self.limits[:, 0])
            or np.any(q > self.limits[:, 1])
        ):
            return False
        self.data.qpos[self.addresses] = q
        mujoco.mj_forward(self.model, self.data)
        return not any(c.dist <= self.margin for c in self.data.contact)


def problem_for(scene, query, payload=None):
    cache_key = json.dumps([scene, payload], sort_keys=True)
    check_resources()
    if cache_key not in _ORACLES:
        while len(_ORACLES) >= ORACLE_CACHE_LIMIT:
            _ORACLES.popitem(last=False)
        _ORACLES[cache_key] = CollisionOracle(scene, payload=payload)
    else:
        _ORACLES.move_to_end(cache_key)
    oracle = _ORACLES[cache_key]
    key = hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()
    return Problem(
        query["start_rad"],
        query["goal_rad"],
        oracle.limits,
        oracle,
        key,
        json.dumps(payload, sort_keys=True),
    )


def verify(problem, path, resolution=0.008726646259971648):
    if path is None:
        return False
    q = np.asarray(path)
    if q.ndim != 2 or q.shape[1] != 5 or len(q) == 0 or not np.all(np.isfinite(q)):
        return False
    if not np.allclose(q[0], problem.start, atol=1e-10, rtol=0) or not np.allclose(
        q[-1], problem.goal, atol=1e-10, rtol=0
    ):
        return False
    if not all(problem.pose_valid(x) for x in q):
        return False
    for a, b in pairwise(q):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / resolution)))
        if not all(
            problem.pose_valid(a + (b - a) * u) for u in np.linspace(0, 1, n + 1)
        ):
            return False
    return True


def generate_suite(path=CONFIG):
    base = json.loads((PROJECT_ROOT / "configs/so101/demo.json").read_text())
    common = {k: base[k] for k in ["frame", "units", "table", "base_collision"]}
    layouts = {
        "empty": [],
        "single": [
            {
                "name": "sphere",
                "type": "sphere",
                "center": [0.15, 0, 0.20],
                "radius": 0.04,
            }
        ],
        "dense": base["cases"]["multi"]["obstacles"]
        + [
            {
                "name": "column",
                "type": "box",
                "center": [0.23, 0.12, 0.16],
                "half_size": [0.025, 0.025, 0.10],
            }
        ],
        "narrow": [
            {
                "name": "gate_left",
                "type": "box",
                "center": [0.24, -0.07, 0.20],
                "half_size": [0.045, 0.035, 0.15],
            },
            {
                "name": "gate_right",
                "type": "box",
                "center": [0.24, 0.07, 0.20],
                "half_size": [0.045, 0.035, 0.15],
            },
        ],
        "staggered": [
            {
                "name": "baffle_left",
                "type": "box",
                "center": [0.17, -0.035, 0.05],
                "half_size": [0.055, 0.008, 0.05],
            },
            {
                "name": "baffle_right",
                "type": "box",
                "center": [0.255, 0.04, 0.05],
                "half_size": [0.055, 0.008, 0.05],
            },
        ],
    }
    suite = {
        "schema_version": 1,
        "scene_seed": 20261106,
        "margin_m": 0.005,
        "planning_resolution_deg": 0.5,
        "verification_resolution_deg": 0.5,
        "formal_seeds": list(range(30)),
        "tuning_seeds": [100, 101, 102],
        "time_checkpoints_s": [1, 3, 10],
        "check_budgets": [10000, 50000],
        "options": asdict(Options()),
        "payload": {"radius_m": 0.027, "offset_m": [0, 0, 0]},
        "scenes": [],
    }
    for index, (name, obstacles) in enumerate(layouts.items()):
        scene = {**common, "obstacles": obstacles}
        oracle = CollisionOracle(scene)
        payload_oracle = CollisionOracle(scene, payload=suite["payload"])
        rng = np.random.default_rng(20261106 + index)
        queries = []
        # 25 个固定查询：5 个正式、20 个路网复用查询；3 个单独调参查询。
        for query_id in range(28):
            for attempt in range(100):
                start = np.deg2rad([-60, 0, 0, 0, 0]) + rng.normal(0, 0.07, 5)
                goal = np.deg2rad([60, 0, 0, 0, 0]) + rng.normal(0, 0.07, 5)
                problem = Problem(start, goal, oracle.limits, oracle)
                result = plan(
                    problem,
                    "rrt_connect",
                    Budget(max_checks=30000),
                    seed=2000 + query_id + attempt,
                )
                if result.path is None or not verify(problem, result.path):
                    continue
                if query_id < 5:
                    pp = Problem(start, goal, payload_oracle.limits, payload_oracle)
                    payload_result = plan(
                        pp,
                        "rrt_connect",
                        Budget(max_checks=50000),
                        seed=3000 + query_id + attempt,
                    )
                    if payload_result.path is None or not verify(
                        pp, payload_result.path
                    ):
                        continue
                break
            else:
                raise RuntimeError(f"{name}: 无法生成可解实例")
            split = (
                "formal"
                if query_id < 5
                else "multiquery"
                if query_id < 25
                else "tuning"
            )
            queries.append(
                {
                    "id": f"{name}-{query_id:02d}",
                    "split": split,
                    "start_rad": start.tolist(),
                    "goal_rad": goal.tolist(),
                    "reference_path_rad": np.asarray(result.path).tolist(),
                    "reference_verified": True,
                    **(
                        {
                            "payload_reference_path_rad": np.asarray(
                                payload_result.path
                            ).tolist()
                        }
                        if query_id < 5
                        else {}
                    ),
                }
            )
        suite["scenes"].append({"id": name, "scene": scene, "queries": queries})
        print("固定实例已生成:", name, flush=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n")
    return suite


def metadata(config):
    sources = list((PROJECT_ROOT / "src/so101").glob("*.py")) + [Path(config)]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "hardware": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        )
        .decode()
        .strip(),
        "source_sha256": {
            str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }


def measure(
    problem, planner, budget, seed, options, checkpoints=(1, 3, 10), roadmap=None
):
    result = plan(problem, planner, budget, seed, options, roadmap)
    began = perf_counter()
    valid = verify(problem, result.path) if result.path is not None else False
    validation_seconds = perf_counter() - began
    rejected_path = (
        np.asarray(result.path).tolist()
        if result.path is not None and not valid
        else None
    )
    simplified = None
    post_seconds = 0.0
    post_checks = 0
    if valid:

        def counted(q):
            nonlocal post_checks
            post_checks += 1
            return problem.pose_valid(q)

        from so101.edge_check import is_edge_valid

        began = perf_counter()
        simplified = simplify_path(
            result.path,
            lambda a, b: is_edge_valid(a, b, counted, max_joint_step=np.deg2rad(0.5)),
        )
        post_seconds = perf_counter() - began
        began = perf_counter()
        if not verify(problem, simplified):
            raise RuntimeError("简化路径未通过独立复核")
        validation_seconds += perf_counter() - began
    history = result.history if valid else []
    snapshots = [
        {
            "seconds": t,
            "length_rad": next(
                (h["length_rad"] for h in reversed(history) if h["seconds"] <= t), None
            ),
        }
        for t in checkpoints
    ]
    return {
        "planner": planner,
        "seed": seed,
        "budget": asdict(budget),
        "options": asdict(options),
        "success": valid,
        "reason": result.reason if rejected_path is None else "verification_failed",
        "rejected_path_rad": rejected_path,
        "first_solution_seconds": result.first_solution_seconds if valid else None,
        "first_solution_checks": result.first_solution_checks if valid else None,
        "length_rad": history[-1]["length_rad"] if history else None,
        "nodes": len(result.path) if valid else 0,
        "simplified_length_rad": float(
            np.linalg.norm(np.diff(simplified, axis=0), axis=1).sum()
        )
        if valid and len(simplified) > 1
        else 0.0
        if valid
        else None,
        "postprocess_seconds": post_seconds,
        "postprocess_checks": post_checks,
        "verification_seconds": validation_seconds,
        "stats": result.stats,
        "history": history,
        "snapshots": snapshots,
        "path_rad": np.asarray(result.path).tolist() if valid else None,
    }


def tasks_for(suite, profile):
    tasks = []
    base_options = Options(**suite["options"])
    for scene in suite["scenes"]:
        formal = [q for q in scene["queries"] if q["split"] == "formal"]
        queries = formal[:1] if profile == "smoke" else formal
        seeds = range(3) if profile == "smoke" else suite["formal_seeds"]
        modes = [("time", Budget(seconds=0.15 if profile == "smoke" else 10))]
        if profile == "full":
            modes += [
                (f"checks-{n}", Budget(max_checks=n, seconds=120))
                for n in suite["check_budgets"]
            ]
        for query in queries:
            for seed in seeds:
                for planner in PLANNERS:
                    for mode, budget in modes:
                        tasks.append(
                            (
                                scene,
                                query,
                                planner,
                                budget,
                                seed,
                                base_options,
                                mode,
                                "single",
                            )
                        )
        if profile == "full":
            payload_scene = {**scene, "payload": suite["payload"]}
            for query in formal:
                for seed in suite["formal_seeds"]:
                    for planner in PLANNERS:
                        tasks.append(
                            (
                                payload_scene,
                                query,
                                planner,
                                Budget(seconds=10),
                                seed,
                                base_options,
                                "time",
                                "payload",
                            )
                        )
            variants = [
                ("goal-5", Options(**{**suite["options"], "goal_bias": 0.05})),
                ("goal-15", Options(**{**suite["options"], "goal_bias": 0.15})),
                ("kdtree", Options(**{**suite["options"], "nearest": "kdtree"})),
                ("normalized", Options(**{**suite["options"], "metric": "normalized"})),
            ]
            for query in formal:
                for seed in suite["formal_seeds"]:
                    for label, options in variants:
                        tasks.append(
                            (
                                scene,
                                query,
                                "rrt_connect",
                                Budget(seconds=10),
                                seed,
                                options,
                                "time",
                                "ablation-" + label,
                            )
                        )
    return tasks


def execute_task(task):
    scene, query, planner, budget, seed, options, mode, group = task
    record = measure(
        problem_for(scene["scene"], query, scene.get("payload")),
        planner,
        budget,
        seed,
        options,
    )
    record.update(scene=scene["id"], query=query["id"], mode=mode, group=group)
    record["worker_affinity"] = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return record


def run_multiquery(suite, seed, seconds=10):
    records = []
    for scene in suite["scenes"]:
        queries = [q for q in scene["queries"] if q["split"] == "multiquery"]
        for planner in PLANNERS:
            cumulative = 0.0
            road = None
            build_seconds = 0.0
            nn_start = 0.0
            if planner == "prm":
                p = problem_for(scene["scene"], queries[0])
                options = Options()
                road = Roadmap(roadmap_key(p, options), np.ones(5), "linear")
                context = Context(p, Budget(seconds=seconds), options, seed)
                from so101.planners import _prm

                began = perf_counter()
                try:
                    _prm(context, road)
                except Exhausted:
                    pass
                build_seconds = perf_counter() - began
                cumulative = build_seconds
                nn_start = road.nn.seconds
            for number, query in enumerate(queries):
                p = problem_for(scene["scene"], query)
                if planner == "prm":
                    # 复用实验只连接查询点并运行 Dijkstra，不扩充采样路网。
                    options = Options()
                    context = Context(
                        p, Budget(seconds=seconds), options, seed + number
                    )
                    context.indices.append(road.nn)
                    began = perf_counter()
                    try:
                        if not context.valid(p.start) or not context.valid(p.goal):
                            raise ValueError("无效查询")
                        ids = []
                        for q in [p.start, p.goal]:
                            neighbors = road.nn.query(q, 15)
                            i = road.add(q)
                            ids.append(i)
                            for j in neighbors:
                                if context.edge(q, road.nodes[j]):
                                    road.connect(i, j)
                        path = road.shortest(*ids, context.gate)
                        if path is not None:
                            context.consider(path)
                    except Exhausted:
                        pass
                    elapsed = perf_counter() - began
                    if context.result.path is not None and not verify(
                        p, context.result.path
                    ):
                        raise RuntimeError("复用查询路径错误")
                    record = {
                        "planner": planner,
                        "success": context.result.path is not None,
                        "query_seconds": elapsed,
                        "build_seconds": build_seconds,
                        "nearest_seconds": road.nn.seconds - nn_start,
                        "query_nodes": len(road.nodes),
                    }
                    nn_start = road.nn.seconds
                else:
                    result = measure(
                        p,
                        planner,
                        Budget(seconds=seconds),
                        seed + number,
                        Options(stop_on_first=True),
                    )
                    elapsed = result["stats"]["seconds"]
                    record = {
                        "planner": planner,
                        "success": result["success"],
                        "query_seconds": elapsed,
                        "build_seconds": 0.0,
                    }
                cumulative += elapsed
                record.update(
                    group="multiquery",
                    scene=scene["id"],
                    query=query["id"],
                    seed=seed,
                    query_number=number + 1,
                    cumulative_seconds=cumulative,
                )
                records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/so101/benchmark"
    )
    parser.add_argument(
        "--workers", type=int, choices=[1], default=1, help="本地安全模式固定单进程"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    guard = enable_guard()
    if args.generate:
        generate_suite(args.config)
        return
    suite = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    info = metadata(args.config)
    info.update(
        workers=args.workers,
        profile=args.profile,
        memory_guard={
            "min_available_bytes": guard.min_available,
            "max_rss_bytes": guard.max_rss,
            "oracle_cache_limit": ORACLE_CACHE_LIMIT,
        },
    )
    metadata_path = args.output / "metadata.json"
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (
            previous["source_sha256"] != info["source_sha256"]
            or previous["profile"] != info["profile"]
            or previous["workers"] != info["workers"]
        ):
            raise ValueError("恢复运行要求源码、配置、profile 和 workers 与原记录一致")
    (args.output / "metadata.json").write_text(json.dumps(info, indent=2))
    (args.output / "suite.json").write_text(json.dumps(suite, indent=2))
    tasks = tasks_for(suite, args.profile)
    import random

    random.Random(321).shuffle(tasks)
    target = args.output / "records.jsonl"
    completed = set()
    if args.resume and target.exists():
        for line in target.read_text().splitlines():
            r = json.loads(line)
            completed.add(
                (r["scene"], r["query"], r["planner"], r["seed"], r["mode"], r["group"])
            )
        tasks = [
            t
            for t in tasks
            if (t[0]["id"], t[1]["id"], t[2], t[4], t[6], t[7]) not in completed
        ]
    mode = "a" if args.resume else "w"
    with target.open(mode) as stream:
        if args.workers == 1:
            iterator = map(execute_task, tasks)
        else:
            from concurrent.futures import ProcessPoolExecutor

            pool = ProcessPoolExecutor(args.workers, initializer=pin_worker)
            iterator = pool.map(execute_task, tasks, chunksize=1)
        for i, record in enumerate(iterator):
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(tasks)} 已完成", flush=True)
        if args.workers != 1:
            pool.shutdown()
    multi = args.output / "multiquery.jsonl"
    multi_seeds = list(range(1) if args.profile == "smoke" else suite["formal_seeds"])
    old = []
    if args.resume and multi.exists():
        old = [
            json.loads(line) for line in multi.read_text().splitlines() if line.strip()
        ]
    complete = {s for s in multi_seeds if sum(r["seed"] == s for r in old) == 400}
    with multi.open("w") as stream:
        for r in old:
            if r["seed"] in complete:
                stream.write(json.dumps(r) + "\n")
        remaining = [s for s in multi_seeds if s not in complete]
        if args.workers == 1:
            iterator = (
                run_multiquery(suite, s, 0.15 if args.profile == "smoke" else 10)
                for s in remaining
            )
        else:
            from concurrent.futures import ProcessPoolExecutor

            pool = ProcessPoolExecutor(args.workers, initializer=pin_worker)
            iterator = pool.map(
                multi_job,
                [
                    (suite, s, 0.15 if args.profile == "smoke" else 10)
                    for s in remaining
                ],
            )
        for rs in iterator:
            for record in rs:
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print("重复查询种子完成", rs[0]["seed"], flush=True)
        if args.workers != 1:
            pool.shutdown()
    from so101.benchmark_report import generate_report

    generate_report(args.output)


def multi_job(args):
    return run_multiquery(*args)


def pin_worker():
    if hasattr(os, "sched_setaffinity"):
        import multiprocessing

        cpus = sorted(os.sched_getaffinity(0))
        index = multiprocessing.current_process()._identity[0] - 1
        os.sched_setaffinity(0, {cpus[index % len(cpus)]})


if __name__ == "__main__":
    main()
