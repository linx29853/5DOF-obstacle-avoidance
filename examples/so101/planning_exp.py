"""五关节 RRT-Connect 实验，复核并保存规划路径。"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from so101.collision import has_clearance, is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene
from so101.edge_check import is_edge_valid
from so101.planning import rrt_connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty", action="store_true")
    parser.add_argument("--scene", help="场景配置路径")
    parser.add_argument("--margin-mm", type=float, default=0.0, help="安全裕量，毫米")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--start", nargs=5, type=float, default=[-60, 0, 0, 0, 0], help="起点，度")
    parser.add_argument("--goal", nargs=5, type=float, default=[60, 0, 0, 0, 0], help="终点，度")
    parser.add_argument("--output", type=Path, help="可选，保存复核后的路径为 .npy")
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error("迭代预算不能为负")
    margin = args.margin_mm / 1000.0
    model, config, joint_ids = load_collision_scene(not args.empty, scene_path=args.scene, query_margin=margin)
    print(f"安全裕量: {args.margin_mm:g} mm")
    checks = 0

    def pose_valid(angles):
        nonlocal checks
        checks += 1
        _, contacts = inspect_contacts(model, config, joint_ids, angles)
        return is_pose_valid(angles, config["joint_limits_rad"], contacts) and has_clearance(contacts, margin)

    def edge_valid(start, goal):
        return is_edge_valid(start, goal, pose_valid)

    start, goal = np.deg2rad(args.start), np.deg2rad(args.goal)
    print("起点有效:", pose_valid(start), "终点有效:", pose_valid(goal))
    print("直接连接有效:", edge_valid(start, goal))
    checks = 0
    began = perf_counter()
    path = rrt_connect(
        start, goal, config["joint_limits_rad"], pose_valid, edge_valid,
        step_size=np.deg2rad(15), max_iterations=args.iterations, seed=args.seed,
    )
    duration = perf_counter() - began
    print(f"规划耗时: {duration:.3f} s；姿态检查次数: {checks}")
    if path is None:
        print("未找到路径")
        return
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 5 or not np.all(np.isfinite(path)):
        raise ValueError("规划器返回的路径格式不正确")
    if not np.allclose(path[0], start) or not np.allclose(path[-1], goal):
        raise ValueError("路径起终点不正确")
    if not all(pose_valid(angles) for angles in path):
        raise ValueError("路径包含无效构型")
    for first, second in zip(path[:-1], path[1:]):
        if not is_edge_valid(first, second, pose_valid, max_joint_step=np.deg2rad(0.5)):
            raise ValueError("路径未通过 0.5° 连接复核")
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    print(f"路径节点数: {len(path)}；关节路径长度: {length:.6f} rad")
    print("起终点及 0.5° 连接复核通过")
    if args.output:
        if args.output.suffix != ".npy":
            parser.error("输出路径请使用 .npy 扩展名")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, path)
        print("路径已保存:", args.output)


if __name__ == "__main__":
    main()
