"""读取球障碍场景的规划路径，调用学习者的路径简化并复核。"""

import argparse
from pathlib import Path

import numpy as np

from so101.collision import has_clearance, is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene
from so101.edge_check import is_edge_valid
from so101.path_simplification import simplify_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/so101/path.npy"))
    parser.add_argument("--output", type=Path, default=Path("outputs/so101/path_simplified.npy"))
    parser.add_argument("--scene", help="与规划时相同的场景配置路径")
    parser.add_argument("--margin-mm", type=float, default=0.0, help="安全裕量，毫米")
    args = parser.parse_args()
    path = np.load(args.path, allow_pickle=False)
    if path.ndim != 2 or path.shape[1] != 5 or len(path) == 0 or not np.all(np.isfinite(path)):
        raise ValueError("输入路径必须是非空的 (N, 5) 有限数组")
    margin = args.margin_mm / 1000.0
    model, config, joint_ids = load_collision_scene(scene_path=args.scene, query_margin=margin)
    print(f"安全裕量: {args.margin_mm:g} mm")

    def pose_valid(angles):
        _, contacts = inspect_contacts(model, config, joint_ids, angles)
        return is_pose_valid(angles, config["joint_limits_rad"], contacts) and has_clearance(contacts, margin)

    def edge_valid(start, goal):
        return is_edge_valid(start, goal, pose_valid)

    def verify(points):
        if not all(pose_valid(angles) for angles in points):
            raise ValueError("路径中包含无效构型")
        for start, goal in zip(points[:-1], points[1:]):
            if not is_edge_valid(start, goal, pose_valid, max_joint_step=np.deg2rad(0.5)):
                raise ValueError("路径连接未通过 0.5° 复核")

    verify(path)
    input_path = [angles.copy() for angles in path]
    try:
        simplified = np.asarray(simplify_path(input_path, edge_valid))
    except NotImplementedError as error:
        print(f"尚未完成：{error}；请填写 src/so101/path_simplification.py")
        return
    if not np.array_equal(np.asarray(input_path), path):
        raise ValueError("简化函数修改了输入路径")
    if simplified.ndim != 2 or simplified.shape[1] != 5 or len(simplified) == 0 or not np.all(np.isfinite(simplified)):
        raise ValueError("简化后必须返回非空的五关节路径")
    if not np.allclose(simplified[0], path[0]) or not np.allclose(simplified[-1], path[-1]):
        raise ValueError("简化改变了起终点")
    verify(simplified)
    length_before = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    length_after = np.linalg.norm(np.diff(simplified, axis=0), axis=1).sum()
    if len(simplified) > len(path) or length_after > length_before + 1e-10:
        raise ValueError("简化增加了路点或路径长度")
    print(f"节点数: {len(path)} → {len(simplified)}")
    print(f"关节路径长度: {length_before:.6f} → {length_after:.6f} rad")
    print("起终点、输入不变及 0.5° 连接复核通过")
    if args.output.suffix != ".npy":
        parser.error("输出路径请使用 .npy 扩展名")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, simplified)
    print("简化路径已保存:", args.output)


if __name__ == "__main__":
    main()
