"""检查规划路径某一段的五关节时间轨迹。"""

import argparse
from pathlib import Path

import numpy as np

from so101.collision import has_clearance, is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene
from so101.edge_check import is_edge_valid
from so101.trajectory import quintic_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/so101/margin/path_simplified.npy"))
    parser.add_argument("--segment", type=int, default=0)
    args = parser.parse_args()
    path = np.load(args.path, allow_pickle=False)
    if path.ndim != 2 or path.shape[1] != 5 or not np.all(np.isfinite(path)):
        raise ValueError("路径必须是 (N, 5) 有限数组")
    if not 0 <= args.segment < len(path) - 1:
        parser.error("段编号必须位于 0 到路径节点数减 2 之间")
    start, goal = path[args.segment:args.segment + 2]
    velocity_limits = np.array([0.5, 0.4, 0.4, 0.6, 0.8])
    acceleration_limits = np.array([1.0, 0.8, 0.8, 1.2, 1.6])
    print("使用教学仿真速度与加速度限制，单位 rad/s、rad/s²")
    try:
        times, positions, velocities, accelerations = quintic_trajectory(
            start, goal, velocity_limits, acceleration_limits, sample_count=1001,
        )
    except NotImplementedError as error:
        print(f"尚未完成：{error}；请填写 src/so101/trajectory.py")
        return
    times = np.asarray(times)
    positions, velocities, accelerations = map(np.asarray, (positions, velocities, accelerations))
    if times.shape != (1001,) or not np.all(np.isfinite(times)):
        raise ValueError("时间数组形状或数值错误")
    for values in (positions, velocities, accelerations):
        if values.shape != (1001, 5) or not np.all(np.isfinite(values)):
            raise ValueError("轨迹数组必须是 (1001, 5) 有限数组")
    if not np.isclose(times[0], 0) or np.any(np.diff(times) < 0):
        raise ValueError("时间必须从零开始且不递减")
    if not np.array_equal(start, goal) and np.any(np.diff(times) <= 0):
        raise ValueError("非零位移的轨迹时间必须严格递增")
    if not np.allclose(positions[0], start) or not np.allclose(positions[-1], goal):
        raise ValueError("轨迹起终点错误")
    if not np.allclose(velocities[[0, -1]], 0) or not np.allclose(accelerations[[0, -1]], 0):
        raise ValueError("起终点速度和加速度应为零")
    peak_velocities = np.max(np.abs(velocities), axis=0)
    peak_accelerations = np.max(np.abs(accelerations), axis=0)
    if np.any(peak_velocities > velocity_limits + 1e-8) or np.any(peak_accelerations > acceleration_limits + 1e-8):
        raise ValueError("速度或加速度超限")
    model, config, joint_ids = load_collision_scene(
        scene_path="configs/so101/scene_table_multi.json", query_margin=0.005,
    )

    def pose_valid(angles):
        _, contacts = inspect_contacts(model, config, joint_ids, angles)
        return is_pose_valid(angles, config["joint_limits_rad"], contacts) and has_clearance(contacts, 0.005)

    if not all(pose_valid(angles) for angles in positions):
        raise ValueError("轨迹采样点不满足碰撞或 5 mm 裕量要求")
    if not all(
        is_edge_valid(first, second, pose_valid, max_joint_step=np.deg2rad(0.5))
        for first, second in zip(positions[:-1], positions[1:])
    ):
        raise ValueError("轨迹采样点之间的连接复核失败")
    print(f"段编号: {args.segment}；时长: {times[-1]:.3f} s")
    print("各关节峰值速度:", peak_velocities)
    print("各关节峰值加速度:", peak_accelerations)
    print("起终点、速度/加速度限制和 5 mm 裕量复核通过")


if __name__ == "__main__":
    main()
