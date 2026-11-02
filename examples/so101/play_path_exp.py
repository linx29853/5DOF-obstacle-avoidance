"""在球障碍场景中播放已保存的五关节路径。"""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from so101.collision import has_clearance, is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/so101/path_simplified.npy"))
    parser.add_argument("--duration", type=float, default=8.0, help="播放时长，秒")
    parser.add_argument("--check-only", action="store_true", help="只复核播放帧，不打开窗口")
    parser.add_argument("--scene", help="与规划时相同的场景配置路径")
    parser.add_argument("--margin-mm", type=float, default=0.0, help="安全裕量，毫米")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("播放时长必须为正有限数")
    path = np.load(args.path, allow_pickle=False)
    if path.ndim != 2 or path.shape[1] != 5 or len(path) == 0 or not np.all(np.isfinite(path)):
        raise ValueError("路径必须是非空的 (N, 5) 有限数组")
    margin = args.margin_mm / 1000.0
    model, config, joint_ids = load_collision_scene(scene_path=args.scene, query_margin=margin)
    print(f"安全裕量: {args.margin_mm:g} mm")
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_length = float(lengths.sum())
    frames = [path[0].copy()]
    moments = [0.0]
    elapsed = 0.0
    for start, goal, length in zip(path[:-1], path[1:], lengths):
        segment_duration = args.duration * length / total_length if total_length > 0 else 0.0
        intervals = max(
            1, int(np.ceil(np.max(np.abs(goal - start)) / np.deg2rad(0.5))),
            int(np.ceil(segment_duration * 30)),
        )
        for fraction in np.linspace(0.0, 1.0, intervals + 1)[1:]:
            frames.append((1.0 - fraction) * start + fraction * goal)
            moments.append(elapsed + fraction * segment_duration)
        elapsed += segment_duration
    for index, angles in enumerate(frames):
        _, contacts = inspect_contacts(model, config, joint_ids, angles)
        if not (is_pose_valid(angles, config["joint_limits_rad"], contacts) and has_clearance(contacts, margin)):
            raise ValueError(f"播放帧 {index} 无效，停止播放")
    print(f"路径节点: {len(path)}；播放帧: {len(frames)}；所有帧检查通过")
    if args.check_only:
        return

    from mujoco import viewer as mujoco_viewer

    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[joint_ids[name]] for name in config["arm_joint_names"]]
    gripper_address = model.jnt_qposadr[joint_ids[config["gripper_joint_name"]]]
    data.qpos[addresses] = frames[0]
    data.qpos[gripper_address] = config["gripper_position_rad"]
    mujoco.mj_forward(model, data)
    with mujoco_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.18]
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.sync()
        began = time.perf_counter()
        for angles, moment in zip(frames, moments):
            if not viewer.is_running():
                return
            time.sleep(max(0.0, began + moment - time.perf_counter()))
            with viewer.lock():
                data.qpos[addresses] = angles
                data.qpos[gripper_address] = config["gripper_position_rad"]
                mujoco.mj_forward(model, data)
            viewer.sync()
        print("播放完成，关闭窗口退出。")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
