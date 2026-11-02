"""静态球障碍接触检查；有效性判断由学习者实现。"""

import argparse
import time

import numpy as np

from so101.collision import is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", nargs=5, type=float, default=[0.0] * 5)
    parser.add_argument("--empty", action="store_true", help="不添加球障碍，仍保留机器人自碰撞检查")
    parser.add_argument("--view", action="store_true", help="打开静态姿态窗口")
    parser.add_argument("--scene", help="场景配置路径")
    args = parser.parse_args()
    model, config, joint_ids = load_collision_scene(with_obstacle=not args.empty, scene_path=args.scene)
    angles = np.deg2rad(args.degrees)
    data, contacts = inspect_contacts(model, config, joint_ids, angles)
    print("角度 (deg):", args.degrees)
    print("检测到的接触数:", len(contacts))
    for contact in contacts:
        print(f"{contact['geom1']} ↔ {contact['geom2']}: dist={contact['dist']:.6f} m")
    try:
        print("姿态有效:", is_pose_valid(angles, config["joint_limits_rad"], contacts))
    except NotImplementedError as error:
        print(f"尚未完成：{error}；请填写 src/so101/collision.py")
    if args.view:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
