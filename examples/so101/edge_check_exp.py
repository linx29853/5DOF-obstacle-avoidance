"""验证学习者的五关节连接检查。"""

import numpy as np

from so101.collision import is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene
from so101.edge_check import is_edge_valid


def main():
    cases = [
        ("空场连接", False, [-60, 0, 0, 0, 0], [60, 0, 0, 0, 0]),
        ("端点安全、中途撞球", True, [-60, 0, 0, 0, 0], [60, 0, 0, 0, 0]),
        ("相同安全构型", True, [-60, 0, 0, 0, 0], [-60, 0, 0, 0, 0]),
        ("终点超限位", False, [0, 0, 0, 0, 0], [180, 0, 0, 0, 0]),
    ]
    for name, with_obstacle, start_degrees, goal_degrees in cases:
        model, config, joint_ids = load_collision_scene(with_obstacle)
        checked = []

        def pose_valid(angles):
            _, contacts = inspect_contacts(model, config, joint_ids, angles)
            valid = is_pose_valid(angles, config["joint_limits_rad"], contacts)
            checked.append(valid)
            return valid

        start = np.deg2rad(start_degrees)
        goal = np.deg2rad(goal_degrees)
        print(f"\n{name}")
        print("起点有效:", pose_valid(start), "终点有效:", pose_valid(goal))
        checked.clear()
        try:
            valid = is_edge_valid(start, goal, pose_valid)
        except NotImplementedError as error:
            print(f"尚未完成：{error}；请填写 src/so101/edge_check.py")
            return
        print("连接有效:", valid, "采样检查次数:", len(checked))


if __name__ == "__main__":
    main()
