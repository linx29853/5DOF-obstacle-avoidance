"""输出 SO-101 模型关节信息；可选打开静态姿态预览。"""

import argparse
import time

import mujoco
import numpy as np

from so101.model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="打开静态姿态预览窗口")
    parser.add_argument(
        "--degrees", nargs=5, type=float, default=[0.0] * 5,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5"), help="五个关节的预览角度（度）",
    )
    args = parser.parse_args()

    model, config, joint_ids = load_model()
    data = mujoco.MjData(model)
    names = config["arm_joint_names"]
    for name, angle in zip(names, np.deg2rad(args.degrees)):
        joint_id = joint_ids[name]
        if not np.isfinite(angle) or (model.jnt_limited[joint_id] and not (
            model.jnt_range[joint_id, 0] <= angle <= model.jnt_range[joint_id, 1]
        )):
            parser.error(f"{name} 角度无效或超出模型限位")
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    gripper_id = joint_ids[config["gripper_joint_name"]]
    data.qpos[model.jnt_qposadr[gripper_id]] = config["gripper_position_rad"]
    mujoco.mj_forward(model, data)

    print("关节名称 | 类型 | 限位 (rad) | qpos 索引 | 速度索引 | 用途")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = mujoco.mjtJoint(int(model.jnt_type[joint_id])).name
        limits = model.jnt_range[joint_id] if model.jnt_limited[joint_id] else "无限位"
        role = "机械臂" if name in names else "夹爪" if name == config["gripper_joint_name"] else "其他"
        print(f"{name} | {joint_type} | {limits} | {model.jnt_qposadr[joint_id]} | {model.jnt_dofadr[joint_id]} | {role}")

    if args.view:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
