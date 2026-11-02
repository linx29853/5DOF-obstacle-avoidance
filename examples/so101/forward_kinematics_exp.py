"""将学习者正运动学与 MuJoCo 的 gripperframe 位姿作对照。"""

import argparse

import mujoco
import numpy as np

from so101.forward_kinematics import forward_kinematics
from so101.model import kinematic_parameters, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", action="store_true", help="仅查看模型固定变换")
    args = parser.parse_args()
    model, config, joint_ids = load_model()
    fixed_transforms, tool_transform = kinematic_parameters(model, config, joint_ids)
    np.set_printoptions(precision=6, suppress=True)
    if args.parameters:
        for name, matrix in zip(config["arm_joint_names"], fixed_transforms):
            print(f"\n{name} 固定安装变换:\n{matrix}")
        print(f"\n工具偏移:\n{tool_transform}")
        return

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cases = [np.zeros(5)]
    cases.extend(np.eye(5) * 20)
    cases.extend([[20, -15, 30, -10, 45], [-30, 25, -20, 15, -40]])
    passed = True
    for degrees in cases:
        angles = np.deg2rad(degrees)
        data = mujoco.MjData(model)
        for name, angle in zip(config["arm_joint_names"], angles):
            data.qpos[model.jnt_qposadr[joint_ids[name]]] = angle
        gripper_id = joint_ids[config["gripper_joint_name"]]
        data.qpos[model.jnt_qposadr[gripper_id]] = config["gripper_position_rad"]
        mujoco.mj_forward(model, data)
        base_rotation = data.xmat[base_id].reshape(3, 3)
        reference = np.eye(4)
        reference[:3, :3] = base_rotation.T @ data.site_xmat[site_id].reshape(3, 3)
        reference[:3, 3] = base_rotation.T @ (data.site_xpos[site_id] - data.xpos[base_id])
        try:
            actual = np.asarray(forward_kinematics(angles, fixed_transforms, tool_transform))
        except NotImplementedError as error:
            print(f"尚未完成：{error}；请填写 src/so101/forward_kinematics.py")
            return
        if actual.shape != (4, 4) or not np.all(np.isfinite(actual)):
            raise ValueError("正运动学必须返回有限值组成的 4×4 矩阵")
        valid = np.allclose(actual, reference, atol=1e-8, rtol=0)
        passed = passed and valid
        print(f"\n角度 (deg): {np.asarray(degrees)}")
        print(f"位置误差 (m): {np.linalg.norm(actual[:3, 3] - reference[:3, 3]):.3e}")
        print(f"旋转矩阵最大元素误差: {np.max(np.abs(actual[:3, :3] - reference[:3, :3])):.3e}")
        print("通过" if valid else f"不一致\n你的结果:\n{actual}\nMuJoCo 参考:\n{reference}")
    if not passed:
        raise SystemExit("存在未通过的对照姿态")
    print("\n全部 8 组姿态对照通过；这只验证正运动学，不验证碰撞或动力学。")


if __name__ == "__main__":
    main()
