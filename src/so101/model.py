"""SO-101 模型配置与加载。"""

import json
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/so101/simulation.json"


def load_model(config_path=DEFAULT_CONFIG):
    """加载配置和模型，并按名称核对五个机械臂关节及夹爪。"""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("配置中未指定 model_path")
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 SO-101 模型: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    names = config["arm_joint_names"] + [config["gripper_joint_name"]]
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("配置必须包含五个不同的机械臂关节和一个夹爪关节")
    joint_ids = {}
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"模型中找不到关节: {name}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"关节不是旋转关节: {name}")
        joint_ids[name] = joint_id
    return model, config, joint_ids


def kinematic_parameters(model, config, joint_ids):
    """读取五段固定安装变换与 gripperframe 工具偏移，不计算关节运动。"""
    def transform(position, quaternion):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.reshape(3, 3)
        matrix[:3, 3] = position
        return matrix

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id < 0:
        raise ValueError("模型缺少 base 坐标系")
    previous_body = base_id
    fixed_transforms = []
    for name in config["arm_joint_names"]:
        joint_id = joint_ids[name]
        body_id = model.jnt_bodyid[joint_id]
        if model.body_parentid[body_id] != previous_body:
            raise ValueError(f"关节链结构与练习假设不符: {name}")
        if not np.allclose(model.jnt_axis[joint_id], [0, 0, 1]):
            raise ValueError(f"关节轴不是局部 +Z: {name}")
        if not np.allclose(model.jnt_pos[joint_id], 0):
            raise ValueError(f"关节原点存在额外偏移: {name}")
        if not np.isclose(model.qpos0[model.jnt_qposadr[joint_id]], 0):
            raise ValueError(f"关节存在非零参考角: {name}")
        fixed_transforms.append(transform(model.body_pos[body_id], model.body_quat[body_id]))
        previous_body = body_id
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if site_id < 0 or model.site_bodyid[site_id] != previous_body:
        raise ValueError("工具坐标系 gripperframe 不在第五关节连杆上")
    tool_transform = transform(model.site_pos[site_id], model.site_quat[site_id])
    return np.asarray(fixed_transforms), tool_transform
