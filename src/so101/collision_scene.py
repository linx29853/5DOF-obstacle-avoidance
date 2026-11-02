"""教学场景加载和接触信息读取，不包含姿态有效性算法。"""

import json
from pathlib import Path

import mujoco
import numpy as np

from so101.model import PROJECT_ROOT, load_model


def load_collision_scene(with_obstacle=True, scene_path=None, query_margin=0.0):
    if not np.isfinite(query_margin) or query_margin < 0:
        raise ValueError("query_margin 必须为非负有限数")
    model, config, joint_ids = load_model()
    if with_obstacle or scene_path is not None:
        scene_path = Path(scene_path or "configs/so101/collision_practice.json")
        if not scene_path.is_absolute():
            scene_path = PROJECT_ROOT / scene_path
        with scene_path.open(encoding="utf-8") as scene_file:
            scene = json.load(scene_file)
        if scene.get("frame") != "world" or scene.get("units") != "m":
            raise ValueError("场景必须使用 world 坐标系和 m 单位")
        spec = mujoco.MjSpec.from_file(str(PROJECT_ROOT / config["model_path"]))

        def add_geometry(body, geometry, color):
            center = np.asarray(geometry["center"], dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                raise ValueError("几何体中心必须为三个有限数")
            kind = geometry.get("type", "sphere")
            if kind == "sphere":
                radius = float(geometry["radius"])
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError("球半径必须为正有限数")
                size = [radius, 0, 0]
                geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
            elif kind == "box":
                size = np.asarray(geometry["half_size"], dtype=float)
                if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
                    raise ValueError("盒体半尺寸必须为三个正有限数")
                geom_type = mujoco.mjtGeom.mjGEOM_BOX
            else:
                raise ValueError(f"不支持的几何类型: {kind}")
            body.add_geom(
                name=geometry["name"], type=geom_type, pos=center, size=size,
                contype=1, conaffinity=1, rgba=color,
            )

        if scene.get("table"):
            add_geometry(spec.worldbody, scene["table"], [0.55, 0.55, 0.55, 1])
        if scene.get("base_collision"):
            base_geometry = {**scene["base_collision"], "type": "box"}
            add_geometry(spec.body("base"), base_geometry, [0.2, 0.4, 0.8, 0.25])
            spec.add_exclude(name="base_shoulder_mount", bodyname1="base", bodyname2="shoulder")
        if with_obstacle:
            obstacles = scene.get("obstacles", [scene["obstacle"]] if "obstacle" in scene else [])
            for obstacle in obstacles:
                add_geometry(spec.worldbody, obstacle, [0.9, 0.3, 0.1, 0.65])
        model = spec.compile()
        joint_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_ids
        }
    if query_margin > 0:
        enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_margin[enabled] = np.maximum(model.geom_margin[enabled], query_margin + 1e-6)
        model.geom_gap[enabled] = model.geom_margin[enabled]
    return model, config, joint_ids


def inspect_contacts(model, config, joint_ids, joint_angles):
    """设置静态姿态并返回 data 和 MuJoCo 检出的接触，不进行动力学步进。"""
    joint_angles = np.asarray(joint_angles, dtype=float)
    if joint_angles.shape != (5,) or not np.all(np.isfinite(joint_angles)):
        raise ValueError("joint_angles 必须是五个有限弧度值")
    data = mujoco.MjData(model)
    for name, angle in zip(config["arm_joint_names"], joint_angles):
        data.qpos[model.jnt_qposadr[joint_ids[name]]] = angle
    gripper_id = joint_ids[config["gripper_joint_name"]]
    data.qpos[model.jnt_qposadr[gripper_id]] = config["gripper_position_rad"]
    mujoco.mj_forward(model, data)
    contacts = []
    for contact in data.contact:
        names = []
        for geom_id in (contact.geom1, contact.geom2):
            body_id = model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            names.append(geom_name or f"{body_name}/geom_{geom_id}")
        contacts.append({"geom1": names[0], "geom2": names[1], "dist": float(contact.dist)})
    return data, contacts
