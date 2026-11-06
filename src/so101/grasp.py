"""SO-101 夹指的分段碰撞形状和双侧夹持校验。"""

from itertools import pairwise

import mujoco
import numpy as np

FINGER_BODIES = ("gripper", "moving_jaw_so101_v1")


def add_finger_collisions(spec):
    """按中性姿态的夹指纵向切片，保留凹形夹持空隙。"""
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    reference = model.body("gripper").id
    for name, meshname in [
        ("gripper", "wrist_roll_follower_so101_v1"),
        ("moving_jaw_so101_v1", "moving_jaw_so101_v1"),
    ]:
        body = spec.body(name)
        for geom in list(body.geoms):
            if geom.meshname != meshname or not geom.contype:
                continue
            geom.contype = 0
            geom.conaffinity = 0
        gid = next(
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body(name).id
            and model.geom_contype[i]
            and model.mesh(model.geom_dataid[i]).name == meshname
        )
        mesh = model.geom_dataid[gid]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh]
            + model.mesh_vertnum[mesh]
        ]
        faces = model.mesh_face[
            model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh]
            + model.mesh_facenum[mesh]
        ]
        vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        vertices = (vertices - data.xpos[reference]) @ data.xmat[reference].reshape(
            3, 3
        )
        edges = np.unique(
            np.sort(
                np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                axis=1,
            ),
            axis=0,
        )
        starts, ends = vertices[edges[:, 0]], vertices[edges[:, 1]]
        bounds = np.linspace(
            vertices[:, 2].min() - 1e-7, vertices[:, 2].max() + 1e-7, 22
        )
        bid = model.body(name).id
        for index, (low, high) in enumerate(pairwise(bounds)):
            points = [vertices[(vertices[:, 2] >= low) & (vertices[:, 2] <= high)]]
            for plane in [low, high]:
                mask = (starts[:, 2] - plane) * (ends[:, 2] - plane) < 0
                p, q = starts[mask], ends[mask]
                fraction = (plane - p[:, 2]) / (q[:, 2] - p[:, 2])
                points.append(p + (q - p) * fraction[:, None])
            points = np.concatenate(points)
            if len(points) < 4:
                continue
            world = points @ data.xmat[reference].reshape(3, 3).T + data.xpos[reference]
            local = (world - data.xpos[bid]) @ data.xmat[bid].reshape(3, 3)
            mesh_name = f"{name}_slice_{index}"
            spec.add_mesh(name=mesh_name, uservert=local.ravel().tolist())
            body.add_geom(
                name=mesh_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                group=3,
                rgba=[0, 0, 0, 0],
                contype=1,
                conaffinity=1,
            )
    # 工具 x 轴朝向指尖；此参考点位于两指有效夹持段之间。
    spec.body("gripper").add_site(
        name="graspframe",
        pos=[0.0041, 0, -0.087],
        quat=[0.7071067811865476, 0, 0.7071067811865476, 0],
        group=3,
    )


def grasp_distances(model, data, payload):
    distances = []
    for name in FINGER_BODIES:
        geoms = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body(name).id
            and model.geom_contype[i]
            and (model.geom(i).name or "").startswith(name + "_slice_")
        ]
        distances.append(
            min(
                mujoco.mj_geomDistance(model, data, payload, g, 0.2, None)
                for g in geoms
            )
        )
    return np.array(distances)


def check_grasp(model, data, payload, tolerance=0.001):
    distances = grasp_distances(model, data, payload)
    if np.any(np.abs(distances) > tolerance):
        raise ValueError(f"双侧夹持未成立，夹指间隙 mm: {(distances * 1000).tolist()}")
    # 两个最近夹持面必须分列物块中心两侧。
    axis = data.xmat[model.body("gripper").id].reshape(3, 3)[:, 0]
    sides = []
    for name in FINGER_BODIES:
        best = (float("inf"), None)
        for i in range(model.ngeom):
            if not (model.geom(i).name or "").startswith(name + "_slice_"):
                continue
            segment = np.zeros(6)
            gap = mujoco.mj_geomDistance(model, data, payload, i, 0.2, segment)
            if gap < best[0]:
                best = (gap, segment[3:].copy())
        sides.append(float((best[1] - data.geom_xpos[payload]) @ axis))
    if not sides[0] < 0 < sides[1]:
        raise ValueError("夹持面未分列物块两侧")
    return distances
