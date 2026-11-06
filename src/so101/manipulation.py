"""固定 SO-101 的分阶段运动学搬运任务与几何复核。"""

import hashlib
import json
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from so101.edge_check import is_edge_valid
from so101.grasp import add_finger_collisions, check_grasp, grasp_distances
from so101.model import PROJECT_ROOT, load_model
from so101.path_simplification import simplify_path
from so101.planning import rrt_connect

DEFAULT_CONFIG = PROJECT_ROOT / "configs/so101/manipulation.json"


def load_task(case, config_path=DEFAULT_CONFIG):
    request = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if case not in request["cases"]:
        raise ValueError(f"未知场景: {case}")
    scene = request["cases"][case]
    _, config, _ = load_model()
    spec = mujoco.MjSpec.from_file(str(PROJECT_ROOT / config["model_path"]))
    add_finger_collisions(spec)
    spec.worldbody.add_light(pos=[0.3, -0.3, 1], dir=[0, 0, -1])
    for geometry in request["environment"] + scene["obstacles"]:
        spec.worldbody.add_geom(
            name=geometry["name"],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=geometry["center"],
            size=geometry["half_size"],
            rgba=geometry["color"],
            contype=1,
            conaffinity=1,
        )
    for index, item in enumerate(scene["items"]):
        body = spec.worldbody.add_body(
            name=f"item_{index}", mocap=True, pos=item["source"]
        )
        body.add_geom(
            name=f"payload_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=item["half_size"],
            rgba=item["color"],
            contype=0,
            conaffinity=0,
        )
        # 目的地区域使用薄色块，所有实体障碍单独参与几何检查。
        spec.worldbody.add_geom(
            name=f"target_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[*item["destination"][:2], 0.0005],
            size=[0.026, 0.026, 0.0005],
            rgba=[*item["color"][:3], 0.35],
            contype=0,
            conaffinity=0,
        )
    model = spec.compile()
    data = mujoco.MjData(model)
    addresses = [
        model.jnt_qposadr[model.joint(n).id] for n in config["arm_joint_names"]
    ]
    gripper = model.jnt_qposadr[model.joint(config["gripper_joint_name"]).id]
    return request, scene, model, data, addresses, gripper


def solve_ik(model, data, addresses, target, seed):
    """约束工具位置与朝下的接近轴，保留绕该轴的自由度。"""
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("目标必须为三个有限坐标")
    limits = model.jnt_range[
        [
            model.joint(n).id
            for n in [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
        ]
    ]
    site = model.site("graspframe").id

    def residual(q):
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        axis = data.site_xmat[site].reshape(3, 3)[:, 0]
        return np.r_[
            data.site_xpos[site] - target,
            0.15 * (axis - [0, 0, -1]),
            0.15 * (data.site_xmat[site].reshape(3, 3)[:, 2] - [1, 0, 0]),
            0.00001 * (q - seed),
        ]

    seeds = [seed, np.deg2rad([0, -15, 25, 80, 0])]
    seeds.extend(np.random.default_rng(42).uniform(limits[:, 0], limits[:, 1], (12, 5)))
    for initial in seeds:
        result = least_squares(
            residual,
            np.clip(initial, limits[:, 0] + 1e-8, limits[:, 1] - 1e-8),
            bounds=limits.T,
            max_nfev=180,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)
        if np.linalg.norm(error[:3]) < 0.0005 and np.linalg.norm(error[3:9]) < 0.001:
            return result.x.copy()
    raise ValueError(f"目标超出朝下抓取工作空间: {target.tolist()}")


def set_frame(
    model, data, addresses, gripper, q, opening, object_positions, object_quaternions
):
    data.qpos[addresses] = q
    data.qpos[gripper] = opening
    for i, (position, quaternion) in enumerate(
        zip(object_positions, object_quaternions)
    ):
        mid = model.body_mocapid[model.body(f"item_{i}").id]
        data.mocap_pos[mid] = position
        data.mocap_quat[mid] = quaternion
    mujoco.mj_forward(model, data)


def run_task(case="tabletop", config_path=DEFAULT_CONFIG, output_dir=None):
    root = PROJECT_ROOT / "outputs/so101/manipulation"
    root.mkdir(parents=True, exist_ok=True)
    output = (
        Path(output_dir)
        if output_dir
        else Path(tempfile.mkdtemp(prefix=f"{case}-", dir=root))
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _run_task(case, config_path, output)
    except (ValueError, KeyError) as exc:
        (output / "result.json").write_text(
            json.dumps(
                {"case": case, "status": "failed", "message": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def _run_task(case, config_path, output_dir):
    request, scene, model, data, addresses, gripper = load_task(case, config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps({"case": case, "status": "running"}), encoding="utf-8"
    )
    (output / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    q = solve_ik(model, data, addresses, scene["home"], np.zeros(5))
    objects = np.array([item["source"] for item in scene["items"]], dtype=float)
    rotations = np.tile([1.0, 0, 0, 0], (len(objects), 1))
    opening = request["gripper_open_rad"]
    site = model.site("graspframe").id
    times, qs, jaws, xyz, quats, stages, references, actuals = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    elapsed = 0.0
    attached = None
    relative_position = relative_rotation = None
    environment = [
        model.geom(g["name"]).id for g in request["environment"] + scene["obstacles"]
    ]
    arm_geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] > 0 and model.geom_contype[i]
    ]
    allowed_grasp = {"gripper", "moving_jaw_so101_v1"}
    checks = 0
    planned_detours = []
    transfer_routes = []
    grasp_records = []

    def pose(qtest, jaw):
        nonlocal checks
        checks += 1
        if np.any(qtest < model.jnt_range[:5, 0]) or np.any(
            qtest > model.jnt_range[:5, 1]
        ):
            return False
        set_frame(model, data, addresses, gripper, qtest, jaw, objects, rotations)
        if attached is not None:
            R = data.site_xmat[site].reshape(3, 3)
            mid = model.body_mocapid[model.body(f"item_{attached}").id]
            data.mocap_pos[mid] = data.site_xpos[site] + R @ relative_position
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, (R @ relative_rotation).ravel())
            data.mocap_quat[mid] = quaternion
            mujoco.mj_forward(model, data)
        # MuJoCo 接触检查覆盖机械臂对场景及自碰撞。
        if any(c.dist < -1e-6 for c in data.contact):
            return False
        for i in range(len(objects)):
            gid = model.geom(f"payload_{i}").id
            for other in arm_geoms:
                bodyname = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[other]
                )
                if bodyname in allowed_grasp:
                    continue
                if mujoco.mj_geomDistance(model, data, gid, other, 0.002, None) < 0:
                    return False
            if i == attached:
                others = environment + [
                    model.geom(f"payload_{j}").id for j in range(len(objects)) if j != i
                ]
                for other in others:
                    if (
                        mujoco.mj_geomDistance(model, data, gid, other, 0.002, None)
                        < -1e-5
                    ):
                        return False
        return True

    def append(stage, qframe, jaw, reference, dt=None):
        nonlocal elapsed
        if not pose(qframe, jaw):
            bad = [
                (int(c.geom1), int(c.geom2), float(c.dist))
                for c in data.contact
                if c.dist < 0
            ]
            raise ValueError(f"{stage}: 碰撞复核失败 {bad}")
        if times:
            elapsed += request["sample_period_s"] if dt is None else dt
        times.append(elapsed)
        qs.append(qframe.copy())
        jaws.append(jaw)
        xyz.append(data.mocap_pos.copy())
        quats.append(data.mocap_quat.copy())
        stages.append(stage)
        references.append(reference.copy())
        actuals.append(data.site_xpos[site].copy())

    def move(stage, target):
        nonlocal q
        target = np.asarray(target, float)
        start_pos = data.site_xpos[site].copy()
        points = []
        # 任务空间直线离散为毫米级路点，逐点求解连续逆运动学。
        previous = q.copy()
        for fraction in np.linspace(
            0, 1, max(2, int(np.ceil(np.linalg.norm(target - start_pos) / 0.006)) + 1)
        )[1:]:
            previous = solve_ik(
                model,
                data,
                addresses,
                start_pos + (target - start_pos) * fraction,
                previous,
            )
            points.append(previous)
        path = np.vstack([q, points])
        if not pose(path[-1], opening):
            raise ValueError(f"{stage}: 目标被障碍阻挡")
        # 长距离转移可调用原有 RRT-Connect 绕障；接近、抬升、放置保持直线。
        edge = lambda a, b: is_edge_valid(
            a, b, lambda x: pose(x, opening), max_joint_step=np.deg2rad(0.5)
        )
        if not all(edge(a, b) for a, b in pairwise(path)):
            if not (
                stage.split("/")[-1] in ("transfer", "return_home", "approach")
                or "transfer_via" in stage
            ):
                raise ValueError(f"{stage}: 直线动作被障碍阻挡")
            found = rrt_connect(
                q,
                path[-1],
                model.jnt_range[:5],
                lambda x: pose(x, opening),
                edge,
                step_size=0.15,
                max_iterations=2500,
                seed=42,
            )
            if found is None:
                raise ValueError(f"{stage}: 转移规划搜索预算用完")
            path = np.asarray(simplify_path(found, edge))
            planned_detours.append(
                {
                    "stage": stage,
                    "direct_blocked": True,
                    "rrt_nodes": len(found),
                    "simplified_nodes": len(path),
                }
            )
        # 全段使用一次五次时间缩放，经过中间路点时保持运动。
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        coordinate = np.r_[0.0, np.cumsum(lengths)]
        if coordinate[-1] < 1e-10:
            q = path[-1].copy()
            pose(q, opening)
            return
        coordinate /= coordinate[-1]
        duration = max(
            1.0,
            float(
                np.max(
                    np.sum(np.abs(np.diff(path, axis=0)), axis=0)
                    / request["velocity_limits_rad_s"]
                )
            )
            * 1.875,
        )
        sample_count = max(3, int(np.ceil(duration / request["sample_period_s"])) + 1)
        u = np.linspace(0, 1, sample_count)
        progress = 10 * u**3 - 15 * u**4 + 6 * u**5
        sampled = PchipInterpolator(coordinate, path, axis=0)(progress)
        dt = duration / (sample_count - 1)
        velocity = np.diff(sampled, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        scale = max(
            1.0,
            float(np.max(np.abs(velocity) / request["velocity_limits_rad_s"])),
            float(
                np.sqrt(
                    np.max(np.abs(acceleration) / request["acceleration_limits_rad_s2"])
                )
            ),
        )
        dt *= scale * 1.01
        for fraction, a, b in zip(progress[1:], sampled[:-1], sampled[1:]):
            if not edge(a, b):
                raise ValueError(f"{stage}: 时间轨迹连接碰撞")
            data.qpos[addresses] = b
            mujoco.mj_forward(model, data)
            append(stage, b, opening, start_pos + (target - start_pos) * fraction, dt)
        q = path[-1].copy()
        pose(q, opening)

    def jaw_motion(stage, goal):
        nonlocal opening
        for value in np.linspace(opening, goal, 21)[1:]:
            append(stage, q, float(value), data.site_xpos[site].copy())
        opening = goal

    pose(q, opening)
    append("home", q, opening, data.site_xpos[site].copy())
    for i, item in enumerate(scene["items"]):
        prefix = f"item{i + 1}/"
        source, destination = np.array(item["source"]), np.array(item["destination"])
        grasp = source
        place = destination
        height = scene["transport_height_m"]
        move(prefix + "approach", [*grasp[:2], height])
        move(prefix + "descend", grasp)
        # 以两侧实际距离求闭合角；只有双侧夹持成立才绑定物体。
        from scipy.optimize import minimize_scalar

        payload = model.geom(f"payload_{i}").id

        def fit_jaw(value, payload=payload):
            pose(q, value)
            gap = grasp_distances(model, data, payload)
            return float(np.sum(gap**2))

        fit = minimize_scalar(
            fit_jaw,
            bounds=(-0.15, request["gripper_open_rad"]),
            method="bounded",
            options={"xatol": 1e-8},
        )
        jaw_motion(prefix + "close", float(fit.x))
        distances = check_grasp(model, data, payload)
        grasp_records.append(
            {
                "item": i,
                "closed_angle_rad": opening,
                "finger_gaps_m": distances.tolist(),
            }
        )

        R = data.site_xmat[site].reshape(3, 3).copy()
        attached = i
        relative_position = R.T @ (objects[i] - data.site_xpos[site])
        relative_rotation = R.T.copy()
        move(prefix + "lift", [*grasp[:2], height])
        via_points = scene.get("transfer_waypoints_m", [])
        if via_points:
            destination_q = solve_ik(model, data, addresses, [*place[:2], height], q)
            direct = is_edge_valid(
                q,
                destination_q,
                lambda x: pose(x, opening),
                max_joint_step=np.deg2rad(0.5),
            )
            transfer_routes.append(
                {
                    "item": i,
                    "direct_joint_connection_valid": bool(direct),
                    "via_points_m": via_points,
                }
            )
            pose(q, opening)
        for k, via in enumerate(via_points):
            move(prefix + f"transfer_via{k + 1}", via)
        move(prefix + "transfer", [*place[:2], height])
        move(prefix + "lower", place)
        # 记录释放时位姿，释放后保持在落点；不将物体瞬移到目标。
        mid = model.body_mocapid[model.body(f"item_{i}").id]
        objects[i] = data.mocap_pos[mid].copy()
        rotations[i] = data.mocap_quat[mid].copy()
        attached = None
        jaw_motion(prefix + "open", request["gripper_open_rad"])
        move(prefix + "retreat", [*place[:2], height])
    move("return_home", scene["home"])
    np.savez_compressed(
        output / "trajectory.npz",
        times=times,
        positions=qs,
        gripper=jaws,
        object_positions=xyz,
        object_quaternions=quats,
        stages=stages,
        reference_xyz=references,
        tool_xyz=actuals,
    )
    errors = np.linalg.norm(
        objects - np.array([x["destination"] for x in scene["items"]]), axis=1
    )
    speed = np.diff(qs, axis=0) / np.diff(times)[:, None]
    accel = (
        np.diff(speed, axis=0)
        / ((np.diff(times)[:-1] + np.diff(times)[1:]) / 2)[:, None]
    )
    report = {
        "case": case,
        "title": scene["title"],
        "status": "success",
        "simulation": "kinematic_attachment",
        "frames": len(times),
        "duration_s": times[-1],
        "pose_checks": checks,
        "placement_errors_m": errors.tolist(),
        "all_placed": bool(np.all(errors < 0.001)),
        "grasps": grasp_records,
        "planned_detours": planned_detours,
        "transfer_routes": transfer_routes,
        "peak_joint_speed_rad_s": np.max(np.abs(speed), axis=0).tolist(),
        "peak_joint_acceleration_rad_s2": np.max(np.abs(accel), axis=0).tolist(),
        "max_deviation_from_straight_line_m": float(
            np.max(np.linalg.norm(np.array(references) - actuals, axis=1))
        ),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("manipulation_media.py"),
                Path(__file__).with_name("grasp.py"),
            ]
        },
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
    }
    (output / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return output


def play_task(run_dir):
    from mujoco import viewer

    root = Path(run_dir)
    report = json.loads((root / "result.json").read_text())
    _, scene, model, data, addresses, gripper = load_task(
        report["case"], root / "request.json"
    )
    with (
        np.load(root / "trajectory.npz", allow_pickle=False) as track,
        viewer.launch_passive(model, data) as window,
    ):
        window.cam.lookat[:] = [0.15, 0, 0.12]
        window.cam.distance = 0.75
        window.cam.azimuth = 135
        window.cam.elevation = -25
        for key, value in scene.get("camera", {}).items():
            if key == "lookat":
                window.cam.lookat[:] = value
            else:
                setattr(window.cam, key, value)
        began = time.perf_counter()
        for t, q, jaw, positions, rotations in zip(
            track["times"],
            track["positions"],
            track["gripper"],
            track["object_positions"],
            track["object_quaternions"],
        ):
            if not window.is_running():
                return
            time.sleep(max(0, began + float(t) - time.perf_counter()))
            with window.lock():
                set_frame(model, data, addresses, gripper, q, jaw, positions, rotations)
            window.sync()
        while window.is_running():
            window.sync()
            time.sleep(0.02)
