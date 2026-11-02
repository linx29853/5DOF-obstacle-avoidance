"""连接学习者已完成的规划、简化和轨迹模块，保存并播放 Demo。"""

import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile
import time

import mujoco
import numpy as np

from so101.collision import has_clearance, is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene
from so101.edge_check import is_edge_valid
from so101.model import PROJECT_ROOT
from so101.path_simplification import simplify_path
from so101.planning import rrt_connect
from so101.trajectory import quintic_trajectory

DEFAULT_CONFIG = PROJECT_ROOT / "configs/so101/demo.json"


def build_trajectory(path, velocity_limits, acceleration_limits, sample_period):
    """调用已有单段轨迹函数，拼接时间轴并移除重复的连接点。"""
    if not np.isfinite(sample_period) or sample_period <= 0:
        raise ValueError("轨迹采样周期必须为正有限数")
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 5 or len(path) == 0 or not np.all(np.isfinite(path)):
        raise ValueError("路径必须是非空的 (N, 5) 有限数组")
    quintic_trajectory(path[0], path[0], velocity_limits, acceleration_limits, sample_count=2)
    times = [np.zeros(1)]
    positions = [path[:1].copy()]
    velocities = [np.zeros((1, 5))]
    accelerations = [np.zeros((1, 5))]
    elapsed = 0.0
    for start, goal in zip(path[:-1], path[1:]):
        if np.array_equal(start, goal):
            continue
        segment_times, _, _, _ = quintic_trajectory(
            start, goal, velocity_limits, acceleration_limits, sample_count=2,
        )
        duration = float(segment_times[-1])
        intervals = max(
            1, int(np.ceil(duration / sample_period)),
            int(np.ceil(1.875 * np.max(np.abs(goal - start)) / np.deg2rad(0.5))),
        )
        segment_times, segment_positions, segment_velocities, segment_accelerations = quintic_trajectory(
            start, goal, velocity_limits, acceleration_limits, sample_count=intervals + 1,
        )
        times.append(elapsed + segment_times[1:])
        positions.append(segment_positions[1:])
        velocities.append(segment_velocities[1:])
        accelerations.append(segment_accelerations[1:])
        elapsed += duration
    return tuple(np.concatenate(values) for values in (times, positions, velocities, accelerations))


def play_trajectory(model, config, joint_ids, times, positions):
    """按轨迹自身的时间戳显示参考姿态。"""
    from mujoco import viewer as mujoco_viewer

    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[joint_ids[name]] for name in config["arm_joint_names"]]
    gripper_address = model.jnt_qposadr[joint_ids[config["gripper_joint_name"]]]
    data.qpos[addresses] = positions[0]
    data.qpos[gripper_address] = config["gripper_position_rad"]
    mujoco.mj_forward(model, data)
    with mujoco_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.18]
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.sync()
        began = time.perf_counter()
        for moment, angles in zip(times, positions):
            if not viewer.is_running():
                return
            time.sleep(max(0.0, began + float(moment) - time.perf_counter()))
            with viewer.lock():
                data.qpos[addresses] = angles
                data.qpos[gripper_address] = config["gripper_position_rad"]
                mujoco.mj_forward(model, data)
            viewer.sync()
        print("播放完成，关闭窗口退出。", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def run_demo(config_path=DEFAULT_CONFIG, case="multi", show_viewer=True):
    """运行一个完整 Demo，返回结果字典；每次运行保存到独立目录。"""
    config_path = Path(config_path).resolve()
    request = json.loads(config_path.read_text(encoding="utf-8"))
    if case not in request["cases"]:
        raise ValueError(f"配置中没有场景: {case}")
    start, goal = np.deg2rad(request["start_degrees"]), np.deg2rad(request["goal_degrees"])
    for angles in (start, goal):
        if angles.shape != (5,) or not np.all(np.isfinite(angles)):
            raise ValueError("起终点必须为五个有限角度值")
    margin = float(request["margin_mm"]) / 1000.0
    planning = request["planning"]
    step_size = np.deg2rad(planning["step_degrees"])
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("规划步长必须为正有限数")
    budget = planning["max_iterations"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("规划预算必须为非负整数")
    root = Path(request["output_dir"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{case}-", dir=root))
    scene = {key: request[key] for key in ("frame", "units", "table", "base_collision") if key in request}
    scene["obstacles"] = request["cases"][case]["obstacles"]
    scene_path = output / "scene.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "case": case, "output_dir": str(output), "status": "running", "margin_m": margin,
        "command": shlex.join([
            sys.executable, str(PROJECT_ROOT / "examples/so101/demo.py"),
            "--config", str(config_path), "--case", case,
        ] + ([] if show_viewer else ["--check-only"])),
    }

    def finish(status, message):
        report.update(status=status, message=message)
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(message, flush=True)
        print("结果目录:", output, flush=True)
        return report

    model, config, joint_ids = load_collision_scene(scene_path=scene_path, query_margin=margin)
    checks = 0

    def pose_valid(angles):
        nonlocal checks
        checks += 1
        _, contacts = inspect_contacts(model, config, joint_ids, angles)
        return is_pose_valid(angles, config["joint_limits_rad"], contacts) and has_clearance(contacts, margin)

    def edge_valid(first, second):
        return is_edge_valid(first, second, pose_valid)

    print(f"场景: {case}；安全裕量: {request['margin_mm']:g} mm", flush=True)
    for name, angles in (("起点", start), ("终点", goal)):
        if not pose_valid(angles):
            return finish("invalid_endpoint", f"{name}无效：请检查关节限位、障碍位置或安全间隙。")
    report["direct_connection_valid"] = bool(edge_valid(start, goal))
    checks = 0
    began = time.perf_counter()
    path = rrt_connect(
        start, goal, config["joint_limits_rad"], pose_valid, edge_valid,
        step_size=step_size, max_iterations=budget, seed=planning["seed"],
    )
    report.update(planning_seconds=time.perf_counter() - began, planning_pose_checks=checks)
    if path is None:
        return finish("budget_exhausted", "搜索预算已用完，本次未找到路径。")

    def verify_path(points):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 5 or len(points) == 0 or not np.all(np.isfinite(points)):
            raise ValueError("路径必须是非空的 (N, 5) 有限数组")
        if not np.allclose(points[0], start) or not np.allclose(points[-1], goal):
            raise ValueError("路径起终点不正确")
        if not all(pose_valid(angles) for angles in points):
            raise ValueError("路径中存在无效构型")
        for first, second in zip(points[:-1], points[1:]):
            if not is_edge_valid(first, second, pose_valid, max_joint_step=np.deg2rad(0.5)):
                raise ValueError("连接未通过 0.5° 复核")

    verify_path(path)
    original = np.asarray(path).copy()
    simplified = np.asarray(simplify_path(path, edge_valid))
    if not np.array_equal(np.asarray(path), original):
        raise ValueError("路径简化修改了输入")
    verify_path(simplified)
    length_before = float(np.linalg.norm(np.diff(original, axis=0), axis=1).sum())
    length_after = float(np.linalg.norm(np.diff(simplified, axis=0), axis=1).sum())
    if len(simplified) > len(original) or length_after > length_before + 1e-10:
        raise ValueError("简化增加了路点或路径长度")
    print(f"路径节点: {len(original)} → {len(simplified)}；开始生成时间轨迹", flush=True)
    limits = request["trajectory"]
    times, positions, velocities, accelerations = build_trajectory(
        simplified, limits["velocity_limits_rad_s"], limits["acceleration_limits_rad_s2"],
        limits["sample_period_s"],
    )
    if not np.all(np.isfinite(times)) or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("轨迹时间必须从零开始并严格递增")
    for values in (positions, velocities, accelerations):
        if values.shape != (len(times), 5) or not np.all(np.isfinite(values)):
            raise ValueError("轨迹数组形状或数值错误")
    if not np.allclose(velocities[[0, -1]], 0) or not np.allclose(accelerations[[0, -1]], 0):
        raise ValueError("轨迹起终速度和加速度应为零")
    peak_velocity = np.max(np.abs(velocities), axis=0)
    peak_acceleration = np.max(np.abs(accelerations), axis=0)
    if np.any(peak_velocity > np.asarray(limits["velocity_limits_rad_s"]) + 1e-8):
        raise ValueError("轨迹速度超限")
    if np.any(peak_acceleration > np.asarray(limits["acceleration_limits_rad_s2"]) + 1e-8):
        raise ValueError("轨迹加速度超限")
    verify_path(positions)
    np.save(output / "path.npy", original)
    np.save(output / "path_simplified.npy", simplified)
    np.savez(output / "trajectory.npz", times=times, positions=positions, velocities=velocities, accelerations=accelerations)
    report.update(
        original_nodes=len(original), simplified_nodes=len(simplified),
        original_length_rad=length_before, simplified_length_rad=length_after,
        trajectory_seconds=float(times[-1]), trajectory_frames=len(times),
        peak_velocity_rad_s=peak_velocity.tolist(), peak_acceleration_rad_s2=peak_acceleration.tolist(),
        total_pose_checks=checks, verification_step_degrees=0.5,
        versions={"python": sys.version.split()[0], "numpy": np.__version__, "mujoco": mujoco.__version__},
        source_sha256={
            str(source.relative_to(PROJECT_ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sorted((PROJECT_ROOT / "src/so101").glob("*.py"))
        },
        model_manifest=json.loads((PROJECT_ROOT / "models/so101/manifest.json").read_text(encoding="utf-8")),
    )
    finish("success", f"Demo 已生成并通过复核：{times[-1]:.3f} 秒，共 {len(times)} 帧。")
    if show_viewer:
        play_trajectory(model, config, joint_ids, times, positions)
    return report
