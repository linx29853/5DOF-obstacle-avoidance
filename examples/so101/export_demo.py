"""将已保存的 MuJoCo Demo 时间轨迹导出为 GIF 与封面。"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from so101.collision_scene import load_collision_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="包含 scene.json 和 trajectory.npz 的 Demo 结果目录")
    parser.add_argument("--output", type=Path, default=Path("assets/demo/so101-obstacle-avoidance.gif"))
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30:
        parser.error("帧率应位于 1 到 30 之间")
    report = json.loads((args.run_dir / "result.json").read_text(encoding="utf-8"))
    if report["status"] != "success":
        parser.error("请选择成功生成轨迹的 Demo 目录")
    with np.load(args.run_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        times = trajectory["times"]
        positions = trajectory["positions"]
    model, config, joint_ids = load_collision_scene(scene_path=args.run_dir / "scene.json")
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 480
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[joint_ids[name]] for name in config["arm_joint_names"]]
    gripper_address = model.jnt_qposadr[joint_ids[config["gripper_joint_name"]]]
    data.qpos[gripper_address] = config["gripper_position_rad"]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.15, 0.0, 0.16]
    camera.distance = 0.85
    camera.azimuth = 135
    camera.elevation = -24
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17)
    small_font = ImageFont.truetype(font_path, 14)
    duration = float(times[-1])
    moments = np.linspace(0.0, duration, max(2, int(np.ceil(duration * args.fps)) + 1))
    sampled = np.column_stack([np.interp(moments, times, positions[:, joint]) for joint in range(5)])
    trace = []
    for angles in sampled:
        data.qpos[addresses] = angles
        mujoco.mj_forward(model, data)
        trace.append(data.site("gripperframe").xpos.copy())
    trace = np.asarray(trace)
    frames = []
    with mujoco.Renderer(model, height=480, width=720) as renderer:
        for index, (moment, angles) in enumerate(zip(moments, sampled)):
            data.qpos[addresses] = angles
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            points = [(point, 0.0018, [1.0, 0.75, 0.1, 0.8]) for point in trace[:index + 1:2]]
            points += [(trace[0], 0.006, [0.15, 0.8, 0.45, 1]), (trace[-1], 0.006, [0.15, 0.55, 1.0, 1])]
            for point, radius, color in points:
                geom_id = renderer.scene.ngeom
                if geom_id >= renderer.scene.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[geom_id], mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([radius, radius, radius]), point, np.eye(3).ravel(), np.asarray(color, dtype=np.float32),
                )
                renderer.scene.ngeom += 1
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 720, 49), fill=(19, 27, 38))
            draw.text((18, 14), "SO-101 | 5-DOF obstacle avoidance", font=font, fill=(240, 245, 250))
            draw.rectangle((0, 438, 720, 480), fill=(19, 27, 38))
            draw.text((18, 451), f"RRT-Connect  |  {report['original_nodes']} to {report['simplified_nodes']} waypoints  |  {report['margin_m'] * 1000:g} mm clearance", font=small_font, fill=(230, 236, 244))
            draw.text((590, 451), f"{moment:04.1f} / {duration:.1f} s", font=small_font, fill=(230, 236, 244))
            frames.append(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_times = np.round(moments * 100).astype(int) * 10
    delays = np.maximum(10, np.diff(frame_times)).tolist() + [1000]
    delays[0] += 500
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=delays, loop=0, optimize=True)
    cover_path = args.output.with_suffix(".png")
    frames[len(frames) // 2].save(cover_path)
    print(f"GIF: {args.output}；{len(frames)} 帧；{args.output.stat().st_size / 1024 / 1024:.2f} MiB")
    print("封面:", cover_path)


if __name__ == "__main__":
    main()
