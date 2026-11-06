"""保存任务轨迹的 MuJoCo 动画、封面与轨迹图。"""

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from so101.manipulation import load_task, set_frame


def export_task(run_dir, output_path=None, fps=12):
    if not isinstance(fps, int) or not 1 <= fps <= 30:
        raise ValueError("fps 必须为 1～30 的整数")
    root = Path(run_dir)
    report = json.loads((root / "result.json").read_text())
    if report["status"] != "success":
        raise ValueError("任务必须成功后再导出")
    _, scene, model, data, addresses, gripper = load_task(
        report["case"], root / "request.json"
    )
    target = Path(output_path or root / "demo.gif")
    target.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.15, 0, 0.12]
    camera.distance = 0.72
    camera.azimuth = 135
    camera.elevation = -30
    for key, value in scene.get("camera", {}).items():
        if key == "lookat":
            camera.lookat[:] = value
        else:
            setattr(camera, key, value)
    frames = []
    with (
        np.load(root / "trajectory.npz", allow_pickle=False) as track,
        mujoco.Renderer(model, height=480, width=640) as renderer,
    ):
        moments = np.linspace(
            0, track["times"][-1], int(np.ceil(track["times"][-1] * fps)) + 1
        )
        for moment in moments:
            # 在保存采样点上播放，保持物体与夹爪的同帧变换。
            i = min(np.searchsorted(track["times"], moment), len(track["times"]) - 1)
            set_frame(
                model,
                data,
                addresses,
                gripper,
                track["positions"][i],
                float(track["gripper"][i]),
                track["object_positions"][i],
                track["object_quaternions"][i],
            )
            renderer.update_scene(data, camera=camera)
            for part in range(track["object_positions"].shape[1]):
                indices = np.flatnonzero(
                    np.char.startswith(track["stages"], f"item{part + 1}/transfer")
                )[::3]
                for point in track["object_positions"][indices, part]:
                    geom_id = renderer.scene.ngeom
                    if geom_id >= renderer.scene.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        renderer.scene.geoms[geom_id],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.0015] * 3),
                        point,
                        np.eye(3).ravel(),
                        np.array([1.0, 0.8, 0.15, 1.0], dtype=np.float32),
                    )
                    renderer.scene.ngeom += 1
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 640, 42), fill=(20, 28, 40))
            draw.text((14, 8), f"SO-101 | {report['title']}", fill="white")
            draw.text(
                (14, 25),
                "Kinematic grasp | yellow: planned loaded path",
                fill=(180, 205, 225),
            )
            draw.rectangle((0, 447, 640, 480), fill=(20, 28, 40))
            draw.text(
                (14, 458),
                f"{track['stages'][i]!s:24}  {moment:4.1f} / {moments[-1]:.1f} s",
                fill="white",
            )
            frames.append(image)
        delays = np.maximum(
            10, np.diff(np.round(moments * 100).astype(int)) * 10
        ).tolist() + [800]
        frames[0].save(
            target, save_all=True, append_images=frames[1:], duration=delays, loop=0
        )
        frames[len(frames) // 2].save(target.with_suffix(".png"))
        lifting = np.flatnonzero(track["stages"] == "item1/lift")
        i = int(lifting[len(lifting) // 2])
        set_frame(
            model,
            data,
            addresses,
            gripper,
            track["positions"][i],
            float(track["gripper"][i]),
            track["object_positions"][i],
            track["object_quaternions"][i],
        )
        camera.lookat[:] = track["object_positions"][i, 0]
        camera.distance = 0.23
        camera.azimuth = 90
        camera.elevation = -12
        renderer.update_scene(data, camera=camera)
        closeup = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(closeup)
        draw.rectangle((0, 0, 640, 30), fill=(20, 28, 40))
        draw.text((14, 10), "Grasp close-up | two-sided contact verified", fill="white")
        closeup.save(target.with_name(target.stem + "-grasp.png"))

    print(f"动画: {target}", flush=True)
    return target


def plot_task(run_dir):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(run_dir)
    with np.load(root / "trajectory.npz", allow_pickle=False) as tr:
        fig = plt.figure(figsize=(11, 4))
        ax = fig.add_subplot(121, projection="3d")
        ax.plot(*tr["tool_xyz"].T, label="Tool path")
        for i in range(tr["object_positions"].shape[1]):
            ax.plot(*tr["object_positions"][:, i].T, label=f"Part {i + 1}")
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.legend()
        ax = fig.add_subplot(122)
        ax.plot(
            tr["times"],
            np.rad2deg(tr["positions"]),
            label=["pan", "shoulder", "elbow", "wrist", "roll"],
        )
        ax.set(xlabel="Time (s)", ylabel="Joint angle (deg)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / "trajectory.png", dpi=150)
        plt.close(fig)
