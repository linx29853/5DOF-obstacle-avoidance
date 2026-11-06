"""真实模型上的任务阶段、碰撞和物体释放回归测试。"""

import json

import mujoco
import numpy as np
import pytest

from so101.manipulation import load_task, run_task, set_frame, solve_ik


@pytest.mark.parametrize("case", ["tabletop", "barrier", "sorting"])
def test_manipulation_end_to_end(case, tmp_path):
    root = run_task(case, output_dir=tmp_path / case)
    report = json.loads((root / "result.json").read_text())
    request, scene, model, data, addresses, gripper = load_task(
        case, root / "request.json"
    )
    assert report["all_placed"]
    assert np.max(report["peak_joint_speed_rad_s"]) > 0
    assert np.all(
        np.array(report["peak_joint_speed_rad_s"])
        <= np.array(request["velocity_limits_rad_s"]) + 1e-7
    )
    assert np.all(
        np.array(report["peak_joint_acceleration_rad_s2"])
        <= np.array(request["acceleration_limits_rad_s2"]) * 1.02
    )
    with np.load(root / "trajectory.npz", allow_pickle=False) as tr:
        assert np.all(np.diff(tr["times"]) > 0)
        assert tr["positions"].shape == (len(tr["times"]), 5)
        for i, item in enumerate(scene["items"]):
            stages = tr["stages"].tolist()
            expected = [
                f"item{i + 1}/{s}"
                for s in [
                    "approach",
                    "descend",
                    "close",
                    "lift",
                    "transfer",
                    "lower",
                    "open",
                    "retreat",
                ]
            ]
            assert [stages.index(s) for s in expected] == sorted(
                stages.index(s) for s in expected
            )
            close = stages.index(f"item{i + 1}/close")
            released = stages.index(f"item{i + 1}/open")
            np.testing.assert_allclose(
                tr["object_positions"][:close, i],
                np.tile(item["source"], (close, 1)),
                atol=1e-7,
            )
            np.testing.assert_allclose(
                tr["object_positions"][released:, i],
                np.tile(item["destination"], (len(stages) - released, 1)),
                atol=0.001,
            )
            assert np.max(tr["object_positions"][:, i, 2]) > item["source"][2] + 0.03
        # 独立回放所有样本，确认机械臂对环境及自碰撞。
        for q, jaw, objects, quats in zip(
            tr["positions"],
            tr["gripper"],
            tr["object_positions"],
            tr["object_quaternions"],
        ):
            set_frame(model, data, addresses, gripper, q, jaw, objects, quats)
            assert all(c.dist >= -1e-6 for c in data.contact)
            for i in range(len(scene["items"])):
                for geom in request["environment"] + scene["obstacles"]:
                    distance = mujoco.mj_geomDistance(
                        model,
                        data,
                        model.geom(f"payload_{i}").id,
                        model.geom(geom["name"]).id,
                        0.01,
                        None,
                    )
                    assert distance >= -1e-5


def test_unreachable_target_rejected():
    _, _, model, data, addresses, _ = load_task("tabletop")
    with pytest.raises(ValueError, match="工作空间"):
        solve_ik(model, data, addresses, [2, 2, 2], np.zeros(5))


def test_blocked_task_records_failure(tmp_path):
    from so101.manipulation import DEFAULT_CONFIG

    request = json.loads(DEFAULT_CONFIG.read_text())
    request["cases"]["tabletop"]["obstacles"].append(
        {
            "name": "blocked_pick",
            "center": [0.20, -0.10, 0.04],
            "half_size": [0.04, 0.04, 0.04],
            "color": [1, 0, 0, 1],
        }
    )
    config = tmp_path / "blocked.json"
    config.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="阻挡"):
        run_task("tabletop", config, tmp_path / "run")
    assert json.loads((tmp_path / "run/result.json").read_text())["status"] == "failed"
