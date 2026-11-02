"""Demo 时间轴拼接与失败结果保存。"""

import json
from pathlib import Path

import numpy as np

from so101.demo import DEFAULT_CONFIG, build_trajectory, run_demo


def test_stitching_keeps_waypoint_and_rest_state():
    path = np.array([[0, 0, 0, 0, 0], [0.2, -0.1, 0, 0, 0], [0.2, -0.1, 0.3, 0, 0]])
    times, positions, velocities, accelerations = build_trajectory(path, np.ones(5), np.ones(5), 0.02)
    assert times[0] == 0
    assert np.all(np.diff(times) > 0)
    assert np.all(np.diff(times) <= 0.02 + 1e-12)
    assert np.allclose(positions[0], path[0])
    assert np.allclose(positions[-1], path[-1])
    junctions = np.flatnonzero(np.all(np.isclose(positions, path[1], atol=1e-12, rtol=0), axis=1))
    assert len(junctions) == 1
    assert np.allclose(velocities[junctions], 0)
    assert np.allclose(accelerations[junctions], 0)


def test_repeated_and_single_waypoints_have_no_duplicate_times():
    start = np.zeros(5)
    goal = np.array([0.1, 0, 0, 0, 0])
    times, positions, _, _ = build_trajectory([start, start, goal, goal], np.ones(5), np.ones(5), 0.02)
    assert np.all(np.diff(times) > 0)
    assert np.allclose(positions[-1], goal)
    times, positions, velocities, accelerations = build_trajectory([goal], np.ones(5), np.ones(5), 0.02)
    assert np.array_equal(times, [0])
    assert np.array_equal(positions, [goal])
    assert not np.any(velocities)
    assert not np.any(accelerations)


def test_failed_runs_save_reason_without_path_files(tmp_path):
    request = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    request["output_dir"] = str(tmp_path / "results")
    request["planning"]["max_iterations"] = 0
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(request), encoding="utf-8")
    for case, expected in (("blocked", "invalid_endpoint"), ("multi", "budget_exhausted")):
        report = run_demo(config_path, case, show_viewer=False)
        assert report["status"] == expected
        output = Path(report["output_dir"])
        saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert saved["status"] == expected
        assert saved["message"]
        assert (output / "scene.json").is_file()
        assert (output / "request.json").is_file()
        assert not (output / "path.npy").exists()
        assert not (output / "trajectory.npz").exists()
