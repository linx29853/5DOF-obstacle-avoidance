"""固定实例、独立验证与评测统计口径。"""

import json

import numpy as np
import pytest

from so101.benchmark import CONFIG, measure, problem_for, tasks_for, verify
from so101.benchmark_report import wilson
from so101.planners import Budget, Options, Problem


def test_frozen_suite_has_verified_formal_and_payload_references():
    suite = json.loads(CONFIG.read_text())
    assert len(suite["scenes"]) == 5
    for scene in suite["scenes"]:
        queries = scene["queries"]
        assert len([q for q in queries if q["split"] == "formal"]) == 5
        assert len([q for q in queries if q["split"] == "multiquery"]) == 20
        assert len([q for q in queries if q["split"] == "tuning"]) == 3
        for query in queries:
            p = problem_for(scene["scene"], query)
            assert verify(p, query["reference_path_rad"])
            if query["split"] == "formal":
                assert verify(
                    problem_for(scene["scene"], query, suite["payload"]),
                    query["payload_reference_path_rad"],
                )


def test_measure_separates_planning_checks_from_verification():
    calls = []
    p = Problem(
        np.zeros(5), np.ones(5) * 0.1, [[-1, 1]] * 5, lambda q: calls.append(q) or True
    )
    r = measure(p, "rrt_connect", Budget(max_checks=1000), 4, Options())
    assert r["success"]
    assert len(calls) > r["stats"]["pose_checks"]
    assert r["verification_seconds"] >= 0 and r["postprocess_seconds"] >= 0
    assert r["simplified_length_rad"] <= r["length_rad"] + 1e-10


def test_report_confidence_intervals_include_extreme_cases():
    lo, hi = wilson(0, 30)
    assert abs(lo) < 1e-15 and 0 < hi < 0.2
    lo, hi = wilson(30, 30)
    assert 0.8 < lo < 1 and hi == pytest.approx(1)


def test_budget_matrix_and_ablation_are_frozen():
    suite = json.loads(CONFIG.read_text())
    tasks = tasks_for(suite, "full")
    assert len(tasks) == 15000
    for scene, query, planner, budget, seed, options, mode, group in tasks:
        assert query["split"] == "formal"
        assert "reference_path_rad" not in vars(problem_for(scene["scene"], query))
        if group == "single":
            assert options.nearest == "linear" and options.metric == "radians"
            assert options.goal_bias is None
        if group == "ablation-kdtree":
            assert options.nearest == "kdtree" and options.goal_bias is None
