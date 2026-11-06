"""评测内存保护与模型缓存边界。"""

import pytest

import so101.benchmark_resources as resources
from so101 import benchmark


def test_guard_stops_before_work_on_low_available_memory(monkeypatch):
    monkeypatch.setattr(
        resources, "memory_snapshot", lambda: {"MemAvailable": 1024, "VmRSS": 1024}
    )
    with pytest.raises(resources.ResourcePressure, match="可用内存"):
        resources.MemoryGuard().check(force=True)


def test_guard_stops_when_process_exceeds_rss(monkeypatch):
    monkeypatch.setattr(
        resources,
        "memory_snapshot",
        lambda: {"MemAvailable": 8 * 1024**3, "VmRSS": 2 * 1024**3},
    )
    with pytest.raises(resources.ResourcePressure, match="进程内存"):
        resources.MemoryGuard().check(force=True)


def test_oracle_cache_keeps_only_one_model(monkeypatch):
    class FakeOracle:
        def __init__(self, *args, **kwargs):
            self.limits = [[-1, 1]] * 5

        def __call__(self, q):
            return True

    monkeypatch.setattr(benchmark, "CollisionOracle", FakeOracle)
    benchmark._ORACLES.clear()
    query = {"start_rad": [0] * 5, "goal_rad": [0.1] * 5}
    try:
        first = benchmark.problem_for({"id": 0}, query)
        assert benchmark.problem_for({"id": 0}, query).pose_valid is first.pose_valid
        for i in range(10):
            benchmark.problem_for({"id": i}, query)
            assert len(benchmark._ORACLES) == 1
    finally:
        benchmark._ORACLES.clear()
