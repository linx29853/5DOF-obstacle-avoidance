"""规划器预算、图结构与可复现性测试。"""

import numpy as np
import pytest

from so101.planners import (
    Budget,
    Nearest,
    Options,
    Problem,
    Roadmap,
    Tree,
    plan,
    roadmap_key,
)


def free_problem():
    return Problem(np.zeros(5), np.ones(5) * 0.5, [[-1, 1]] * 5, lambda q: True, "free")


@pytest.mark.parametrize("planner", ["rrt", "rrt_connect", "rrt_star", "prm"])
def test_planner_returns_valid_endpoints(planner):
    p = free_problem()
    r = plan(p, planner, Budget(max_checks=15000), 42)
    assert r.path is not None
    np.testing.assert_allclose(r.path[0], p.start)
    np.testing.assert_allclose(r.path[-1], p.goal)
    assert r.stats["pose_checks"] <= 15000
    assert np.all(np.diff([h["length_rad"] for h in r.history]) < 0)
    assert r.first_solution_checks <= r.stats["pose_checks"]


@pytest.mark.parametrize("planner", ["rrt", "rrt_connect", "rrt_star", "prm"])
def test_budget_checked_inside_edges(planner):
    calls = []
    p = free_problem()
    p.pose_valid = lambda q: calls.append(q.copy()) or True
    r = plan(p, planner, Budget(max_checks=7), 42)
    assert len(calls) == r.stats["pose_checks"] == 7
    assert r.stats["termination"] == "pose_checks"
    assert r.path is None


def test_zero_time_and_invalid_endpoints():
    p = free_problem()
    assert plan(p, "rrt_connect", Budget(seconds=0)).stats["pose_checks"] == 0
    p.pose_valid = lambda q: False
    assert plan(p, "rrt", Budget(max_checks=100)).reason == "invalid_start"
    p.pose_valid = lambda q: not np.array_equal(q, p.goal)
    assert plan(p, "rrt", Budget(max_checks=100)).reason == "invalid_goal"


def test_same_pose_and_blocked():
    p = free_problem()
    p.goal = p.start.copy()
    assert len(plan(p, "rrt_star", Budget(max_checks=3)).path) == 1
    p = free_problem()
    p.pose_valid = lambda q: not 0.2 < q[0] < 0.3
    for planner in ["rrt", "rrt_connect", "rrt_star", "prm"]:
        assert plan(p, planner, Budget(max_checks=1500), 42).path is None


def test_tree_reparent_updates_descendants_and_rejects_cycle():
    t = Tree(np.zeros(5), np.ones(5), "linear")
    a = t.add(np.array([1.0, 1, 0, 0, 0]), 0)
    b = t.add(np.array([1.0, 0, 0, 0, 0]), a)
    c = t.add(np.array([2.0, 0, 0, 0, 0]), b)
    t.reparent(b, 0)
    assert t.costs[b] == pytest.approx(1)
    assert t.costs[c] == pytest.approx(2)
    assert b not in t.children[a] and b in t.children[0]
    with pytest.raises(ValueError, match="环"):
        t.reparent(b, c)


def test_kdtree_matches_linear_with_pending_tail_and_ties():
    a = Nearest(np.ones(5), "linear")
    b = Nearest(np.ones(5), "kdtree")
    rng = np.random.default_rng(5)
    for q in rng.random((90, 5)):
        a.add(q)
        b.add(q)
    for q in [np.zeros(5), np.ones(5), *rng.random((10, 5))]:
        for k in [1, 7, 90]:
            assert a.query(q, k) == b.query(q, k)
    a.add(np.zeros(5))
    b.add(np.zeros(5))
    a.add(np.zeros(5))
    b.add(np.zeros(5))
    assert a.query(np.zeros(5), 2) == b.query(np.zeros(5), 2)


def test_prm_graph_undirected_and_dijkstra():
    p = free_problem()
    road = Roadmap(roadmap_key(p, Options()), np.ones(5), "linear")
    for q in [np.zeros(5), np.ones(5), np.ones(5) * 0.5]:
        road.add(q)
    road.connect(0, 2)
    road.connect(2, 1)
    assert road.adjacency[0][2] == road.adjacency[2][0]
    path = road.shortest(0, 1)
    assert len(path) == 3
    p.payload_key = "changed"
    with pytest.raises(ValueError, match="重新构建"):
        plan(p, "prm", Budget(max_checks=100), roadmap=road)


def test_seed_reproduces_count_limited_search():
    p = free_problem()
    for planner in ["rrt", "rrt_connect", "rrt_star", "prm"]:
        first = plan(p, planner, Budget(max_checks=5000), 123)
        second = plan(p, planner, Budget(max_checks=5000), 123)
        assert first.stats["pose_checks"] == second.stats["pose_checks"]
        assert first.stats["samples"] == second.stats["samples"]
        if first.path is not None:
            np.testing.assert_allclose(first.path, second.path)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_budget(value):
    with pytest.raises(ValueError):
        Budget(seconds=value)
