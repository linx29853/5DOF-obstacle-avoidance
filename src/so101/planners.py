"""五维采样规划器、共享预算和运行计量。"""

import heapq
import math
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from so101.benchmark_resources import check_resources


@dataclass
class Problem:
    start: np.ndarray
    goal: np.ndarray
    joint_limits: np.ndarray
    pose_valid: object
    scene_key: str = ""
    payload_key: str = "empty"

    def __post_init__(self):
        self.start = np.asarray(self.start, float)
        self.goal = np.asarray(self.goal, float)
        self.joint_limits = np.asarray(self.joint_limits, float)
        if (
            self.start.shape != (5,)
            or self.goal.shape != (5,)
            or self.joint_limits.shape != (5, 2)
        ):
            raise ValueError("规划问题必须使用五关节构型及 (5,2) 限位")
        if not all(
            np.all(np.isfinite(x)) for x in [self.start, self.goal, self.joint_limits]
        ):
            raise ValueError("规划输入必须有限")
        if np.any(self.joint_limits[:, 0] >= self.joint_limits[:, 1]):
            raise ValueError("关节下限必须小于上限")


@dataclass
class Budget:
    seconds: float | None = None
    max_checks: int | None = None
    max_samples: int | None = None

    def __post_init__(self):
        if (
            self.seconds is None
            and self.max_checks is None
            and self.max_samples is None
        ):
            raise ValueError("至少指定一种预算")
        if self.seconds is not None and (
            not np.isfinite(self.seconds) or self.seconds < 0
        ):
            raise ValueError("时间预算必须为非负有限数")
        for value in [self.max_checks, self.max_samples]:
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("计数预算必须为非负整数")


@dataclass
class Options:
    step_size: float = float(np.deg2rad(15))
    edge_resolution: float = float(np.deg2rad(0.5))
    goal_bias: float | None = None
    nearest: str = "linear"
    metric: str = "radians"
    prm_neighbors: int = 15
    prm_batch: int = 32
    stop_on_first: bool = False

    def __post_init__(self):
        if self.nearest not in ("linear", "kdtree") or self.metric not in (
            "radians",
            "normalized",
        ):
            raise ValueError("未知最近邻或距离模式")
        if (
            not np.isfinite(self.step_size)
            or self.step_size <= 0
            or not np.isfinite(self.edge_resolution)
            or self.edge_resolution <= 0
        ):
            raise ValueError("步长和连接分辨率必须为正有限数")
        if self.goal_bias is not None and not 0 <= self.goal_bias <= 1:
            raise ValueError("目标偏置必须位于 [0,1]")
        if self.prm_neighbors < 1 or self.prm_batch < 1:
            raise ValueError("PRM 邻居和批大小必须为正数")


@dataclass
class Result:
    path: list | None = None
    reason: str = "budget_exhausted"
    first_solution_seconds: float | None = None
    first_solution_checks: int | None = None
    history: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class Exhausted(Exception):
    pass


class Nearest:
    """精确最近邻；KD-tree 包含已建树前缀与线性查询尾部。"""

    def __init__(self, scale, backend="linear"):
        self.scale = scale
        self.backend = backend
        self.points = []
        self.tree = None
        self.frozen = 0
        self.seconds = 0.0

    def add(self, q):
        self.points.append(np.asarray(q) * self.scale)
        if self.backend == "kdtree" and (
            self.tree is None or len(self.points) - self.frozen >= 64
        ):
            began = perf_counter()
            self.tree = cKDTree(np.asarray(self.points))
            self.frozen = len(self.points)
            self.seconds += perf_counter() - began

    def query(self, q, k=1):
        began = perf_counter()
        p = np.asarray(q) * self.scale
        k = min(k, len(self.points))
        if self.backend == "linear":
            distances = np.linalg.norm(np.asarray(self.points) - p, axis=1)
            answer = np.lexsort((np.arange(len(distances)), distances))[:k].tolist()
        else:
            dist, ids = self.tree.query(p, k=min(k, self.frozen))
            candidates = [
                (float(d), int(i))
                for d, i in zip(np.atleast_1d(dist), np.atleast_1d(ids))
            ]
            candidates += [
                (float(np.linalg.norm(v - p)), i)
                for i, v in enumerate(self.points[self.frozen :], self.frozen)
            ]
            # 边界并列节点也纳入，保持与线性搜索相同的稳定排序。
            radius = (
                max(d for d, _ in candidates)
                if len(candidates) <= k
                else sorted(candidates)[k - 1][0]
            )
            candidates += [
                (float(np.linalg.norm(self.points[i] - p)), i)
                for i in self.tree.query_ball_point(p, radius + 1e-12)
            ]
            unique = {
                i: float(np.linalg.norm(self.points[i] - p)) for _, i in candidates
            }
            answer = sorted(unique, key=lambda i: (unique[i], i))[:k]
        self.seconds += perf_counter() - began
        return answer


class Tree:
    def __init__(self, start, scale, backend):
        self.nodes = [start.copy()]
        self.parents = [None]
        self.children = [set()]
        self.costs = [0.0]
        self.nn = Nearest(scale, backend)
        self.nn.add(start)
        self.scale = scale

    def add(self, q, parent):
        i = len(self.nodes)
        self.nodes.append(q.copy())
        self.parents.append(parent)
        self.children.append(set())
        self.children[parent].add(i)
        self.costs.append(
            self.costs[parent]
            + float(np.linalg.norm((q - self.nodes[parent]) * self.scale))
        )
        self.nn.add(q)
        return i

    def path(self, i):
        out = []
        while i is not None:
            out.append(self.nodes[i].copy())
            i = self.parents[i]
        return out[::-1]

    def reparent(self, node, parent):
        cur = parent
        while cur is not None:
            if cur == node:
                raise ValueError("重连会形成环")
            cur = self.parents[cur]
        old = self.parents[node]
        self.children[old].remove(node)
        self.parents[node] = parent
        self.children[parent].add(node)
        todo = [node]
        while todo:
            i = todo.pop()
            p = self.parents[i]
            self.costs[i] = self.costs[p] + float(
                np.linalg.norm((self.nodes[i] - self.nodes[p]) * self.scale)
            )
            todo.extend(self.children[i])


class Context:
    def __init__(self, problem, budget, options, seed):
        self.p, self.b, self.o = problem, budget, options
        self.rng = np.random.default_rng(seed)
        self.began = perf_counter()
        self.scale = (
            1 / (problem.joint_limits[:, 1] - problem.joint_limits[:, 0])
            if options.metric == "normalized"
            else np.ones(5)
        )
        self.result = Result()
        self.s = {
            "samples": 0,
            "accepted_samples": 0,
            "valid_extensions": 0,
            "edge_checks": 0,
            "pose_checks": 0,
            "collision_seconds": 0.0,
            "nearest_seconds": 0.0,
            "nodes": 0,
            "rewires": 0,
            "termination": "first_solution",
        }
        self.indices = []

    def gate(self):
        check_resources()
        if self.b.seconds is not None and perf_counter() - self.began >= self.b.seconds:
            self.s["termination"] = "wall_time"
            raise Exhausted
        if self.b.max_checks is not None and self.s["pose_checks"] >= self.b.max_checks:
            self.s["termination"] = "pose_checks"
            raise Exhausted

    def valid(self, q):
        self.gate()
        if np.any(q < self.p.joint_limits[:, 0]) or np.any(
            q > self.p.joint_limits[:, 1]
        ):
            return False
        self.s["pose_checks"] += 1
        began = perf_counter()
        try:
            return bool(self.p.pose_valid(q))
        finally:
            self.s["collision_seconds"] += perf_counter() - began

    def edge(self, a, b):
        self.s["edge_checks"] += 1
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / self.o.edge_resolution)))
        for u in np.linspace(0, 1, n + 1):
            if not self.valid(a + (b - a) * u):
                return False
        return True

    def sample(self, target, bias):
        self.gate()
        if self.b.max_samples is not None and self.s["samples"] >= self.b.max_samples:
            self.s["termination"] = "samples"
            raise Exhausted
        self.s["samples"] += 1
        if bias and self.rng.random() < bias:
            return target.copy()
        return self.rng.uniform(self.p.joint_limits[:, 0], self.p.joint_limits[:, 1])

    def distance(self, a, b):
        return float(np.linalg.norm((a - b) * self.scale))

    def consider(self, path):
        cost = (
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if len(path) > 1
            else 0.0
        )
        if (
            self.result.history
            and cost >= self.result.history[-1]["length_rad"] - 1e-12
        ):
            return
        now = perf_counter() - self.began
        if self.result.path is None:
            self.result.first_solution_seconds = now
            self.result.first_solution_checks = self.s["pose_checks"]
        self.result.path = [q.copy() for q in path]
        self.result.history.append(
            {"seconds": now, "pose_checks": self.s["pose_checks"], "length_rad": cost}
        )

    def extend(self, tree, target):
        self.gate()
        near = tree.nn.query(target)[0]
        delta = target - tree.nodes[near]
        length = self.distance(target, tree.nodes[near])
        if length < 1e-12:
            return near
        q = tree.nodes[near] + delta * min(1.0, self.o.step_size / length)
        if not self.edge(tree.nodes[near], q):
            return None
        self.s["valid_extensions"] += 1
        return tree.add(q, near)


def _rrt(ctx, star=False):
    p, o = ctx.p, ctx.o
    t = Tree(p.start, ctx.scale, o.nearest)
    ctx.indices.append(t.nn)
    bias = 0.05 if o.goal_bias is None else o.goal_bias
    goal_links = set()
    while True:
        ctx.gate()
        target = ctx.sample(p.goal, bias)
        near = t.nn.query(target)[0]
        delta = target - t.nodes[near]
        dist = ctx.distance(target, t.nodes[near])
        if dist < 1e-12:
            continue
        q = t.nodes[near] + delta * min(1.0, o.step_size / dist)
        if not ctx.edge(t.nodes[near], q):
            continue
        parent = near
        neighbors = []
        if star:
            k = max(1, math.ceil(2 * math.e * (1 + 1 / 5) * math.log(len(t.nodes) + 1)))
            neighbors = t.nn.query(q, k)
            for i in sorted(
                neighbors, key=lambda i: t.costs[i] + ctx.distance(t.nodes[i], q)
            ):
                ctx.gate()
                if t.costs[i] + ctx.distance(t.nodes[i], q) < t.costs[
                    parent
                ] + ctx.distance(t.nodes[parent], q) - 1e-12 and ctx.edge(
                    t.nodes[i], q
                ):
                    parent = i
        node = t.add(q, parent)
        ctx.s["accepted_samples"] += 1
        ctx.s["valid_extensions"] += 1
        ctx.s["nodes"] = len(t.nodes)
        if star:
            for i in neighbors:
                ctx.gate()
                if (
                    i != 0
                    and t.costs[node] + ctx.distance(q, t.nodes[i]) < t.costs[i] - 1e-12
                    and ctx.edge(q, t.nodes[i])
                ):
                    t.reparent(i, node)
                    ctx.s["rewires"] += 1
        if ctx.distance(q, p.goal) <= o.step_size and ctx.edge(q, p.goal):
            goal_links.add(node)
        if goal_links:
            best = min(
                goal_links, key=lambda i: t.costs[i] + ctx.distance(t.nodes[i], p.goal)
            )
            path = t.path(best)
            if not np.allclose(path[-1], p.goal, atol=1e-12, rtol=0):
                path.append(p.goal.copy())
            ctx.consider(path)
            if not star or o.stop_on_first:
                return


def _connect(ctx):
    p, o = ctx.p, ctx.o
    trees = [Tree(p.start, ctx.scale, o.nearest), Tree(p.goal, ctx.scale, o.nearest)]
    ctx.indices.extend(t.nn for t in trees)
    attempt = 0
    bias = 0.0 if o.goal_bias is None else o.goal_bias
    while True:
        a, b = trees[attempt % 2], trees[1 - attempt % 2]
        target = ctx.sample(b.nodes[0], bias)
        ia = ctx.extend(a, target)
        if ia is not None:
            ctx.s["accepted_samples"] += 1
            while True:
                ib = ctx.extend(b, a.nodes[ia])
                if ib is None:
                    break
                if ctx.distance(b.nodes[ib], a.nodes[ia]) < 1e-12:
                    pa, pb = a.path(ia), b.path(ib)
                    path = pa + pb[-2::-1] if attempt % 2 == 0 else pb + pa[-2::-1]
                    ctx.consider(path)
                    ctx.s["nodes"] = sum(len(t.nodes) for t in trees)
                    return
        ctx.s["nodes"] = sum(len(t.nodes) for t in trees)
        attempt += 1


class Roadmap:
    """无向 PRM；仅可在相同场景、载荷、度量与碰撞规则下复用。"""

    def __init__(self, key, scale, backend):
        self.key = key
        self.nodes = []
        self.adjacency = []
        self.nn = Nearest(scale, backend)
        self.scale = scale

    def add(self, q):
        i = len(self.nodes)
        self.nodes.append(q.copy())
        self.adjacency.append({})
        self.nn.add(q)
        return i

    def connect(self, a, b):
        cost = float(np.linalg.norm((self.nodes[a] - self.nodes[b]) * self.scale))
        self.adjacency[a][b] = cost
        self.adjacency[b][a] = cost

    def shortest(self, start, goal, gate=lambda: None):
        distances = {start: 0.0}
        parents = {start: None}
        queue = [(0.0, start)]
        while queue:
            gate()
            dist, node = heapq.heappop(queue)
            if dist != distances[node]:
                continue
            if node == goal:
                path = []
                while node is not None:
                    path.append(self.nodes[node].copy())
                    node = parents[node]
                return path[::-1]
            for neighbor, cost in self.adjacency[node].items():
                candidate = dist + cost
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        return None


def roadmap_key(problem, options):
    return (
        problem.scene_key,
        problem.payload_key,
        options.metric,
        options.nearest,
        options.edge_resolution,
        tuple(problem.joint_limits.ravel()),
    )


def _prm(ctx, roadmap=None):
    key = roadmap_key(ctx.p, ctx.o)
    road = roadmap or Roadmap(key, ctx.scale, ctx.o.nearest)
    if road.key != key:
        raise ValueError("场景、载荷或碰撞规则变化，需要重新构建路网")
    ctx.indices.append(road.nn)

    def insert(q):
        neighbors = road.nn.query(q, ctx.o.prm_neighbors) if road.nodes else []
        i = road.add(q)
        for j in neighbors:
            if ctx.edge(q, road.nodes[j]):
                road.connect(i, j)
        ctx.s["nodes"] = len(road.nodes)
        return i

    start = insert(ctx.p.start)
    goal = insert(ctx.p.goal)

    def query():
        path = road.shortest(start, goal, ctx.gate)
        if path is not None:
            ctx.consider(path)

    query()
    if ctx.result.path is not None and ctx.o.stop_on_first:
        return
    while True:
        for _ in range(ctx.o.prm_batch):
            q = ctx.sample(ctx.p.goal, 0.0)
            if ctx.valid(q):
                ctx.s["accepted_samples"] += 1
                insert(q)
        query()
        if ctx.result.path is not None and ctx.o.stop_on_first:
            return


def plan(problem, planner, budget, seed=42, options=None, roadmap=None):
    """统一规划入口；所有算法在连接内部响应预算。"""
    if planner not in ("rrt", "rrt_connect", "rrt_star", "prm", "apf"):
        raise ValueError("未知规划器")
    options = options or Options()
    ctx = Context(problem, budget, options, seed)
    try:
        if not ctx.valid(problem.start):
            ctx.result.reason = "invalid_start"
            return ctx.result
        if not ctx.valid(problem.goal):
            ctx.result.reason = "invalid_goal"
            return ctx.result
        if np.array_equal(problem.start, problem.goal):
            ctx.consider([problem.start])
            return ctx.result
        if planner == "apf":
            from so101.apf import search

            search(ctx)
        elif planner == "rrt":
            _rrt(ctx)
        elif planner == "rrt_connect":
            _connect(ctx)
        elif planner == "rrt_star":
            _rrt(ctx, star=True)
        else:
            _prm(ctx, roadmap)
    except Exhausted:
        pass
    finally:
        ctx.s["seconds"] = perf_counter() - ctx.began
        ctx.s["nearest_seconds"] = sum(i.seconds for i in ctx.indices)
        ctx.result.stats = ctx.s
        if ctx.result.path is not None:
            ctx.result.reason = "success"
    return ctx.result
