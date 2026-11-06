"""确定性关节空间 APF：环境距离斥力、有限差分梯度和回溯下降。"""

from time import perf_counter

import numpy as np


def search(ctx):
    repulsion = getattr(ctx.p.pose_valid, "repulsive_potential", None)
    if repulsion is None:
        raise ValueError("APF 需要提供 repulsive_potential(q) 的碰撞接口")
    ctx.s.update(potential_evaluations=0, potential_seconds=0.0)

    def energy(q):
        ctx.gate()
        # 势能评估包含一次前向运动学，计入构型资源预算。
        ctx.s["pose_checks"] += 1
        ctx.s["potential_evaluations"] += 1
        began = perf_counter()
        try:
            return 0.5 * float(np.sum((q - ctx.p.goal) ** 2)) + repulsion(q)
        finally:
            ctx.s["potential_seconds"] += perf_counter() - began

    q = ctx.p.start.copy()
    path = [q.copy()]
    low, high = ctx.p.joint_limits.T
    epsilon = 0.001
    while True:
        ctx.gate()
        if np.linalg.norm(q - ctx.p.goal) <= ctx.o.step_size and ctx.edge(
            q, ctx.p.goal
        ):
            ctx.consider(path + [ctx.p.goal.copy()])
            return
        current = energy(q)
        gradient = np.zeros(5)
        for j in range(5):
            a, b = q.copy(), q.copy()
            a[j] = min(high[j], q[j] + epsilon)
            b[j] = max(low[j], q[j] - epsilon)
            gradient[j] = (energy(a) - energy(b)) / (a[j] - b[j])
        norm = np.linalg.norm(gradient)
        if norm < 1e-8:
            ctx.result.reason = "local_minimum"
            return
        direction = -gradient / norm
        accepted = False
        for step in ctx.o.step_size * 0.5 ** np.arange(14):
            candidate = np.clip(q + direction * step, low, high)
            if np.linalg.norm(candidate - q) < 1e-8:
                continue
            if energy(candidate) < current - 1e-10 and ctx.edge(q, candidate):
                q = candidate
                path.append(q.copy())
                ctx.s["valid_extensions"] += 1
                ctx.s["nodes"] = len(path)
                accepted = True
                break
        if not accepted:
            ctx.result.reason = "local_minimum_or_blocked"
            return
