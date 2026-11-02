
import numpy as np


def quintic_trajectory(q_start, q_goal, velocity_limits, acceleration_limits, sample_count=101):
    """返回 times、positions、velocities、accelerations。

    输入姿态为长度为 5 的数组，单位 rad；逐关节限制均为长度为 5 的正数组，
    单位分别为 rad/s、rad/s²。输出形状为 (N,) 及三个 (N, 5)。
    按五次曲线的理论峰值选择时长，不检查路径碰撞或执行器力矩。
    起终点完全相同时返回 N 个零时间、重复位置及零速度、零加速度。
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    velocity_limits = np.asarray(velocity_limits, dtype=float)
    acceleration_limits = np.asarray(acceleration_limits, dtype=float)
    for name, values in (
        ("q_start", q_start), ("q_goal", q_goal),
        ("velocity_limits", velocity_limits), ("acceleration_limits", acceleration_limits),
    ):
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} 必须为五个有限数")
    if np.any(velocity_limits <= 0) or np.any(acceleration_limits <= 0):
        raise ValueError("速度和加速度限制必须为正")
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)) or sample_count < 2:
        raise ValueError("sample_count 必须为至少 2 的整数")

    delta_q = q_goal - q_start
    if np.all(delta_q == 0.0):
        times = np.zeros(sample_count)
        positions = np.repeat(q_start[None, :], sample_count, axis=0)
        return times, positions, np.zeros_like(positions), np.zeros_like(positions)

    T_velocity = 1.875 * np.max(np.abs(delta_q) / velocity_limits)
    T_acceleration = np.sqrt(
        (10 / np.sqrt(3)) * np.max(np.abs(delta_q) / acceleration_limits)
    )
    T = max(T_velocity, T_acceleration)
    times = np.linspace(0.0, T, sample_count)
    positions = []
    velocities = []
    accelerations = []

    for t in times:
        u = t / T
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds_du = 30 * u**2 - 60 * u**3 + 30 * u**4
        d2s_du2 = 60 * u - 180 * u**2 + 120 * u**3
        q = q_start + s * delta_q
        q_dot = ds_du * delta_q / T
        q_ddot = d2s_du2 * delta_q / T**2
        positions.append(q.copy())
        velocities.append(q_dot.copy())
        accelerations.append(q_ddot.copy())

    return (
        times,
        np.array(positions),
        np.array(velocities),
        np.array(accelerations),
    )
