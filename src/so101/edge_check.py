
import numpy as np


def is_edge_valid(q_start, q_goal, pose_valid, max_joint_step=np.deg2rad(1.0)):
    """逐点检查关节空间直线连接，返回 bool。
    一次求出五个关节中的最大变化量
    根据这个最大值计算区间数，只执行一次插值循环，包含起终点。
    任一点无效返回 False，循环结束返回 True    
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    for angles in (q_start, q_goal):
        if angles.shape != (5,) or not np.all(np.isfinite(angles)):
            raise ValueError("起终构型必须是五个有限弧度值")
    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError("max_joint_step 必须为正有限数")

    max_diff = np.max(np.abs(q_goal - q_start))

    if max_diff == 0:
        return bool(pose_valid(q_start))

    num_steps = max(1, int(np.ceil(max_diff / max_joint_step)))

    for step in range(num_steps + 1):
        s = step / num_steps
        q_interp = (1 - s) * q_start + s * q_goal
        if not pose_valid(q_interp):
            return False

    return True
