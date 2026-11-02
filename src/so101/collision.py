import numpy as np


def is_pose_valid(joint_angles, joint_limits, contacts):
    """检查关节角度是否在关节限制内，且所有接触点距离大于零"""
    joint_angles = np.asarray(joint_angles, dtype=float)
    joint_limits = np.asarray(joint_limits, dtype=float)
    if joint_angles.shape != (5,) or not np.all(np.isfinite(joint_angles)):
        raise ValueError("joint_angles 必须是五个有限弧度值")
    if joint_limits.shape != (5, 2) or not np.all(np.isfinite(joint_limits)):
        raise ValueError("joint_limits 必须是 (5, 2) 有限数组")
    if np.any(joint_limits[:, 0] > joint_limits[:, 1]):
        raise ValueError("关节下限不能大于上限")

    for i in range(5):
        if not (joint_limits[i, 0] <= joint_angles[i] <= joint_limits[i, 1]):
            return False
        

    for contact in contacts:
        if contact["dist"] <= 0:
            return False

    return True


def has_clearance(contacts, safety_margin):
    if not np.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("safety_margin 必须为非负有限数")
    for contact in contacts:
        if contact["dist"] <= safety_margin:
            return False
    return True
