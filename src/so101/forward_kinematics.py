
import numpy as np


def rotation_z(angle):
    """返回绕 z 轴旋转的 4×4 齐次变换矩阵。"""
    return np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

def forward_kinematics(joint_angles, fixed_transforms, tool_transform):
    """返回 base 到 gripperframe 的 4×4 变换。
    joint_angles: 五关节弧度数组，顺序为底座、肩、肘、腕俯仰、腕旋转。
    fixed_transforms: (5, 4, 4)，每段的固定安装变换，已由模型读取。
    tool_transform: (4, 4)，第五连杆到工具参考点的固定变换。

    从单位矩阵出发，按顺序依次右乘当前段的固定变换和关节旋转；
    五段完成后右乘工具变换，再返回结果。
    """

    joint_angles = np.asarray(joint_angles, dtype=float)
    fixed_transforms = np.asarray(fixed_transforms, dtype=float)
    tool_transform = np.asarray(tool_transform, dtype=float)

    for name, array, shape in [
        ("joint_angles", joint_angles, (5,)),
        ("fixed_transforms", fixed_transforms, (5, 4, 4)),
        ("tool_transform", tool_transform, (4, 4)),
    ]:
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} 必须形状为 {shape} 且所有值有限")

    final_transform = np.eye(4)
    for i in range(5):
        final_transform = final_transform @ fixed_transforms[i] @ rotation_z(joint_angles[i])
    final_transform = final_transform @ tool_transform

    return final_transform
