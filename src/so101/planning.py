"""
RRT-Connet算法搜索路径规划
"""

import numpy as np

def extend_once(
    tree, parents, q_target, step_size,edge_valid,
):
    """尝试扩展一次；成功返回节点索引，受阻返回 None。"""
    nearest_index = None
    nearest_distance = float("inf")
    for index, node in enumerate(tree):
        distance = np.linalg.norm(q_target - node)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index

    q_near = tree[nearest_index]
    direction = q_target - q_near
    distance = np.linalg.norm(direction)
    if distance < 1e-12:
        return nearest_index
    if distance <= step_size:
        q_new = q_target.copy()
    else:
        q_new = q_near + step_size * direction / distance

    valid = edge_valid(q_near, q_new)
    if valid:
        tree.append(q_new.copy())
        parents.append(nearest_index)
        return len(tree) - 1
    else:
        return None


def connect(
    tree, parents, q_target, step_size, edge_valid,
):
    """朝固定目标连续扩展；接通返回索引，受阻返回 None。"""
    result = extend_once(
        tree, parents, q_target, step_size, edge_valid,
    )
    while True:
        if result is None:
            return None
        elif np.linalg.norm(tree[result] - q_target) < 1e-12:
            return result
        else:
            result = extend_once(
                tree, parents, q_target, step_size, edge_valid,
            )


def rrt_connect(
    q_start, q_goal,
    joint_limits, pose_valid, edge_valid,
    step_size,  max_iterations=1000, seed=42,
):
    """搜索并返回弧度姿态列表；无效起终点或预算耗尽返回 None。"""

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    joint_limits = np.asarray(joint_limits, dtype=float)

    rng = np.random.default_rng(seed)
    tree_start = [q_start.copy()]
    parents_start = [None]
    tree_goal = [q_goal.copy()]
    parents_goal = [None]
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    
    start_valid = pose_valid(tree_start[0])
    goal_valid = pose_valid(tree_goal[0])
    if not start_valid:
        return None
    if not goal_valid:
        return None

    if np.linalg.norm(q_start - q_goal) < 1e-12:
        return [q_start.copy()]

    found = False
    for attempt in range(max_iterations):
        if attempt % 2 == 0:
            tree_a, parents_a = tree_start, parents_start
            tree_b, parents_b = tree_goal, parents_goal
        else:
            tree_a, parents_a = tree_goal, parents_goal
            tree_b, parents_b = tree_start, parents_start
        index_a = extend_once(
            tree_a, parents_a,
            rng.uniform(lower, upper),
            step_size=step_size,
            edge_valid=edge_valid
        )
        if index_a is None:
            continue
        q_meet = tree_a[index_a]
        index_b = connect(
            tree_b, parents_b, q_meet, step_size,
            edge_valid=edge_valid
        )

        if index_b is not None:
            found = True
            if attempt % 2 == 0:
                path_start = []
                index = index_a
                while index is not None:
                    path_start.append(tree_start[index])
                    index = parents_start[index]
                path_start.reverse()

                path_goal = []
                index = index_b
                while index is not None:
                    path_goal.append(tree_goal[index])
                    index = parents_goal[index]
            else:
                path_start = []
                index = index_b
                while index is not None:
                    path_start.append(tree_start[index])
                    index = parents_start[index]
                path_start.reverse()

                path_goal = []
                index = index_a
                while index is not None:
                    path_goal.append(tree_goal[index])
                    index = parents_goal[index]
            break

    if found:
        path = path_start + path_goal[1:]
        return path
    else:
        return None
