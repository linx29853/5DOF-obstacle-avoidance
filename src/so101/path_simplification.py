
def simplify_path(path, edge_valid):
    """
    从当前路点优先尝试最远后续路点，保留起终点并复制输出构型。
    删除二连杆几何参数和采样数计算，连接检查直接调用传入的 edge_valid。
    空路径返回 []，单节点路径返回该节点的副本。
    """
    #处理空路径，直接返回空列表
    if not path:
        return []
    #初始化简化路径列表，添加起点
    simplified_path = [path[0].copy()]
    i = 0
    #遍历路径，尝试找到最远的可行连接点
    while i < len(path) - 1:
        next_index = i + 1  # 原路径已验证，相邻连接作为保底

        for j in range(len(path) - 1, i + 1, -1):
            q_from = path[i]
            q_to = path[j]
            direct_valid = edge_valid(q_from, q_to,)
            #如果找到可行连接点，更新 next_index 并退出循环
            if direct_valid:
                next_index = j
                break
        #将找到的最远可行连接点添加到简化路径中
        simplified_path.append(path[next_index].copy())
        i = next_index

    return simplified_path