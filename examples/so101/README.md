# SO-101 实验入口

一键 Demo：`.venv/bin/python examples/so101/demo.py`。
统一参数在 configs/so101/demo.json，使用说明见 docs/so101/demo.md。

inspect_model.py 加载配置与模型，打印关节信息，并可选预览静态姿态。从仓库根目录运行：

```bash
.venv/bin/python examples/so101/inspect_model.py
.venv/bin/python examples/so101/inspect_model.py --view --degrees 20 0 0 0 0
```

角度顺序与 simulation.json 中的 arm_joint_names 一致，单位为度；夹爪使用配置的固定位置。预览不运行动力学仿真。

正运动学练习在 src/so101/forward_kinematics.py，参数从模型读取，参考点为 gripperframe，相对于 base 连杆坐标系：

```bash
.venv/bin/python examples/so101/forward_kinematics_exp.py --parameters
.venv/bin/python examples/so101/forward_kinematics_exp.py
```

第二条命令在核心函数未填写时提示尚未完成；实现后对照零位、五个单关节姿态及两组组合姿态。

碰撞练习入口（判断逻辑在 src/so101/collision.py）：

```bash
.venv/bin/python examples/so101/collision_exp.py --empty
.venv/bin/python examples/so101/collision_exp.py --view
.venv/bin/python examples/so101/collision_exp.py --empty --degrees 180 0 0 0 0
```

助手已检查：零位空场没有接触，加入教学球障碍后 lower_arm 与球发生穿透。
超限位姿态仍可用于观察，是否有效由学习者函数判定。默认使用原来的单球场景。

第一个入口只加载模型、显示关节名称并逐个预览关节，不连接硬件。
五关节连接检查练习在 src/so101/edge_check.py，运行 `.venv/bin/python examples/so101/edge_check_exp.py` 查看四组连接实验。
后续依次加入碰撞检查、静态规划、时间轨迹与跟踪。

每个入口需要说明命令、依赖、配置、预期输出和验证状态。
规划与简化入口（均从项目根目录运行）：

```bash
.venv/bin/python examples/so101/planning_exp.py --empty
.venv/bin/python examples/so101/planning_exp.py --output outputs/so101/path.npy
.venv/bin/python examples/so101/path_simplification_exp.py
```

简化入口使用球障碍场景；输入路径应由同一场景生成。

播放简化后的路径：

```bash
.venv/bin/python examples/so101/play_path_exp.py
.venv/bin/python examples/so101/play_path_exp.py --check-only
```

默认 8 秒播放，结束后保留终点画面，关闭窗口退出。`--path outputs/so101/path.npy` 可播放原路径。

桌面、底座与多障碍场景：

```bash
.venv/bin/python examples/so101/planning_exp.py --scene configs/so101/scene_table_multi.json --output outputs/so101/multi/path.npy
.venv/bin/python examples/so101/path_simplification_exp.py --scene configs/so101/scene_table_multi.json --path outputs/so101/multi/path.npy --output outputs/so101/multi/path_simplified.npy
.venv/bin/python examples/so101/play_path_exp.py --scene configs/so101/scene_table_multi.json --path outputs/so101/multi/path_simplified.npy
```

盒体尺寸使用 half_size（半尺寸），桌面顶面 z=-0.004 m。底座碰撞盒包围官方底座的四个网格。
底座与肩的固定安装连接不检查碰撞；保留模型默认的相邻连杆过滤，其他连杆对进行自碰撞检查。
`--empty --scene configs/so101/scene_table_multi.json` 仅移除障碍，保留桌面和底座。

场景回归：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/so101/test_scene.py -q`。

安全裕量练习：填写 src/so101/collision.py 的 has_clearance，运行 `.venv/bin/python examples/so101/clearance_exp.py`。
场景查询器会同时提供 5 mm 范围内尚未接触的几何体间隙。

规划、简化和播放均支持 `--margin-mm 5`，三个入口使用同一个裕量。
已生成的 5 mm 多障碍路径在 outputs/so101/margin/path_simplified.npy。

固定场景多种子评测：`.venv/bin/python examples/so101/evaluate_planning_exp.py`。
默认每个场景运行种子 0～4，评测记录保存至 outputs/so101/evaluation.json。

五关节时间轨迹练习：填写 src/so101/trajectory.py，运行 `.venv/bin/python examples/so101/trajectory_exp.py`。
默认检查 5 mm 多障碍路径的第一段；`--segment 1` 或 `--segment 2` 检查后续段。
真机入口另行明确命名，默认不开启电机或执行运动。
