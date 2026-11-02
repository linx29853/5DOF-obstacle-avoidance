# 五自由度避障 Demo

## 运行

在项目根目录运行：

```bash
.venv/bin/python examples/so101/demo.py
```

自动完成规划、路径简化、五次时间轨迹、碰撞复核和播放。关闭窗口退出。

```bash
.venv/bin/python examples/so101/demo.py --case single
.venv/bin/python examples/so101/demo.py --case blocked
.venv/bin/python examples/so101/demo.py --check-only
```

- `multi`：球和盒子，默认场景。
- `single`：单球障碍。
- `blocked`：障碍覆盖起点，输出失败原因。
- `--check-only`：计算、检查并保存结果，不打开窗口。

## 修改参数

编辑 `configs/so101/demo.json`：

- `start_degrees`、`goal_degrees`：起终关节角，单位度。
- `cases` 中的 `center`：障碍中心，单位米。
- 球的 `radius`：半径；盒子的 `half_size`：三个方向的半尺寸。
- `margin_mm`：安全裕量，单位毫米。
- `planning`：种子、扩展步长和搜索预算。
- `trajectory`：逐关节速度、加速度和采样周期。

## 输出

每次运行保存在 `outputs/so101/demo/` 下的独立目录：

- `request.json`、`scene.json`：本次参数和障碍场景。
- `path.npy`、`path_simplified.npy`：原路径和简化路径。
- `trajectory.npz`：时间、位置、速度、加速度。
- `result.json`：结果、时长、路点数量和版本信息。

播放按轨迹时间戳设置参考姿态；动力学跟踪在后续阶段开展。
