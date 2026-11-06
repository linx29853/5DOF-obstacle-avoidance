# 固定 SO-101 桌面操作仿真

## 场景与动作

项目在现有 SO-101 MuJoCo 模型上增加物体搬运任务，基座保持固定。任务动作参考 [Mobile-Manipulation](https://github.com/Sharwin24/Mobile-Manipulation) 的分阶段抓取与放置流程。

| 场景 | 任务 | 物料数量 |
| --- | --- | --- |
| `tabletop` | 将桌面方块从一侧搬到另一侧的色块目标区域 | 1 |
| `barrier` | 抬高方块，跨过工作台隔挡后放置 | 1 |
| `sorting` | 将两个不同颜色的方块依次搬入各自目标区域 | 2 |

每个物料依次执行：接近 → 下降 → 闭合 → 抬升 → 转移 → 降低 → 张开 → 撤离。所有物料操作结束后返回初始工作位姿。

## 安装与运行

```bash
python -m pip install -e ".[sim,dev,media,plot]"
python examples/so101/manipulation.py --case tabletop
python examples/so101/manipulation.py --case barrier
python examples/so101/manipulation.py --case sorting
```

执行计算和几何复核：

```bash
python examples/so101/manipulation.py --case sorting --check-only
```

导出 GIF、封面和轨迹图：

```bash
MUJOCO_GL=egl python examples/so101/manipulation.py --case barrier --check-only --export
```

`egl` 用于支持 EGL 的无窗口渲染环境。桌面图形环境可直接运行并省略该环境变量。

## 实现方法

- **逆运动学**：SciPy 有界最小二乘同时约束末端位置及朝下的工具接近轴；五自由度机械臂保留绕接近轴的自由度。使用上一解作为下一路点初值，以减小关节跳变。
- **动作路径**：按 6 mm 间距离散任务空间直线。转移路径遇到碰撞时调用项目原有 RRT-Connect 和路径简化模块，搜索关节空间绕行路径。
- **时间轨迹**：PCHIP 插值关节路点，以五次时间缩放生成全段运动；根据离散速度、加速度峰值增加时长，并复核插值后的连接。
- **夹爪与物体**：夹爪角度单独插值。闭合结束后记录物体到工具的相对位姿；搬运时通过刚体坐标变换更新物体，张开前解除绑定并保留释放位置。
- **仿真方式**：运动学回放使用 `mj_forward`，物料使用 mocap 刚体。抓取采用显式刚体绑定；摩擦抓持、滑落、物体落地及伺服动力学属于后续实验任务。
- **几何复核**：机械臂检查场景接触、自碰撞和关节限位；物体与非夹持连杆进行距离检查，搬运中的物体额外检查桌面、隔挡及其他物体。固定指和活动指允许与物料接触。连接按最大逐关节 0.5° 间隔检查。

目标色块用于标识放置区域，碰撞障碍由配置中的实体盒体定义。分拣动作按配置顺序执行，物料位置和类别来自配置。

## 配置

配置文件：`configs/so101/manipulation.json`。

| 参数 | 含义 |
| --- | --- |
| `items[].source` / `destination` | 物体中心的世界坐标，单位 m |
| `items[].half_size` | 物体的三轴半尺寸，单位 m |
| `home` | 工具初始位置，单位 m |
| `transport_height_m` | 工具抬升和转移的高度 |
| `grasp_offset_m` | 抓取时工具参考点高于物体中心的距离 |
| `gripper_open_rad` / `gripper_closed_rad` | 夹爪张开、闭合角度 |
| `velocity_limits_rad_s` | 五关节速度限制 |
| `acceleration_limits_rad_s2` | 五关节加速度限制 |
| `obstacles` | 实体隔挡等环境盒体 |

SO-101 的朝下抓取工作空间受到腕关节限位约束。默认场景的目标位于基座前方约 0.16～0.21 m、左右约 0.10 m，转移高度为 0.085 m。修改坐标后通过 `--check-only` 检查可达性与碰撞。

## 输出与结果

每次运行创建独立的 `outputs/so101/manipulation/<场景>-<编号>/`：

- `request.json`：本次配置快照。
- `trajectory.npz`：时间、五关节角、夹爪角、物体位置与四元数、动作阶段、末端路径。
- `result.json`：成功状态、放置误差、速度与加速度峰值、碰撞检查次数、依赖版本及源码校验值。
- `demo.gif` / `demo.png`：通过 `--export` 生成的动画与封面。
- `trajectory.png`：末端与物料三维路径、关节角时间曲线。

在 MuJoCo 3.17.0、NumPy 2.4.6 环境下，默认桌面搬运约 14.2 s，隔挡转移约 15.5 s，双物料分拣约 64.2 s。三个场景的物料最终中心均落在目标 1 mm 范围内；该误差衡量运动学任务落点，伺服跟踪误差需通过动力学控制实验另行测量。

`max_deviation_from_straight_line_m` 记录实际末端路径相对直线参考的偏离。关节空间绕行允许偏离直线，分拣中的绕行约 8.85 cm。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/so101 -q
```

测试覆盖三个场景的阶段顺序、夹持前静止、搬运抬升、释放后位置保持、关节速度与加速度约束、逐帧几何复核、不可达目标以及阻挡场景失败记录。

全仓库测试中的两项二连杆异常文本断言要求英文 `unreachable`，对应实现返回中文“目标位置不可达”。这两项断言与固定 SO-101 操作模块分开维护。
