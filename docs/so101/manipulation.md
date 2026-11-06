# 固定 SO-101 桌面操作仿真

## 场景与动作

项目在现有 SO-101 MuJoCo 模型上增加物体搬运任务，基座保持固定。任务动作参考 [Mobile-Manipulation](https://github.com/Sharwin24/Mobile-Manipulation) 的分阶段抓取与放置流程。

| 场景 | 任务 | 物料数量 |
| --- | --- | --- |
| `tabletop` | 将桌面方块从一侧搬到另一侧的色块目标区域 | 1 |
| `barrier` | 抬高方块，跨过工作台隔挡后放置 | 1 |
| `sorting` | 将两个不同颜色的方块依次搬入各自目标区域 | 2 |
| `detour` | 直线路径被 10 cm 高隔挡阻断，物块从单侧绕行 | 1 |
| `clutter` | 两块左右交错的 10 cm 高挡板，形成 S 形搬运通道 | 1 |

每个物料依次执行：接近 → 下降 → 闭合 → 抬升 → 转移 → 降低 → 张开 → 撤离。所有物料操作结束后返回初始工作位姿。

## 安装与运行

```bash
python -m pip install -e ".[sim,dev,media,plot]"
python examples/so101/manipulation.py --case tabletop
python examples/so101/manipulation.py --case barrier
python examples/so101/manipulation.py --case sorting
python examples/so101/manipulation.py --case detour
python examples/so101/manipulation.py --case clutter
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

- **逆运动学**：SciPy 有界最小二乘约束两指夹持中心的位置及朝下、与方块对齐的工具方向。使用上一解作为下一路点初值，以减小关节跳变。
- **动作路径**：按 6 mm 间距离散任务空间直线。转移、接近和空手返回路径遇到碰撞时调用项目原有 RRT-Connect 和路径简化模块，搜索关节空间绕行路径。
- **时间轨迹**：PCHIP 插值关节路点，以五次时间缩放生成全段运动；根据离散速度、加速度峰值增加时长，并复核插值后的连接。
- **夹爪与物体**：夹爪闭合角通过两侧夹指到物块的距离求解。固定夹指与活动夹指的最近夹持面必须分列物块中心两侧，两侧间隙与穿透绝对值均不超过 1 mm，才记录物体到工具的相对位姿；搬运时通过刚体坐标变换更新物体，张开前解除绑定并保留释放位置。
- **仿真方式**：运动学回放使用 `mj_forward`，物料使用 mocap 刚体。抓取采用显式刚体绑定；摩擦抓持、滑落、物体落地及伺服动力学属于后续实验任务。
- **几何复核**：机械臂检查场景接触、自碰撞和关节限位；物体与非夹持连杆进行距离检查，搬运中的物体额外检查桌面、隔挡及其他物体。固定指和活动指允许与物料接触，绑定前单独执行双侧接触检查。凹形指部网格沿纵向分为 21 个凸片，避免整个零件的凸包填满夹持空隙；外观网格保持原模型。连接按最大逐关节 0.5° 间隔检查。

目标色块用于标识放置区域，碰撞障碍由配置中的实体盒体定义。分拣动作按配置顺序执行，物料位置和类别来自配置。

## 配置

配置文件：`configs/so101/manipulation.json`。

| 参数 | 含义 |
| --- | --- |
| `items[].source` / `destination` | 物体中心的世界坐标，单位 m |
| `items[].half_size` | 物体的三轴半尺寸，单位 m |
| `home` | 工具初始位置，单位 m |
| `transport_height_m` | 工具抬升和转移的高度 |
| `gripper_open_rad` | 夹爪张开角度；闭合角由双侧距离自动求解 |
| `velocity_limits_rad_s` | 五关节速度限制 |
| `acceleration_limits_rad_s2` | 五关节加速度限制 |
| `obstacles` | 实体隔挡等环境盒体 |
| `transfer_waypoints_m` | 携物转移的通道引导点，逐段求逆运动学、规划并复核 |
| `camera` | 场景视角；交错挡板场景采用俯视斜角展示完整通道 |

SO-101 的朝下抓取工作空间受到腕关节限位约束。默认场景的目标位于基座前方约 0.16～0.21 m、左右约 0.10 m，转移高度为 0.085 m。修改坐标后通过 `--check-only` 检查可达性与碰撞。

## 输出与结果

每次运行创建独立的 `outputs/so101/manipulation/<场景>-<编号>/`：

- `request.json`：本次配置快照。
- `trajectory.npz`：时间、五关节角、夹爪角、物体位置与四元数、动作阶段、末端路径。
- `result.json`：成功状态、放置误差、速度与加速度峰值、碰撞检查次数、依赖版本及源码校验值。
- `demo.gif` / `demo.png`：通过 `--export` 生成的动画与封面，黄色点线显示携物转移路径。
- `demo-grasp.png`：抬升过程的双侧夹持特写。
- `trajectory.png`：末端与物料三维路径、关节角时间曲线。

在 MuJoCo 3.17.0、NumPy 2.4.6 环境下，五个默认场景的物料最终中心均落在目标 1 mm 范围内；该误差衡量运动学任务落点，伺服跟踪误差需通过动力学控制实验另行测量。

`max_deviation_from_straight_line_m` 记录实际末端路径相对直线参考的偏离。关节空间绕行允许偏离直线，`planned_detours` 记录直线路径被阻断的动作、RRT 搜索节点与简化后路点。`grasps` 记录闭合角与双侧间隙。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/so101 -q
```

测试覆盖五个场景的阶段顺序、搬运过程双侧接触、偏离夹持区的抓取拒绝、夹持前静止、搬运抬升、释放后位置保持、关节速度与加速度约束、逐帧几何复核、不可达目标以及阻挡场景失败记录。

全仓库测试中的两项二连杆异常文本断言要求英文 `unreachable`，对应实现返回中文“目标位置不可达”。这两项断言与固定 SO-101 操作模块分开维护。

## 夹持修正记录

初版以原模型 `gripperframe` 加世界坐标高度偏移确定物块位置；在原位置检测到固定指穿入约 6.3 mm、活动指间隙约 11.7 mm。操作模块改用 `graspframe`：夹爪连杆局部坐标 `[0.0041, 0, -0.087]` m，位于两指有效夹持段之间。物块中心直接作为抓取目标，结合分段指部碰撞形状与双侧接触检查，避免把偏离夹持区的物块直接绑定到工具。

## 两种绕障场景的区别

- `detour`：单块挡板在 x=0.17～0.23 m，物块经 x=0.275 m 一侧通过，完成一次侧向绕行。
- `clutter`：第一块挡板在 y=-0.035 m、右端 x=0.225 m；第二块在 y=0.04 m、左端 x=0.20 m。物块先经 x=0.265 m 从第一块外侧通过，再移动到 x=0.15 m，从第二块内侧通过。

两个场景的转移高度设为 0.055 m。测试同时检查物块顶部低于 0.10 m 的挡板顶部、穿越挡板所在截面时处于正确侧边，以及原始起终关节直连被阻断。通道引导点由场景配置指定，段内使用现有逆运动学与关节规划模块。结果中的 `transfer_routes` 保存直连有效性及引导点，`planned_detours` 保存触发 RRT 的路段。
