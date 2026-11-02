# SO-101 模型资产

已下载官方模型及其资源，版本和许可证见 manifest.json；模型已通过 MuJoCo 加载检查，关节运动和碰撞尚待核对。

来源：https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

模型引入时必须：

- 保留 URDF/MJCF 与所需网格的相对路径及许可证。
- 在 manifest.json 中记录真实来源提交、源文件路径和本地入口。
- 区分官方原始模型与为 MuJoCo 转换或简化的模型，记录修改。
- 验证关节名称、轴向、零位、限位、夹爪与碰撞几何。
- 不把 URDF 可显示直接视为动力学与碰撞模型已经正确。

manifest.json 中 null 代表待确认；任何加载程序都必须拒绝未配置的模型路径，
不能默认为二连杆模型或连接真机。
