# VLA 机器人学习项目

基于视觉-语言-动作（Vision-Language-Action）模型的机器人操作学习项目，使用 MuJoCo 物理仿真环境和 LeRobot 库。

## 项目简介

这是一个完整的机器人模仿学习项目，提供从数据收集、数据可视化、模型训练到策略部署的全流程示例。项目在 UR10e 机械臂仿真环境中完成杯子放置到盘子上的拾取放置任务。

## 功能特性

- **演示数据收集** - 键盘遥操作收集机器人操作数据
- **数据可视化** - 回放和可视化已收集的演示数据
- **模型训练** - 使用 ACT（Action Chunking with Transformer）模型进行训练
- **策略部署** - 在仿真环境中部署训练好的策略

## 技术栈

| 技术 | 用途 |
|-----|------|
| Python | 主要编程语言 |
| MuJoCo | 物理仿真环境 |
| PyTorch | 深度学习框架 |
| LeRobot | 机器人学习库 |
| NumPy | 数值计算库 |
| PIL | 图像处理库 |

## 项目结构

```
vla/
├── 1.collect_data.py          # 数据收集脚本
├── 2.visualize_data.py        # 数据可视化脚本
├── 3.train.py                 # 模型训练脚本
├── 4.deploy.py                # 策略部署脚本
├── KeyControl.py              # 键盘控制模块
├── LoadMode.py                # 模型加载模块
├── OpenDemo.py                # 演示模块
├── README.md                  # 项目说明文档
├── .gitignore                 # Git 忽略文件
├── ckpt/                      # 模型检查点目录
│   └── act_y/                 # ACT 模型检查点
├── demo_data/                 # 演示数据目录（运行时生成）
├── mode/                      # MuJoCo 仿真模型
│   ├── mug_5/                 # 杯子模型
│   ├── plate_11/              # 盘子模型
│   ├── universal_robots_ur10e/ # UR10e 机械臂模型
│   ├── robotiq_2f85/          # Robotiq 夹爪模型
│   ├── realsense_d435i/       # RealSense 相机模型
│   ├── tabletop/              # 桌面和物体模型
│   └── demo_scene.xml         # 主场景配置文件
└── mujoco_env/                # MuJoCo 环境模块
    ├── __init__.py
    ├── y_env.py               # 主要环境类
    ├── y_env2.py
    ├── ik.py                  # 逆运动学求解
    ├── mujoco_parser.py       # MuJoCo 解析器
    ├── transforms.py          # 坐标变换
    └── utils.py               # 工具函数
```

## 安装说明

### 环境要求

- Python 3.10 或更高版本
- CUDA（如需 GPU 加速训练）

### 依赖安装

```bash
# 使用 requirements.txt 安装所有依赖
pip install -r requirements.txt
```

或者单独安装：
```bash
pip install numpy torch==2.6.0 pillow mujoco==3.6.0 lerobot==0.4.4 transformers
```

## 使用说明

项目按照以下四个步骤顺序执行：

### 1. 收集演示数据

```bash
python 1.collect_data.py
```

使用键盘控制机器人完成任务，数据将保存到 `./demo_data` 目录。

**配置参数**（在 1.collect_data.py 中修改）：
- `SEED = 0`：固定种子，物体位置每次相同
- `SEED = None`：随机种子，物体位置每次不同
- `NUM_DEMO = 10`：收集演示的次数

**控制按键：**
| 按键 | 功能 |
|------|------|
| W/S | 前后移动 |
| A/D | 左右移动 |
| R/F | 上下移动 |
| Q/E | 左右倾斜 |
| 方向键 ↑↓←→ | 旋转 |
| 空格键 | 夹爪开合 |
| Z | 重置环境 |

### 2. 可视化数据

```bash
python 2.visualize_data.py
```

回放已收集的演示数据，查看数据质量。

### 3. 训练模型

```bash
python 3.train.py
```

训练 ACT 模型，模型检查点保存到 `./ckpt/act_y`。

### 4. 部署策略

```bash
python 4.deploy.py
```

加载训练好的模型，在仿真环境中自主运行。

## 仿真环境配置

项目使用以下机器人和场景配置：

- **机械臂** - Universal Robots UR10e
- **夹爪** - Robotiq 2F-85
- **相机** - RealSense D435i
  - agentview：第三方视角
  - d435i_rgb：手腕视角
  - topview：顶视图
  - sideview：侧视图
- **任务** - 将杯子放到盘子上（Put mug cup on the plate）
- **场景** - 木质工作台面

## 物体位置配置

杯子和盘子的初始位置在 `mujoco_env/y_env.py` 的 `reset()` 方法中配置：

- X 轴范围：[0.24, 0.4]
- Y 轴范围：[-0.4, 0.2]
- Z 轴高度：0.83（固定）
- 物体间最小距离：0.15m

## 任务成功条件

在 `mujoco_env/y_env.py` 的 `check_success()` 方法中定义：
1. 杯子在盘子上方（XY 距离 < 0.1m）
2. 杯子高度接近盘子
3. 夹爪打开
4. 机械臂末端上升到 0.9m 以上

