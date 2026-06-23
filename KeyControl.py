import sys
import random
import numpy as np
import os
from mujoco_env.y_env import SimpleEnv

# 环境导入与初始化
xml_path = './mode/demo_scene.xml'

# 创建环境实例
env = SimpleEnv(
    xml_path=xml_path,
    action_type='eef_pose',
    state_type='joint_angle',
    seed=None
)

# 主循环
while env.env.is_viewer_alive():

    env.step_env()
    # 检测任务是否成功
    is_success = env.check_success()
    if is_success:
        env.reset(seed=None)
        continue
    # 读取键盘输入
    action, reset = env.teleop_robot()
    if reset:
        env.reset(seed=None)
        continue
    # 将键盘给出的位移指令传给机器人
    env.step(action)
    # 先抓取图像更新
    env.grab_image()
    # 渲染场景
    env.render(teleop=True)
# 清理
env.env.close_viewer()