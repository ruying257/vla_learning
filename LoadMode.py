import sys
import random
import numpy as np
import os
from mujoco_env.y_env import SimpleEnv


# 环境导入与初始化
xml_path = './mode/scene.xml'

# 创建环境实例
env = SimpleEnv(
    xml_path=xml_path,
    action_type='eef_pose',      # 随便选，不影响观察
    state_type='joint_angle',
    seed=None                     # 随机种子，None 表示完全随机
)

# 主循环
while env.env.is_viewer_alive():

    # 先抓取图像更新
    env.grab_image()
    # 渲染场景
    env.render(teleop=True)

# 清理
env.env.close_viewer()