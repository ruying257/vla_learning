"""
收集演示数据
用于训练机器人执行任务的演示数据
请在收集数据时，保持机器人在任务中执行，避免在任务完成后停止
注意：使用wsl2录制专家数据时，打开mujoco会很卡
"""
import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 如果要随机化物体位置，请将此设置为 None
# 如果固定 seed，物体位置每次都将相同
SEED = 0
SEED = None  # <- 取消注释此行以随机化物体位置

REPO_NAME = 'ur_pnp'
NUM_DEMO = 50  # 要收集的演示次数
ROOT = "./demo_data"  # 保存演示数据的根目录

TASK_NAME = '将马克杯放在盘子上'
xml_path = './mode/demo_scene.xml'

create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False

if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT,
                robot_type="ur",
                fps=20,  # 每秒20帧
                features={
                    "observation.image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"],  # x, y, z, roll, pitch, yaw（位置和姿态）
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"],  # 6个关节角度和1个夹爪控制
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"],  # 物体的初始位置，训练时不使用
                    },
                },
                image_writer_threads=10,    # 图像写入线程数
                image_writer_processes=0,   # 图像写入进程数
        )
else:
    print("从之前的数据集加载")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# 定义环境
PnPEnv = SimpleEnv(xml_path, seed=SEED, state_type='joint_angle')

action = np.zeros(7)
episode_id = 0
record_flag = False  # 机器人开始移动时开始录制
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查 episode 是否完成
        done = PnPEnv.check_success()
        if done:
            # 只有在真正开始录制了（人类操作了）的情况下，才保存数据
            if record_flag:
                dataset.save_episode()
                episode_id += 1
                print(f"成功保存第 {episode_id} 个 Episode！")
            else:
                # 触发了罕见的“幸运刷新”（开局即成功），这是一条废数据，直接跳过不保存
                print("幸运刷新！杯子已经在盘子上了，跳过保存，重新刷新环境...")
            
            # 重置环境和底层缓冲区
            PnPEnv.reset(seed=SEED)
            dataset.clear_episode_buffer() # 清理掉可能残留的空帧
            
            # 务必关闭录制标志，等待人类下一步操作
            record_flag = False
        # 遥控机器人并获取带夹爪的末端执行器增量位姿
        action, reset = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("开始录制")
        if reset:
            # 重置环境并清理 episode 缓冲区
            # 可以通过按 'z' 键完成
            PnPEnv.reset(seed=SEED)
            # PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # 执行环境步进
        # 获取末端执行器位姿和图像
        ee_pose = PnPEnv.get_ee_pose()
        agent_image, wrist_image = PnPEnv.grab_image()
        # # 调整大小到 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        if record_flag:
            # 将帧添加到数据集
            dataset.add_frame({
                "observation.image": agent_image,
                "observation.wrist_image": wrist_image,
                "observation.state": ee_pose,
                "action": joint_q,
                "obj_init": PnPEnv.obj_init_pose,
                "task": TASK_NAME,
            })
        PnPEnv.render(teleop=True)

PnPEnv.env.close_viewer()
dataset.finalize()
# 清理 images 文件夹
import shutil
shutil.rmtree(dataset.root / 'images')