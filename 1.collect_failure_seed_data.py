"""
针对失败 seed 补采专家演示数据
用于模型再训练
请在收集数据时，保持机器人在任务中执行，避免在任务完成后停止
注意：使用wsl2录制专家数据时，打开mujoco会很卡，建议在Ubuntu系统上录制
"""
import sys
import argparse
import random
import numpy as np
import os

# 在这里修改默认失败 seed；也可以通过命令行 --seeds 覆盖
DEFAULT_SEEDS = [5,6,12,13,16]

REPO_NAME = 'ur_pnp'
DEMOS_PER_SEED = 1  # 每个 seed 默认录制的成功演示次数
ROOT = "./demo_failure_seed_data"  # 补采数据默认保存到独立目录，避免覆盖原始数据

TASK_NAME = '将马克杯放在盘子上'
xml_path = './mode/demo_scene.xml'


def parse_seed_list(seed_text):
    """解析逗号分隔的 seed 列表，例如 3 或 3,7,11。"""
    try:
        seeds = [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds 只支持整数或逗号分隔的整数列表") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds 至少需要指定一个整数 seed")
    return seeds


def parse_args():
    """命令行参数用于临时覆盖代码里的默认补采配置。"""
    parser = argparse.ArgumentParser(description="按指定 seed 补采专家演示数据")
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=None,
        help="要补采的 seed，支持单个整数或逗号分隔列表，例如 3 或 3,7,11",
    )
    parser.add_argument(
        "--demos-per-seed",
        type=int,
        default=DEMOS_PER_SEED,
        help="每个 seed 录制的成功 episode 数，默认 1",
    )
    parser.add_argument(
        "--root",
        default=ROOT,
        help=f"数据集保存目录，默认 {ROOT}",
    )
    args = parser.parse_args()
    if args.demos_per_seed <= 0:
        parser.error("--demos-per-seed 必须大于 0")
    return args


args = parse_args()
TARGET_SEEDS = args.seeds if args.seeds is not None else DEFAULT_SEEDS
DEMOS_PER_SEED = args.demos_per_seed
ROOT = args.root

TOTAL_DEMOS = len(TARGET_SEEDS) * DEMOS_PER_SEED
print(f"补采 seed 列表: {TARGET_SEEDS}")
print(f"每个 seed 录制 {DEMOS_PER_SEED} 条成功 episode，总计 {TOTAL_DEMOS} 条")
print(f"数据保存目录: {ROOT}")

from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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

# 定义环境；当前 seed 完成指定条数后，再切换到下一个 seed
seed_index = 0
current_seed = TARGET_SEEDS[seed_index]
current_seed_episode_id = 0
PnPEnv = SimpleEnv(xml_path, seed=current_seed, state_type='joint_angle')

action = np.zeros(7)
episode_id = 0
record_flag = False  # 机器人开始移动时开始录制
while PnPEnv.env.is_viewer_alive() and episode_id < TOTAL_DEMOS:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查 episode 是否完成
        done = PnPEnv.check_success()
        if done:
            # 只有在真正开始录制了（人类操作了）的情况下，才保存数据
            if record_flag:
                dataset.save_episode()
                episode_id += 1
                current_seed_episode_id += 1
                print(
                    f"成功保存 seed={current_seed} 的第 {current_seed_episode_id}/{DEMOS_PER_SEED} 条 "
                    f"Episode，总进度 {episode_id}/{TOTAL_DEMOS}！"
                )
            else:
                # 触发了罕见的“幸运刷新”（开局即成功），这是一条废数据，直接跳过不保存
                print("幸运刷新！杯子已经在盘子上了，跳过保存，重新刷新环境...")

            if current_seed_episode_id >= DEMOS_PER_SEED and episode_id < TOTAL_DEMOS:
                # 当前 seed 达标后切换到下一个 seed，保证失败场景按清单逐个补采
                seed_index += 1
                current_seed = TARGET_SEEDS[seed_index]
                current_seed_episode_id = 0
                print(f"切换到下一个 seed: {current_seed}")
            
            # 重置环境和底层缓冲区
            PnPEnv.reset(seed=current_seed)
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
            PnPEnv.reset(seed=current_seed)
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
