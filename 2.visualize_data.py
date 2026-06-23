"""
回放演示数据
"""
import os

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_env.y_env import SimpleEnv


# 默认回放仓库内已验证可读的数据集；需要回放新采集数据时设置 ACT_DATASET_ROOT=./demo_data。
DATASET_ROOT = os.environ.get("ACT_DATASET_ROOT", "datasets/demo_v5_30demos_random")
EPISODE_INDEX = int(os.environ.get("ACT_EPISODE_INDEX", "0"))
MAX_STEPS = int(os.environ.get("ACT_VIS_MAX_STEPS", "0"))


class EpisodeSampler(torch.utils.data.Sampler):
    """只回放一个 episode，方便检查演示质量。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
        to_idx = int(dataset.meta.episodes[episode_index]["dataset_to_index"])
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def main():
    print(f"dataset_root: {DATASET_ROOT}")
    dataset = LeRobotDataset("ur_pnp", root=DATASET_ROOT, video_backend="pyav")
    episode_sampler = EpisodeSampler(dataset, EPISODE_INDEX)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=episode_sampler,
    )

    env = SimpleEnv("./mode/demo_scene.xml", action_type="joint_angle")
    env.reset()
    iter_dataloader = iter(dataloader)
    step = 0

    while env.env.is_viewer_alive():
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue

        try:
            data = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            env.reset()
            step = 0
            continue

        if step == 0:
            # 使用数据集中记录的初始物体位姿，保证回放场景和采集场景一致。
            env.set_obj_pose(data["obj_init"][0, :3], data["obj_init"][0, 3:])

        action = data["action"].numpy()
        env.step(action[0])

        env.rgb_agent = (data["observation.image"][0].numpy() * 255).astype(np.uint8)
        env.rgb_ego = (data["observation.wrist_image"][0].numpy() * 255).astype(np.uint8)
        env.rgb_agent = np.transpose(env.rgb_agent, (1, 2, 0))
        env.rgb_ego = np.transpose(env.rgb_ego, (1, 2, 0))
        env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        env.render()

        step += 1
        if MAX_STEPS and step >= MAX_STEPS:
            print(f"Reached ACT_VIS_MAX_STEPS={MAX_STEPS}")
            break

    env.env.close_viewer()


if __name__ == "__main__":
    main()
