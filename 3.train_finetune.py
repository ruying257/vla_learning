"""
基于已有 ACT checkpoint 继续训练。
命令示例：
  ACT_RESUME_CKPT_DIR=ckpt/v5 \
  ACT_DATASET_ROOT=failure_seed_data \
  ACT_CKPT_DIR=ckpt/v5_finetune_new_data \
  ACT_LR=1e-5 \
  python 3.train_finetune.py
"""
import json
import os
import time
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.modeling_act import ACTPolicy
from torchvision import transforms

try:
    import wandb
except ImportError:
    wandb = None


# 续训脚本必须显式指定输入 checkpoint，避免误以为普通训练就是 resume。
DATASET_ROOT = os.environ.get("ACT_DATASET_ROOT", "failure_seed_data")    # 新训练数据集根目录
RESUME_CKPT_DIR = os.environ.get("ACT_RESUME_CKPT_DIR", "")                             # 已训练好的 checkpoint 目录
CKPT_DIR = os.environ.get("ACT_CKPT_DIR", "./ckpt/v5_finetune_new_data")                      # 续训输出目录
TRAINING_STEPS = int(os.environ.get("ACT_TRAINING_STEPS", "1000"))                      # 续训步数
LOG_FREQ = int(os.environ.get("ACT_LOG_FREQ", "10"))                                   # 日志记录频率
BATCH_SIZE = int(os.environ.get("ACT_BATCH_SIZE", "64"))                                # 批次大小
NUM_WORKERS = int(os.environ.get("ACT_NUM_WORKERS", "4"))                               # 数据加载器工作线程数
LEARNING_RATE = float(os.environ.get("ACT_LR", "1e-5"))                                 # 续训默认用更小学习率
SHOW_PLOT = os.environ.get("ACT_SHOW_PLOT", "0") == "1"                                 # 是否显示训练过程中的可视化
METRICS_PATH = os.environ.get("ACT_METRICS_PATH", "")                                   # 训练指标保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   # 训练设备

# 续训默认记录 WandB；如需本地离线跑通，可显式设置 ACT_USE_WANDB=0 关闭。
USE_WANDB = os.environ.get("ACT_USE_WANDB", "1") == "1"
if wandb is not None and not USE_WANDB:
    wandb = None


class AddGaussianNoise:
    """给图像张量加入轻量高斯噪声，用于提升视觉输入的鲁棒性。"""

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 噪声与输入保持相同形状，随后再裁剪到合法像素范围。
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class EpisodeSampler(torch.utils.data.Sampler):
    """只抽取一个 episode，便于训练后快速计算动作误差。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
        to_idx = int(dataset.meta.episodes[episode_index]["dataset_to_index"])
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def move_tensors_to_device(batch):
    """只迁移 tensor 字段，保留 task 等非张量字段。"""
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def ensure_resume_paths():
    """检查续训输入和输出目录，避免误覆盖源 checkpoint。"""
    if not RESUME_CKPT_DIR:
        raise SystemExit("Please set ACT_RESUME_CKPT_DIR to an existing ACT checkpoint directory.")

    resume_path = Path(RESUME_CKPT_DIR).expanduser()
    output_path = Path(CKPT_DIR).expanduser()
    if not resume_path.exists():
        raise SystemExit(f"Resume checkpoint missing: {resume_path}")

    if resume_path.resolve() == output_path.resolve():
        raise SystemExit(
            "ACT_CKPT_DIR is the same as ACT_RESUME_CKPT_DIR. "
            "Use a new output dir to keep the source checkpoint unchanged."
        )


def feature_signature(features):
    """提取特征 key 和 shape，用于确认新数据集能匹配旧 checkpoint。"""
    signature = {}
    for key, feature in features.items():
        shape = getattr(feature, "shape", None)
        signature[key] = tuple(shape) if shape is not None else str(feature)
    return signature


def validate_dataset_compatible(policy, dataset_features):
    """新数据集的输入/输出字段必须和 checkpoint 配置一致。"""
    output_features = {key: ft for key, ft in dataset_features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in dataset_features.items() if key not in output_features}

    expected_input = feature_signature(policy.config.input_features)
    actual_input = feature_signature(input_features)
    expected_output = feature_signature(policy.config.output_features)
    actual_output = feature_signature(output_features)

    if expected_input != actual_input or expected_output != actual_output:
        raise SystemExit(
            "New dataset features do not match the resume checkpoint.\n"
            f"expected_input={expected_input}\n"
            f"actual_input={actual_input}\n"
            f"expected_output={expected_output}\n"
            f"actual_output={actual_output}"
        )


def main():
    ensure_resume_paths()

    print(f"device: {DEVICE}")
    print(f"dataset_root: {DATASET_ROOT}")
    print(f"resume_ckpt_dir: {RESUME_CKPT_DIR}")
    print(f"ckpt_dir: {CKPT_DIR}")

    # 先读取新数据集元数据，用于构建 dataloader 和检查 checkpoint 兼容性。
    dataset_metadata = LeRobotDataset("ur_pnp", root=DATASET_ROOT, video_backend="pyav")
    features = dataset_to_policy_features(dataset_metadata.features)

    # 从已有 checkpoint 加载模型权重和原始 ACT 配置。
    policy = ACTPolicy.from_pretrained(RESUME_CKPT_DIR, local_files_only=True, strict=True)
    validate_dataset_compatible(policy, features)
    policy.train()
    policy.to(DEVICE)

    cfg = policy.config
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    if wandb is not None:
        wandb.init(
            project="ur_pnp",
            name="act_finetune",
            config={
                "dataset_root": DATASET_ROOT,
                "resume_ckpt_dir": RESUME_CKPT_DIR,
                "ckpt_dir": CKPT_DIR,
                "use_wandb": USE_WANDB,
                "training_steps": TRAINING_STEPS,
                "log_freq": LOG_FREQ,
                "chunk_size": cfg.chunk_size,
                "n_action_steps": cfg.n_action_steps,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
        )

    transform = transforms.Compose(
        [
            AddGaussianNoise(mean=0.0, std=0.02),
            transforms.Lambda(lambda x: x.clamp(0, 1)),
        ]
    )

    dataset = LeRobotDataset(
        "ur_pnp",
        delta_timestamps=delta_timestamps,
        root=DATASET_ROOT,
        image_transforms=transform,
        video_backend="pyav",
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type != "cpu",
        drop_last=False,
    )

    step = 0
    done = False
    last_loss = None
    start_time = time.time()
    while not done:
        for batch in dataloader:
            inp_batch = move_tensors_to_device(batch)
            loss, _ = policy.forward(inp_batch)
            last_loss = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
                policy.save_pretrained(CKPT_DIR)
                if wandb is not None:
                    wandb.log({"loss": loss.item(), "step": step})

            step += 1
            if step >= TRAINING_STEPS:
                done = True
                break

    total_time = time.time() - start_time
    print(f"total_training_time_sec: {total_time:.2f}")
    print(f"total_training_time_min: {total_time / 60:.2f}")
    print(f"final_loss: {last_loss:.3f}")
    policy.save_pretrained(CKPT_DIR)

    policy.eval()
    actions = []
    gt_actions = []
    episode_sampler = EpisodeSampler(dataset, episode_index=0)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=1,
        shuffle=False,
        pin_memory=DEVICE.type != "cpu",
        sampler=episode_sampler,
    )
    policy.reset()
    for batch in test_dataloader:
        inp_batch = move_tensors_to_device(batch)
        action = policy.select_action(inp_batch)
        actions.append(action)
        gt_actions.append(inp_batch["action"][:, 0, :])

    actions = torch.cat(actions, dim=0)
    gt_actions = torch.cat(gt_actions, dim=0)
    mean_action_error = torch.mean(torch.abs(actions - gt_actions)).item()
    print(f"Mean action error: {mean_action_error:.3f}")

    if METRICS_PATH:
        # 将续训来源和输出目录都写入指标，便于后续追踪模型来源。
        metrics = {
            "dataset_root": DATASET_ROOT,
            "resume_ckpt_dir": RESUME_CKPT_DIR,
            "ckpt_dir": CKPT_DIR,
            "use_wandb": USE_WANDB,
            "training_steps": TRAINING_STEPS,
            "log_freq": LOG_FREQ,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "chunk_size": cfg.chunk_size,
            "n_action_steps": cfg.n_action_steps,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE),
            "final_loss": last_loss,
            "mean_action_error": mean_action_error,
            "total_training_time_sec": total_time,
        }
        metrics_dir = os.path.dirname(METRICS_PATH)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if wandb is not None:
        wandb.log({"mean_action_error": mean_action_error, "total_training_time": total_time})
        wandb.finish()

    if SHOW_PLOT:
        import matplotlib.pyplot as plt

        action_dim = gt_actions.shape[-1]
        fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))
        for i in range(action_dim):
            axs[i].plot(actions[:, i].cpu().detach().numpy(), label="pred")
            axs[i].plot(gt_actions[:, i].cpu().detach().numpy(), label="gt")
            axs[i].legend()
        plt.show()


if __name__ == "__main__":
    main()
