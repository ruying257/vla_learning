"""
基于已有 ACT checkpoint，使用“旧数据回放 + 补充数据”进行全参数微调。

脚本默认在每个 batch 中采样 80% 旧数据和 20% 补充数据，用旧任务分布抑制
灾难性遗忘，同时让模型学习失败 seed 对应的补充示范。训练完成 250、500、
750 和 1000 次参数更新后分别保存 checkpoint，并在两套数据的首个 episode 上
分别计算动作平均绝对误差（MAE）。

运行示例：
  ACT_RESUME_CKPT_DIR=ckpt/v5 \
  ACT_DATASET_ROOT=datasets/failure_seed_data \
  ACT_REPLAY_DATASET_ROOT=datasets/demo_v5_30demos_random \
  ACT_REPLAY_RATIO=0.8 \
  ACT_CKPT_DIR=ckpt/v5_finetune_replay \
  ACT_TRAINING_STEPS=1000 \
  ACT_SAVE_STEPS=250,500,750,1000 \
  ACT_LR=1e-5 \
  python 3.train_finetune_replay.py

输出结构：
  ckpt/v5_finetune_replay/
  ├── step_0250/                 # 完成 250 次参数更新后的模型
  ├── step_0500/
  ├── step_0750/
  ├── step_1000/
  ├── config.json                # 最终 1000-step 模型，便于直接部署
  ├── model.safetensors
  └── training_metrics.json
"""
import json
import os
import random
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


# 训练参数集中放在脚本前方，既可直接修改默认值，也可通过环境变量覆盖。
DATASET_ROOT = os.environ.get("ACT_DATASET_ROOT", "datasets/failure_seed_data")
REPLAY_DATASET_ROOT = os.environ.get(
    "ACT_REPLAY_DATASET_ROOT", "datasets/demo_v5_30demos_random"
)
RESUME_CKPT_DIR = os.environ.get("ACT_RESUME_CKPT_DIR", "")
CKPT_DIR = os.environ.get("ACT_CKPT_DIR", "./ckpt/v5_finetune_replay")
TRAINING_STEPS = int(os.environ.get("ACT_TRAINING_STEPS", "1000"))
SAVE_STEPS_TEXT = os.environ.get("ACT_SAVE_STEPS", "250,500,750,1000")
REPLAY_RATIO = float(os.environ.get("ACT_REPLAY_RATIO", "0.8"))
LOG_FREQ = int(os.environ.get("ACT_LOG_FREQ", "10"))
BATCH_SIZE = int(os.environ.get("ACT_BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("ACT_NUM_WORKERS", "4"))
LEARNING_RATE = float(os.environ.get("ACT_LR", "1e-5"))
TRAIN_SEED = int(os.environ.get("ACT_TRAIN_SEED", "0"))
SHOW_PLOT = os.environ.get("ACT_SHOW_PLOT", "0") == "1"
METRICS_PATH = os.environ.get(
    "ACT_METRICS_PATH", str(Path(CKPT_DIR) / "training_metrics.json")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 微调默认记录 WandB；本地 smoke test 可设置 ACT_USE_WANDB=0。
USE_WANDB = os.environ.get("ACT_USE_WANDB", "1") == "1"
if wandb is not None and not USE_WANDB:
    wandb = None


class AddGaussianNoise:
    """给图像张量加入轻量高斯噪声，用于提升视觉输入的鲁棒性。"""

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 噪声与输入保持相同形状，随后由下游变换裁剪到合法像素范围。
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class EpisodeSampler(torch.utils.data.Sampler):
    """只抽取指定 episode，用于保存点的快速离线评估。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
        to_idx = int(dataset.meta.episodes[episode_index]["dataset_to_index"])
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


class ReplayBatchSampler(torch.utils.data.Sampler):
    """按固定来源数量构造 batch，避免按数据集自然大小稀释补充数据。"""

    def __init__(
        self,
        replay_size: int,
        supplemental_size: int,
        batch_size: int,
        replay_ratio: float,
        num_batches: int,
        seed: int,
    ):
        self.replay_size = replay_size
        self.supplemental_size = supplemental_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed

        # 每批至少保留一个旧样本和一个补充样本。
        replay_count = round(batch_size * replay_ratio)
        self.replay_count = min(max(replay_count, 1), batch_size - 1)
        self.supplemental_count = batch_size - self.replay_count

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        for _ in range(self.num_batches):
            replay_indices = torch.randint(
                self.replay_size,
                (self.replay_count,),
                generator=generator,
            )
            supplemental_indices = torch.randint(
                self.supplemental_size,
                (self.supplemental_count,),
                generator=generator,
            )
            # ConcatDataset 中补充数据的索引位于旧数据之后。
            supplemental_indices += self.replay_size
            batch_indices = torch.cat([replay_indices, supplemental_indices])
            order = torch.randperm(self.batch_size, generator=generator)
            yield batch_indices[order].tolist()

    def __len__(self):
        return self.num_batches


def parse_save_steps(value: str):
    """解析并规范化逗号分隔的保存步数。"""
    try:
        save_steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise SystemExit(f"ACT_SAVE_STEPS must be comma-separated integers: {value}") from exc
    if not save_steps or any(step <= 0 for step in save_steps):
        raise SystemExit("ACT_SAVE_STEPS must contain positive integers.")
    return save_steps


def move_tensors_to_device(batch):
    """只迁移 tensor 字段，保留 task 等非张量字段。"""
    return {key: (value.to(DEVICE) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def ensure_runtime_config(save_steps):
    """检查训练路径和关键参数，避免覆盖源 checkpoint 或产生无效采样。"""
    if not RESUME_CKPT_DIR:
        raise SystemExit("Please set ACT_RESUME_CKPT_DIR to an existing ACT checkpoint directory.")

    resume_path = Path(RESUME_CKPT_DIR).expanduser()
    output_path = Path(CKPT_DIR).expanduser()
    if not resume_path.exists():
        raise SystemExit(f"Resume checkpoint missing: {resume_path}")
    if not Path(DATASET_ROOT).expanduser().exists():
        raise SystemExit(f"Supplemental dataset missing: {DATASET_ROOT}")
    if not Path(REPLAY_DATASET_ROOT).expanduser().exists():
        raise SystemExit(f"Replay dataset missing: {REPLAY_DATASET_ROOT}")
    if resume_path.resolve() == output_path.resolve():
        raise SystemExit(
            "ACT_CKPT_DIR is the same as ACT_RESUME_CKPT_DIR. "
            "Use a new output dir to keep the source checkpoint unchanged."
        )
    if not 0.0 < REPLAY_RATIO < 1.0:
        raise SystemExit("ACT_REPLAY_RATIO must be between 0 and 1.")
    if BATCH_SIZE < 2:
        raise SystemExit("ACT_BATCH_SIZE must be at least 2 for replay training.")
    if TRAINING_STEPS <= 0:
        raise SystemExit("ACT_TRAINING_STEPS must be positive.")
    if save_steps[-1] > TRAINING_STEPS:
        raise SystemExit("ACT_SAVE_STEPS cannot contain a step larger than ACT_TRAINING_STEPS.")
    if TRAINING_STEPS not in save_steps:
        raise SystemExit("ACT_SAVE_STEPS must include ACT_TRAINING_STEPS.")


def feature_signature(features):
    """提取特征 key 和 shape，用于检查数据集与 checkpoint 是否兼容。"""
    signature = {}
    for key, feature in features.items():
        shape = getattr(feature, "shape", None)
        signature[key] = tuple(shape) if shape is not None else str(feature)
    return signature


def validate_dataset_compatible(policy, dataset_features, dataset_name):
    """输入和输出字段必须与 checkpoint 配置一致。"""
    output_features = {
        key: feature
        for key, feature in dataset_features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in dataset_features.items() if key not in output_features
    }

    expected_input = feature_signature(policy.config.input_features)
    actual_input = feature_signature(input_features)
    expected_output = feature_signature(policy.config.output_features)
    actual_output = feature_signature(output_features)
    if expected_input != actual_input or expected_output != actual_output:
        raise SystemExit(
            f"{dataset_name} features do not match the resume checkpoint.\n"
            f"expected_input={expected_input}\n"
            f"actual_input={actual_input}\n"
            f"expected_output={expected_output}\n"
            f"actual_output={actual_output}"
        )


def evaluate_first_episode(policy, dataset):
    """在无增强的首个完整 episode 上计算动作平均绝对误差。"""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=1,
        shuffle=False,
        pin_memory=DEVICE.type != "cpu",
        sampler=EpisodeSampler(dataset, episode_index=0),
    )
    actions = []
    gt_actions = []
    policy.eval()
    policy.reset()
    with torch.no_grad():
        for batch in dataloader:
            inp_batch = move_tensors_to_device(batch)
            actions.append(policy.select_action(inp_batch))
            gt_actions.append(inp_batch["action"][:, 0, :])

    actions = torch.cat(actions, dim=0)
    gt_actions = torch.cat(gt_actions, dim=0)
    mean_action_error = torch.mean(torch.abs(actions - gt_actions)).item()
    policy.reset()
    policy.train()
    return mean_action_error, actions, gt_actions


def write_metrics(metrics):
    """增量写入指标，保证中途停止时已完成保存点的数据仍可追踪。"""
    metrics_path = Path(METRICS_PATH).expanduser()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def main():
    save_steps = parse_save_steps(SAVE_STEPS_TEXT)
    ensure_runtime_config(save_steps)
    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_SEED)

    print(f"device: {DEVICE}")
    print(f"replay_dataset_root: {REPLAY_DATASET_ROOT}")
    print(f"supplemental_dataset_root: {DATASET_ROOT}")
    print(f"resume_ckpt_dir: {RESUME_CKPT_DIR}")
    print(f"ckpt_dir: {CKPT_DIR}")

    # 先读取两套元数据，再统一按照 checkpoint 配置生成时间窗口。
    replay_metadata = LeRobotDataset(
        "ur_pnp", root=REPLAY_DATASET_ROOT, video_backend="pyav"
    )
    supplemental_metadata = LeRobotDataset(
        "ur_pnp", root=DATASET_ROOT, video_backend="pyav"
    )
    replay_features = dataset_to_policy_features(replay_metadata.features)
    supplemental_features = dataset_to_policy_features(supplemental_metadata.features)

    policy = ACTPolicy.from_pretrained(
        RESUME_CKPT_DIR, local_files_only=True, strict=True
    )
    validate_dataset_compatible(policy, replay_features, "Replay dataset")
    validate_dataset_compatible(policy, supplemental_features, "Supplemental dataset")
    policy.train()
    policy.to(DEVICE)

    cfg = policy.config
    replay_delta_timestamps = resolve_delta_timestamps(cfg, replay_metadata)
    supplemental_delta_timestamps = resolve_delta_timestamps(cfg, supplemental_metadata)
    transform = transforms.Compose(
        [
            AddGaussianNoise(mean=0.0, std=0.02),
            transforms.Lambda(lambda tensor: tensor.clamp(0, 1)),
        ]
    )

    replay_dataset = LeRobotDataset(
        "ur_pnp",
        delta_timestamps=replay_delta_timestamps,
        root=REPLAY_DATASET_ROOT,
        image_transforms=transform,
        video_backend="pyav",
    )
    supplemental_dataset = LeRobotDataset(
        "ur_pnp",
        delta_timestamps=supplemental_delta_timestamps,
        root=DATASET_ROOT,
        image_transforms=transform,
        video_backend="pyav",
    )
    # 评估数据集不添加随机图像噪声，保证四个保存点可直接比较。
    replay_eval_dataset = LeRobotDataset(
        "ur_pnp",
        delta_timestamps=replay_delta_timestamps,
        root=REPLAY_DATASET_ROOT,
        video_backend="pyav",
    )
    supplemental_eval_dataset = LeRobotDataset(
        "ur_pnp",
        delta_timestamps=supplemental_delta_timestamps,
        root=DATASET_ROOT,
        video_backend="pyav",
    )

    combined_dataset = torch.utils.data.ConcatDataset(
        [replay_dataset, supplemental_dataset]
    )
    batch_sampler = ReplayBatchSampler(
        replay_size=len(replay_dataset),
        supplemental_size=len(supplemental_dataset),
        batch_size=BATCH_SIZE,
        replay_ratio=REPLAY_RATIO,
        num_batches=TRAINING_STEPS,
        seed=TRAIN_SEED,
    )
    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        num_workers=NUM_WORKERS,
        batch_sampler=batch_sampler,
        pin_memory=DEVICE.type != "cpu",
    )

    actual_replay_ratio = batch_sampler.replay_count / BATCH_SIZE
    print(
        "batch_sources: "
        f"replay={batch_sampler.replay_count}, "
        f"supplemental={batch_sampler.supplemental_count}, "
        f"actual_replay_ratio={actual_replay_ratio:.6f}"
    )

    if wandb is not None:
        wandb.init(
            project="ur_pnp",
            name="act_finetune_replay",
            config={
                "replay_dataset_root": REPLAY_DATASET_ROOT,
                "supplemental_dataset_root": DATASET_ROOT,
                "resume_ckpt_dir": RESUME_CKPT_DIR,
                "ckpt_dir": CKPT_DIR,
                "training_steps": TRAINING_STEPS,
                "save_steps": save_steps,
                "target_replay_ratio": REPLAY_RATIO,
                "actual_replay_ratio": actual_replay_ratio,
                "replay_samples_per_batch": batch_sampler.replay_count,
                "supplemental_samples_per_batch": batch_sampler.supplemental_count,
                "chunk_size": cfg.chunk_size,
                "n_action_steps": cfg.n_action_steps,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "train_seed": TRAIN_SEED,
            },
        )

    metrics = {
        "replay_dataset_root": REPLAY_DATASET_ROOT,
        "supplemental_dataset_root": DATASET_ROOT,
        "resume_ckpt_dir": RESUME_CKPT_DIR,
        "ckpt_dir": CKPT_DIR,
        "training_steps": TRAINING_STEPS,
        "save_steps": save_steps,
        "target_replay_ratio": REPLAY_RATIO,
        "actual_replay_ratio": actual_replay_ratio,
        "replay_samples_per_batch": batch_sampler.replay_count,
        "supplemental_samples_per_batch": batch_sampler.supplemental_count,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "chunk_size": cfg.chunk_size,
        "n_action_steps": cfg.n_action_steps,
        "learning_rate": LEARNING_RATE,
        "train_seed": TRAIN_SEED,
        "device": str(DEVICE),
        "checkpoints": [],
    }

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    last_loss = None
    start_time = time.time()
    for step, batch in enumerate(dataloader, start=1):
        inp_batch = move_tensors_to_device(batch)
        loss, _ = policy.forward(inp_batch)
        last_loss = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step == 1 or step % LOG_FREQ == 0:
            print(f"step: {step} loss: {last_loss:.3f}")
            if wandb is not None:
                wandb.log({"loss": last_loss, "step": step})

        if step in save_steps:
            checkpoint_path = Path(CKPT_DIR) / f"step_{step:04d}"
            policy.save_pretrained(checkpoint_path)
            replay_error, _, _ = evaluate_first_episode(policy, replay_eval_dataset)
            supplemental_error, _, _ = evaluate_first_episode(
                policy, supplemental_eval_dataset
            )
            checkpoint_metrics = {
                "step": step,
                "checkpoint_path": str(checkpoint_path),
                "loss": last_loss,
                "replay_mean_action_error": replay_error,
                "supplemental_mean_action_error": supplemental_error,
            }
            metrics["checkpoints"].append(checkpoint_metrics)
            metrics["elapsed_time_sec"] = time.time() - start_time
            write_metrics(metrics)
            print(
                f"saved: {checkpoint_path} "
                f"replay_mae={replay_error:.3f} "
                f"supplemental_mae={supplemental_error:.3f}"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "step": step,
                        "replay_mean_action_error": replay_error,
                        "supplemental_mean_action_error": supplemental_error,
                    }
                )

    total_time = time.time() - start_time
    # 根目录额外保存最终模型，使 deploy.py 可直接读取 CKPT_DIR。
    policy.save_pretrained(CKPT_DIR)
    metrics["final_loss"] = last_loss
    metrics["total_training_time_sec"] = total_time
    metrics["final_checkpoint_path"] = CKPT_DIR
    write_metrics(metrics)
    print(f"total_training_time_sec: {total_time:.2f}")
    print(f"total_training_time_min: {total_time / 60:.2f}")
    print(f"final_loss: {last_loss:.3f}")

    if SHOW_PLOT:
        import matplotlib.pyplot as plt

        _, actions, gt_actions = evaluate_first_episode(
            policy, supplemental_eval_dataset
        )
        action_dim = gt_actions.shape[-1]
        fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))
        for index in range(action_dim):
            axs[index].plot(actions[:, index].cpu().numpy(), label="pred")
            axs[index].plot(gt_actions[:, index].cpu().numpy(), label="gt")
            axs[index].legend()
        plt.show()

    if wandb is not None:
        wandb.log({"final_loss": last_loss, "total_training_time": total_time})
        wandb.finish()


if __name__ == "__main__":
    main()
