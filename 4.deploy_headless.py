import json
import os
import traceback
from datetime import datetime

# WSL2 headless 部署默认使用 OSMesa，避免走 WSLg 的可见窗口渲染链路。
os.environ.setdefault("MUJOCO_GL", "osmesa")

import cv2
import torch
import torchvision
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from PIL import Image

from mujoco_env.y_env import SimpleEnv


DATASET_ROOT = os.environ.get("ACT_DATASET_ROOT", "datasets/demo_v5_30demos_random")
CKPT_DIR = os.environ.get("ACT_CKPT_DIR", "./ckpt/v5")
XML_PATH = os.environ.get("ACT_XML_PATH", "./mode/demo_scene.xml")
OUTPUT_DIR = os.environ.get("ACT_VIDEO_DIR", "./videos")
RECORD_VIDEO = os.environ.get("ACT_RECORD_VIDEO", "1") == "1"
MAX_STEPS = int(os.environ.get("ACT_DEPLOY_MAX_STEPS", "300"))
DEPLOY_SEED = int(os.environ.get("ACT_DEPLOY_SEED", "0"))
CHUNK_SIZE = int(os.environ.get("ACT_CHUNK_SIZE", "50"))
N_ACTION_STEPS = int(os.environ.get("ACT_N_ACTION_STEPS", "1"))
TEMPORAL_ENSEMBLE_COEFF = float(os.environ.get("ACT_TEMPORAL_ENSEMBLE_COEFF", "0.9"))
METRICS_PATH = os.environ.get("ACT_DEPLOY_METRICS_PATH", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_policy():
    """根据数据集特征和 checkpoint 构建 ACT 策略。"""
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=DATASET_ROOT)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        temporal_ensemble_coeff=TEMPORAL_ENSEMBLE_COEFF,
    )
    policy = ACTPolicy.from_pretrained(CKPT_DIR, config=cfg, dataset_stats=dataset_metadata.stats)
    policy.to(DEVICE)
    policy.reset()
    policy.eval()
    return policy


def write_deploy_metrics(video_path, step, success, failure_mode, status="ok", error=""):
    """写入 headless 闭环部署指标，方便和 viewer 部署结果分开分析。"""
    if not METRICS_PATH:
        return

    metrics = {
        "dataset_root": DATASET_ROOT,
        "ckpt_dir": CKPT_DIR,
        "xml_path": XML_PATH,
        "video_path": video_path if RECORD_VIDEO else "",
        "record_video": RECORD_VIDEO,
        "headless": True,
        "deploy_seed": DEPLOY_SEED,
        "max_steps": MAX_STEPS,
        "executed_steps": step,
        "success": success,
        "failure_mode": failure_mode,
        "status": status,
        "error": error,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "temporal_ensemble_coeff": TEMPORAL_ENSEMBLE_COEFF,
        "device": DEVICE,
    }
    metrics_dir = os.path.dirname(METRICS_PATH)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def close_env(env):
    """释放离屏渲染资源；无 viewer 时 close_viewer 也会安全清理 renderer。"""
    if env is None:
        return
    try:
        env.env.close_viewer()
    except Exception as exc:
        print(f"renderer close skipped: {exc}")


def main():
    print(f"device: {DEVICE}")
    print(f"dataset_root: {DATASET_ROOT}")
    print(f"ckpt_dir: {CKPT_DIR}")
    print(f"temporal_ensemble_coeff: {TEMPORAL_ENSEMBLE_COEFF}")
    print("headless: true")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUTPUT_DIR, f"deployment_headless_{timestamp}.mp4")
    video_writer = None
    env = None
    step = 0
    success = False
    failure_mode = "max_steps"
    status = "ok"
    error = ""

    try:
        policy = build_policy()
        env = SimpleEnv(XML_PATH, action_type="joint_angle", use_viewer=False)
        env.reset(seed=DEPLOY_SEED)
        img_transform = torchvision.transforms.ToTensor()

        while True:
            env.step_env()
            if env.check_success():
                print("Success")
                success = True
                failure_mode = "success"
                break

            state = env.get_ee_pose()
            image, wrist_image = env.grab_image()

            image = img_transform(Image.fromarray(image).resize((256, 256)))
            wrist_image = img_transform(Image.fromarray(wrist_image).resize((256, 256)))

            # 策略输入字段保持和训练数据一致；脚本只去掉 viewer，不改变模型接口。
            data = {
                "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(DEVICE),
                "observation.image": image.unsqueeze(0).to(DEVICE),
                "observation.wrist_image": wrist_image.unsqueeze(0).to(DEVICE),
                "task": ["Put mug cup on the plate"],
                "timestamp": torch.tensor([step / 20]).to(DEVICE),
            }

            action = policy.select_action(data)[0].cpu().detach().numpy()
            env.step(action)

            if RECORD_VIDEO:
                frame, _ = env.grab_image()
                if video_writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                # OpenCV 使用 BGR，需要从 MuJoCo 的 RGB 帧转换。
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            step += 1
            if MAX_STEPS and step >= MAX_STEPS:
                print(f"Reached ACT_DEPLOY_MAX_STEPS={MAX_STEPS}")
                failure_mode = "max_steps"
                break
    except Exception:
        status = "error"
        failure_mode = "runtime_error"
        error = traceback.format_exc()
        print(error)
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"video saved to: {video_path}")
        close_env(env)
        write_deploy_metrics(video_path, step, success, failure_mode, status=status, error=error)


if __name__ == "__main__":
    main()
