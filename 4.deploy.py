import json
import os
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from PIL import Image

from mujoco_env.y_env import SimpleEnv


# 部署默认使用仓库内已有数据集和 checkpoint；需要切换时用环境变量覆盖。
DATASET_ROOT = os.environ.get("ACT_DATASET_ROOT", "datasets/demo_v5_30demos_random")
CKPT_DIR = os.environ.get("ACT_CKPT_DIR", "./ckpt/v5")
XML_PATH = os.environ.get("ACT_XML_PATH", "./mode/demo_scene.xml")
OUTPUT_DIR = os.environ.get("ACT_VIDEO_DIR", "./videos")
RECORD_VIDEO = os.environ.get("ACT_RECORD_VIDEO", "1") == "1"
MAX_STEPS = int(os.environ.get("ACT_DEPLOY_MAX_STEPS", "100"))
DEPLOY_SEED = int(os.environ.get("ACT_DEPLOY_SEED", "0"))
CHUNK_SIZE = int(os.environ.get("ACT_CHUNK_SIZE", "50"))
N_ACTION_STEPS = int(os.environ.get("ACT_N_ACTION_STEPS", "1"))
TEMPORAL_ENSEMBLE_COEFF = float(os.environ.get("ACT_TEMPORAL_ENSEMBLE_COEFF", "0.9"))
METRICS_PATH = os.environ.get("ACT_DEPLOY_METRICS_PATH", "")
PLACEMENT_XY_THRESHOLD = float(os.environ.get("ACT_PLACEMENT_XY_THRESHOLD", "0.1"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_policy():
    """根据数据集特征和 checkpoint 构建 ACT 策略。"""
    dataset_metadata = LeRobotDatasetMetadata("ur_pnp", root=DATASET_ROOT)
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


def mean_or_none(values):
    """空列表写 None，避免把缺失指标误写成 0。"""
    return None if not values else float(np.mean(values))


def max_or_none(values):
    """空列表写 None，避免把缺失指标误写成 0。"""
    return None if not values else float(np.max(values))


def infer_failure_mode(success, final_task_metrics, current_failure_mode):
    """根据双口径成功信号细分失败原因，突出夹爪未松开的案例。"""
    if success:
        return "success"
    if not final_task_metrics:
        return current_failure_mode
    if final_task_metrics.get("placement_success"):
        if final_task_metrics.get("final_gripper_qpos", 0.0) >= 0.1:
            return "placement_success_gripper_closed"
        return "placement_success_incomplete"
    return current_failure_mode


def select_action_and_measure(policy, data, step, chunk_history):
    """保持 LeRobot temporal ensemble 行为，同时读取 action chunk 计算预测不一致性。"""
    if getattr(policy.config, "temporal_ensemble_coeff", None) is None:
        action = policy.select_action(data)[0].cpu().detach().numpy()
        return action, None

    with torch.no_grad():
        actions = policy.predict_action_chunk(data)
        chunk = actions[0].detach().cpu().numpy()
        chunk_history.append((step, chunk.copy()))
        chunk_history[:] = [
            (start_step, old_chunk)
            for start_step, old_chunk in chunk_history
            if step - start_step < old_chunk.shape[0]
        ]

        current_predictions = []
        for start_step, old_chunk in chunk_history:
            offset = step - start_step
            if 0 <= offset < old_chunk.shape[0]:
                current_predictions.append(old_chunk[offset])

        prediction_inconsistency = None
        if len(current_predictions) >= 2:
            stacked = np.stack(current_predictions, axis=0)
            mean_action = np.mean(stacked, axis=0)
            prediction_inconsistency = float(np.mean(np.linalg.norm(stacked - mean_action, axis=1)))

        action = policy.temporal_ensembler.update(actions)[0].cpu().detach().numpy()
        return action, prediction_inconsistency


def build_eval_metrics(env, initial_task_metrics, action_deltas, prediction_inconsistencies, trajectory_metrics):
    """汇总单次部署的论文评价指标。"""
    final_task_metrics = env.get_task_metrics(PLACEMENT_XY_THRESHOLD) if env is not None else {}
    xy_dists = [item["mug_plate_xy_dist"] for item in trajectory_metrics]
    z_gaps = [item["mug_plate_z_gap"] for item in trajectory_metrics]
    metrics = {
        "placement_xy_threshold": PLACEMENT_XY_THRESHOLD,
        "initial_mug_position": initial_task_metrics.get("mug_position", []),
        "initial_plate_position": initial_task_metrics.get("plate_position", []),
        "strict_success": final_task_metrics.get("strict_success", False),
        "placement_success": final_task_metrics.get("placement_success", False),
        "action_smoothness_mean": mean_or_none(action_deltas),
        "action_smoothness_max": max_or_none(action_deltas),
        "prediction_inconsistency_mean": mean_or_none(prediction_inconsistencies),
        "prediction_inconsistency_max": max_or_none(prediction_inconsistencies),
        "final_mug_plate_xy_dist": final_task_metrics.get("mug_plate_xy_dist"),
        "min_mug_plate_xy_dist": min(xy_dists) if xy_dists else final_task_metrics.get("mug_plate_xy_dist"),
        "final_mug_plate_z_gap": final_task_metrics.get("mug_plate_z_gap"),
        "min_mug_plate_z_gap": min(z_gaps) if z_gaps else final_task_metrics.get("mug_plate_z_gap"),
        "final_gripper_qpos": final_task_metrics.get("final_gripper_qpos"),
        "final_ee_z": final_task_metrics.get("ee_z"),
    }
    return metrics


def write_deploy_metrics(video_path, step, success, failure_mode, eval_metrics, status="ok", error=""):
    """写入闭环部署指标，保证异常实验也能留下可分析证据。"""
    if not METRICS_PATH:
        return

    metrics = {
        "dataset_root": DATASET_ROOT,
        "ckpt_dir": CKPT_DIR,
        "xml_path": XML_PATH,
        "video_path": video_path if RECORD_VIDEO else "",
        "record_video": RECORD_VIDEO,
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
        "placement_xy_threshold": PLACEMENT_XY_THRESHOLD,
        "device": DEVICE,
    }
    metrics.update(eval_metrics)
    metrics_dir = os.path.dirname(METRICS_PATH)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def close_env(env):
    """显式释放 MuJoCo viewer，降低 WSLg 连续部署时的窗口资源残留。"""
    if env is None:
        return
    try:
        env.env.close_viewer()
    except Exception as exc:
        print(f"viewer close skipped: {exc}")


def main():
    print(f"device: {DEVICE}")
    print(f"dataset_root: {DATASET_ROOT}")
    print(f"ckpt_dir: {CKPT_DIR}")
    print(f"temporal_ensemble_coeff: {TEMPORAL_ENSEMBLE_COEFF}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUTPUT_DIR, f"deployment_{timestamp}.mp4")
    video_writer = None
    env = None
    step = 0
    success = False
    failure_mode = "viewer_closed"
    status = "ok"
    error = ""
    initial_task_metrics = {}
    action_deltas = []
    prediction_inconsistencies = []
    trajectory_metrics = []
    chunk_history = []
    prev_action = None

    try:
        policy = build_policy()
        env = SimpleEnv(XML_PATH, action_type="joint_angle")
        env.reset(seed=DEPLOY_SEED)
        initial_task_metrics = env.get_task_metrics(PLACEMENT_XY_THRESHOLD)
        img_transform = torchvision.transforms.ToTensor()

        while env.env.is_viewer_alive():
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            task_metrics = env.get_task_metrics(PLACEMENT_XY_THRESHOLD)
            trajectory_metrics.append(task_metrics)

            if task_metrics["strict_success"]:
                print("Success")
                success = True
                failure_mode = "success"
                break

            state = env.get_ee_pose()
            image, wrist_image = env.grab_image()

            image = img_transform(Image.fromarray(image).resize((256, 256)))
            wrist_image = img_transform(Image.fromarray(wrist_image).resize((256, 256)))

            # 策略输入需要和训练数据的 observation/task/timestamp 字段保持一致。
            data = {
                "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(DEVICE),
                "observation.image": image.unsqueeze(0).to(DEVICE),
                "observation.wrist_image": wrist_image.unsqueeze(0).to(DEVICE),
                "task": ["Put mug cup on the plate"],
                "timestamp": torch.tensor([step / 20]).to(DEVICE),
            }

            action, prediction_inconsistency = select_action_and_measure(policy, data, step, chunk_history)
            if prediction_inconsistency is not None:
                prediction_inconsistencies.append(prediction_inconsistency)
            if prev_action is not None:
                action_deltas.append(float(np.linalg.norm(action - prev_action)))
            prev_action = action.copy()

            env.step(action)
            env.render()

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
        eval_metrics = build_eval_metrics(
            env,
            initial_task_metrics,
            action_deltas,
            prediction_inconsistencies,
            trajectory_metrics,
        )
        success = success or bool(eval_metrics.get("strict_success", False))
        failure_mode = infer_failure_mode(success, eval_metrics, failure_mode)
        close_env(env)
        write_deploy_metrics(video_path, step, success, failure_mode, eval_metrics, status=status, error=error)


if __name__ == "__main__":
    main()

