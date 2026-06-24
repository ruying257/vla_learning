# WSL2 ACT 跑通说明

本说明用于在 WSL2 中跑通当前 ACT 项目。建议使用独立 `vla-code` 环境，不复用 `mp1`，避免破坏 MP1 的旧版 MuJoCo/Gym 依赖。

## 环境创建

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n vla-code python=3.10 -y
conda activate vla-code
cd /mnt/d/Desktop/vla_learning
```

安装依赖时优先使用国内镜像，并显式关闭代理：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  "numpy>=2,<2.3" mujoco==3.6.0 lerobot==0.4.4 transformers==4.46.3 \
  "wandb>=0.15.0" opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 \
  "pyautogui>=0.9.54" matplotlib scipy
```

## 快速验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import mujoco, lerobot, transformers, cv2, PIL, wandb, numpy; print('imports OK')"
python -m py_compile 1.collect_data.py 2.visualize_data.py 3.train.py 4.deploy.py
```

## Smoke Test

训练脚本默认读取 `datasets/demo_v5_30demos_random`，部署脚本默认加载 `ckpt/v5`。

```bash
ACT_VIS_MAX_STEPS=1 python 2.visualize_data.py
ACT_TRAINING_STEPS=1 ACT_LOG_FREQ=1 ACT_BATCH_SIZE=2 ACT_NUM_WORKERS=0 python 3.train.py
ACT_DEPLOY_MAX_STEPS=20 ACT_RECORD_VIDEO=1 python 4.deploy.py
```

常用环境变量：

- `ACT_DATASET_ROOT`：训练/部署读取的数据集目录，默认 `datasets/demo_v5_30demos_random`
- `ACT_CKPT_DIR`：训练保存或部署加载的 checkpoint 目录
- `ACT_TRAINING_STEPS`：训练步数，跑通验证可设为 `1`
- `ACT_USE_WANDB`：设为 `1` 时启用 WandB，默认关闭
- `ACT_VIS_MAX_STEPS`：回放最多运行步数，跑通验证可设为 `1`
- `ACT_DEPLOY_MAX_STEPS`：部署最多运行步数，设为 `0` 表示不限制
- `ACT_TEMPORAL_ENSEMBLE_COEFF`：部署时间集成平滑系数，默认 `0.9`

## 调参实验

跑通基础链路后，按 `README_ACT_TUNING.md` 执行动作块长度、学习率、数据分布和部署平滑实验。

## Headless 部署录制

WSL2 中实时 MuJoCo viewer 可能很卡；批量部署评估时推荐使用独立 headless 脚本，不弹出窗口，直接用固定相机离屏渲染并保存视频：

```bash
ACT_RECORD_VIDEO=1 ACT_DEPLOY_MAX_STEPS=300 ACT_CKPT_DIR=./ckpt/act_y python 4.deploy_headless.py
```

两种部署脚本的定位：

- `4.deploy.py`：保留原 viewer 部署流程，适合少量人工观察。
- `4.deploy_headless.py`：WSL2 推荐的批量评估/录视频流程，不依赖 `is_viewer_alive()`。

本机已验证 `MUJOCO_GL=osmesa` 可用于 headless smoke；如果换机器后离屏渲染初始化失败，可以显式指定 MuJoCo OpenGL 后端后再运行：

```bash
export MUJOCO_GL=osmesa
ACT_RECORD_VIDEO=1 ACT_DEPLOY_MAX_STEPS=300 python 4.deploy_headless.py
```

