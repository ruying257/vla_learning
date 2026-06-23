from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

# 加载你刚刚录制的数据集
dataset = LeRobotDataset("omy_pnp", root="./demo_data")

print(f"✅ 成功加载数据集！总帧数: {len(dataset)}")

# 取出第 50 帧的数据（你可以随便换个数字测试）
frame_data = dataset[340]

# 提取主视角和手腕视角的图片
agent_img = frame_data["observation.image"]
wrist_img = frame_data["observation.wrist_image"]

# 把 PyTorch Tensor 转换回 PIL 图片并显示出来
# （LeRobot 默认读取出来的是 CxHxW 格式的张量，并归一化到了 0-1 之间）
import torchvision.transforms.functional as F
agent_pil = F.to_pil_image(agent_img)
wrist_pil = F.to_pil_image(wrist_img)

print("正在打开第 50 帧的图片，请查看弹出的窗口...")
agent_pil.show()