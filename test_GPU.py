import torch

# 是否可用 GPU
print(torch.cuda.is_available())  # True = 可用

# 显卡数量
print(torch.cuda.device_count())

# 显卡型号
print(torch.cuda.get_device_name(0))