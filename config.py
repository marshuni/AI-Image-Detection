import torch

# 设置随机种子保证可重复性
torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 2