import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights

import tqdm

from config import *

from dataloader import get_dataloaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=256)


from model.baseline import ResNet34Classifier
# 初始化模型

model = ResNet34Classifier(num_classes=num_classes, weights=ResNet34_Weights.DEFAULT)
model = model.to(device)

# 加载保存的模型
# loaded_model = ResNet34Classifier(num_classes=num_classes, weights=None)
# loaded_model.load_state_dict(torch.load('resnet34_classifier.pth'))
# loaded_model = loaded_model.to(device)


from train import train_model
from test import test_model

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 训练和测试模型
train_model(model, train_loader, criterion, optimizer, num_epochs=5)
test_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), './checkpoint/resnet34_classifier.pth')


