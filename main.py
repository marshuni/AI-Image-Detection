import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

import tqdm

from config import *

from dataloader import get_dataloaders
train_loader, test_loader = get_dataloaders(batch_size=256)


from model.baseline import ResNet50Classifier
# 初始化模型
num_classes = 10  # CIFAR10有10个类别
model = ResNet50Classifier(num_classes=num_classes, weights=ResNet50_Weights.DEFAULT)
model = model.to(device)

# 加载保存的模型
loaded_model = ResNet50Classifier(num_classes=num_classes, weights=None)
loaded_model.load_state_dict(torch.load('resnet50_classifier.pth'))
loaded_model = loaded_model.to(device)


from train import train_model
from test import test_model

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练和测试模型
# train_model(model, train_loader, criterion, optimizer, num_epochs=5)
test_model(loaded_model, test_loader)

# 保存模型
# torch.save(model.state_dict(), 'resnet50_classifier.pth')


