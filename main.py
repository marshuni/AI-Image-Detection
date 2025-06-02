import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights

import tqdm

from config import *

from dataloader import get_dataloaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=256)


from model.baseline import ResNet34Classifier
# 初始化模型

# model = ResNet34Classifier(weights=ResNet34_Weights.DEFAULT)
# model = model.to(device)

# 加载保存的模型
model = ResNet34Classifier(weights=None)
model.load_state_dict(torch.load('./checkpoint/resnet34_classifier.pth'))
model = model.to(device)


from train import *
from test import *

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 训练和保存模型
# train_model(model, train_loader, criterion, optimizer, num_epochs=5)
# torch.save(model.state_dict(), './checkpoint/resnet34_classifier.pth')

# 测试模型
# labels,predictions = test_model(model, val_loader)
test_model_for_kaggle_submission(model, test_loader)





