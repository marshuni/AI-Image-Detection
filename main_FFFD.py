import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights

import tqdm

from config import *
from train import *
from test import *
from model.FFFD import FeatureFusionForgeryDetector

# 采用原始数据集
# from dataloader.multimodal import get_dataloaders
# train_loader, val_loader, test_loader = get_dataloaders(batch_size=256)

# 采用GenImage数据集
from dataloader.genimage import get_dataloaders
train_loader, val_loader = get_dataloaders(batch_size=256)

# Multimodel Fusion Forgery Detection
model_path = './checkpoint/FFFD_resnet18_classifier_10.pth'
def train_FFFD():
    # 初始化模型
    model = FeatureFusionForgeryDetector()
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 训练和保存模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=25, model_path=model_path)
    torch.save(model.state_dict(), model_path)

    return model

def load_FFFD():
    # 加载保存的模型
    model = FeatureFusionForgeryDetector()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model

def test_FFFD(model):
    # 测试模型
    labels,predictions,scores = test_model(model, val_loader)
    print(f"测试集大小: {len(labels)}")
    # print(labels)
    # print(predictions)

    accuracy, precision, recall, f1 = calculate_metrics(labels,predictions)
    auroc = calculate_roc(labels,scores)

    results = [
        f"准确率 (Accuracy): {accuracy:.4f}",
        f"精确率 (Precision): {precision:.4f}",
        f"召回率 (Recall): {recall:.4f}",
        f"F1-Score: {f1:.4f}",
        f"ROC-AUC: {auroc:.4f}",
    ]
    for line in results:
        print(line)

if __name__ == "__main__":
    # model = train_FFFD()
    model = load_FFFD()

    test_FFFD(model)
    # test_model_for_kaggle_submission(model, test_loader)
