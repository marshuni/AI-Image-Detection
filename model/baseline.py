import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet34_Weights

# 定义ResNet34模型
class ResNet34Classifier(nn.Module):
    def __init__(self, weights=ResNet34_Weights.DEFAULT):
        super(ResNet34Classifier, self).__init__()
        # 加载预训练的ResNet34模型
        self.resnet34 = torchvision.models.resnet34(weights=weights)
        
        # 冻结所有层（如果只想训练分类层）
        # for param in self.resnet34.parameters():
        #     param.requires_grad = False
            
        # 替换最后的全连接层以适应我们的分类任务
        num_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.resnet34(x)