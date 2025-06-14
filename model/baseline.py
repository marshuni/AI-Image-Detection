import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

# 定义ResNet18模型
class ResNet18Classifier(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super(ResNet18Classifier, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet18 = torchvision.models.resnet18(weights=weights)
        
        # 冻结所有层（如果只想训练分类层）
        # for param in self.resnet18.parameters():
        #     param.requires_grad = False
            
        # 替换最后的全连接层以适应我们的分类任务
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.resnet18(x)