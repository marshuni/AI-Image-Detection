import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

# 定义ResNet50模型
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=10, weights=ResNet50_Weights.DEFAULT):
        super(ResNet50Classifier, self).__init__()
        # 加载预训练的ResNet50模型
        self.resnet50 = torchvision.models.resnet50(weights=weights)
        
        # 冻结所有层（如果只想训练分类层）
        # for param in self.resnet50.parameters():
        #     param.requires_grad = False
            
        # 替换最后的全连接层以适应我们的分类任务
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet50(x)