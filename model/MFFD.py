import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
    
class MultimodalFusionForgeryDetector(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super(MultimodalFusionForgeryDetector, self).__init__()
        self.resnet18 = torchvision.models.resnet18(weights=weights)
        
        # 修改conv1层：支持6通道
        old_conv1 = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=6,  # 修改为6通道输入
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )

        # 初始化新通道的权重
        with torch.no_grad():
            self.resnet18.conv1.weight[:, :3] = old_conv1.weight  # 拷贝前3通道权重
            # 初始化第4-6通道为平均值
            self.resnet18.conv1.weight[:, 3:] = old_conv1.weight.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        
        
        # 取消冻结所有层
        # for param in self.resnet18.parameters():
        #     param.requires_grad = True

        # 替换最后的全连接层
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 1)  # 用于二分类

    def forward(self, x):
        return self.resnet18(x)