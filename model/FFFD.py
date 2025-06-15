import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Truncated(nn.Module):
    def __init__(self, in_channels=3, weights=ResNet18_Weights.DEFAULT):
        super().__init__()
        base = resnet18(weights=weights)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
        )
        # 新增降维卷积，输出256通道
        self.reduce = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.stem(x)      # [B, 128, H/8, W/8]
        x = self.reduce(x)    # [B, 256, H/16, W/16]
        x = self.bn(x)
        x = self.relu(x)
        return x

class FeatureFusionForgeryDetector(nn.Module):
    def __init__(self, 
                 resnet18_weights=ResNet18_Weights.DEFAULT,
                 mobilenet_weights=None):
        super().__init__()
        # RGB: ResNet18裁剪+降维
        self.rgb_model = ResNet18Truncated(in_channels=3, weights=resnet18_weights)
        # FFT: ResNet18裁剪+降维
        self.fft_model = ResNet18Truncated(in_channels=1, weights=resnet18_weights)
        # SRM: ResNet18裁剪+降维
        self.srm_model = ResNet18Truncated(in_channels=1, weights=resnet18_weights)
        # LBP: ResNet18裁剪+降维
        self.lbp_model = ResNet18Truncated(in_channels=1, weights=resnet18_weights)

        self.fusion_conv = nn.Conv2d(256 * 4, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        rgb = x[:, 0:3, :, :]
        fft = x[:, 3:4, :, :]
        srm = x[:, 4:5, :, :]
        lbp = x[:, 5:6, :, :]

        rgb_feat = self.rgb_model(rgb)    # [B, 256, H/16, W/16]
        fft_feat = self.fft_model(fft)    # [B, 256, H/16, W/16]
        srm_feat = self.srm_model(srm)    # [B, 256, H/16, W/16]
        lbp_feat = self.lbp_model(lbp)  # [B, 256, H/16, W/16]

        feat = torch.cat([rgb_feat, fft_feat, srm_feat, lbp_feat], dim=1)  # [B, 1024, H/16, W/16]
        feat = self.fusion_conv(feat)  # [B, 256, H/16, W/16]
        feat = self.pool(feat)         # [B, 256, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 256]
        out = self.fc(feat)
        return out