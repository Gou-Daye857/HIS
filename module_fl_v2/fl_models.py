# module_fl/fl_models.py
import torch.nn as nn
from torchvision import models

class BareMetalUnet(nn.Module):
    """
    轻量化皮肤病灶分割与分类双任务网络
    (从 fl_core 中剥离，便于后期替换为 Swin-Unet 等更高级架构)
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # 编码器 (Encoder) - 采用预训练的 ResNet34 作为主干
        resnet = models.resnet34(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # 解码器 (Decoder) - 用于病灶区域分割 (Segmentation)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        )

        # 分类头 (Classification Head) - 用于病种分类
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)
        # 同时返回：分类 Logits 和 分割特征图
        return self.cls_head(feat), self.decoder(feat)