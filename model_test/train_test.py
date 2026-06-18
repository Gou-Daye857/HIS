import torch
import torch.nn as nn
from torchvision import models

class BareMetalUnet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # 显式定义 encoder
        resnet = models.resnet34(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)
        seg_mask = self.decoder(feat)
        cls_logits = self.cls_head(feat)
        return cls_logits, seg_mask

# --- 测试 ---
if __name__ == "__main__":
    # 关闭 cuDNN 以防万一
    torch.backends.cudnn.enabled = False

    device = torch.device("cuda")
    model = BareMetalUnet().to(device)

    # 一个极简的测试输入
    test_input = torch.randn(1, 3, 256, 256).to(device)

    # 如果这行代码不报错，说明我们彻底清理了 SMP 这个累赘
    cls, seg = model(test_input)
    print(f"成功！输出形状: 分类 {cls.shape}, 分割 {seg.shape}")