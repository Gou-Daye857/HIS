import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from PIL import Image
import sqlite3
import os
from tqdm import tqdm

# --- 1. 配置参数 ---
DB_PATH = "/home/sunjingbo/py/HIS/client_database/db_node_001.db"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAPPING = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

# --- 2. 模型定义 ---
class BareMetalUnet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        resnet = models.resnet34(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
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

# --- 3. 数据集 ---
class HAM10000NodeDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, target_size=(256, 256)):
        self.target_size = target_size
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path, mask_path, ground_truth FROM pacs_dermatology WHERE image_path IS NOT NULL")
        self.samples = cursor.fetchall()
        conn.close()
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label_str = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = TF.resize(image, self.target_size)
        image_tensor = self.norm(image)

        mask = Image.open(mask_path).convert('L') if mask_path and os.path.exists(mask_path) else Image.new('L', self.target_size, color=0)
        mask = TF.resize(mask, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        mask_tensor = TF.to_tensor(mask) > 0.5

        label_idx = CLASS_MAPPING.get(label_str.upper(), 0)
        return image_tensor, mask_tensor.float(), torch.tensor(label_idx, dtype=torch.long)

# --- 4. 辅助指标计算 ---
def calculate_dice(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

# --- 5. 主训练流程 ---
def main():
    torch.backends.cudnn.enabled = False

    # 初始化模型与数据
    model = BareMetalUnet(num_classes=7).to(DEVICE)
    full_dataset = HAM10000NodeDataset(DB_PATH)

    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    print(f"🚀 开始训练，设备: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        # 训练过程
        for imgs, masks, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks, lbls = imgs.to(DEVICE), masks.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()

            cls_out, seg_out = model(imgs)
            loss = 0.5 * criterion_cls(cls_out, lbls) + 0.5 * criterion_seg(seg_out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证过程
        model.eval()
        val_acc = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, masks, lbls in val_loader:
                imgs, masks, lbls = imgs.to(DEVICE), masks.to(DEVICE), lbls.to(DEVICE)
                cls_out, seg_out = model(imgs)

                # Accuracy
                preds = cls_out.argmax(dim=1)
                val_acc += (preds == lbls).sum().item()
                # Dice
                val_dice += calculate_dice(seg_out, masks).item()

        print(f"Epoch {epoch+1} 总结: Loss={train_loss/len(train_loader):.4f}, Acc={val_acc/len(val_ds):.4f}, Dice={val_dice/len(val_loader):.4f}")

        # 保存模型权重
        torch.save(model.state_dict(), f"weights_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()