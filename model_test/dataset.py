import sqlite3
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# 疾病标签到整数索引的严格映射字典
CLASS_MAPPING = {
    'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3,
    'BKL': 4, 'DF': 5, 'VASC': 6
}

class HAM10000NodeDataset(Dataset):
    def __init__(self, db_path, target_size=(256, 256)):
        """
        单节点皮肤科多任务数据集加载器
        :param db_path: 该节点的 SQLite 数据库路径 (例如: db_node_001.db)
        :param target_size: 统一缩放的图像尺寸，适配 EfficientNet-B3
        """
        self.db_path = db_path
        self.target_size = target_size
        self.samples = self._load_from_db()

        # 仅针对原图的颜色变换 (归一化到 ImageNet 标准)
        self.image_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_from_db(self):
        """从 pacs_dermatology 表中提取有效数据"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件未找到: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 查询图像路径、掩码路径和真实标签
        cursor.execute('''
                       SELECT image_path, mask_path, ground_truth
                       FROM pacs_dermatology
                       WHERE image_path IS NOT NULL
                         AND ground_truth IS NOT NULL
                       ''')
        rows = cursor.fetchall()
        conn.close()

        print(f"✅ 成功从 {self.db_path} 中加载了 {len(rows)} 条影像记录。")
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label_str = self.samples[idx]

        # 1. 解析分类标签 (Label)
        label_idx = CLASS_MAPPING.get(label_str.upper(), -1)
        if label_idx == -1:
            raise ValueError(f"遇到未知的疾病标签: {label_str}")
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # 2. 读取并处理原图 (Image)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"无法读取原图 {img_path}: {e}")

        # 缩放原图 (双线性插值)
        image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        image_tensor = self.image_normalize(image)

        # 3. 读取并处理分割掩码 (Mask)
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert('L') # 转为灰度图
            except Exception as e:
                raise RuntimeError(f"无法读取掩码 {mask_path}: {e}")
        else:
            # 如果某张图恰好没有掩码，生成一个全黑的零矩阵防止程序崩溃
            mask = Image.new('L', image.size, color=0)

        # 缩放掩码 (最近邻插值，极其关键！)
        mask = TF.resize(mask, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        mask_tensor = TF.to_tensor(mask) # ToTensor 会将 0-255 转为 0.0-1.0

        # 二值化确保掩码只有 0 和 1 (针对病灶区域)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor, label_tensor

# --- 测试主函数 ---
if __name__ == "__main__":
    # 假设我们要测试 NODE_001 的数据库
    TEST_DB = "/home/sunjingbo/py/HIS/client_database/db_node_001.db"

    try:
        # 实例化 Dataset
        dataset = HAM10000NodeDataset(db_path=TEST_DB, target_size=(256, 256))

        # 实例化 DataLoader，设置 batch_size 为我们之前规划的 16
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        # 模拟一次 Epoch 的数据抓取
        print("🚀 开始测试数据管道流动...")
        for batch_idx, (images, masks, labels) in enumerate(dataloader):
            print(f"\n--- Batch {batch_idx + 1} ---")
            print(f"原图张量形状: {images.shape}  -> (Batch, Channels, Height, Width)")
            print(f"掩码张量形状: {masks.shape}   -> (Batch, 1, Height, Width)")
            print(f"标签张量形状: {labels.shape}     -> (Batch,)")
            print(f"当前批次的真实标签: {labels.tolist()}")

            # 只测试提取第一个 Batch 就退出
            break

    except Exception as e:
        print(f"❌ 测试失败: {e}")