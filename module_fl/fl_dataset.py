import os
import json
import time
import os
import sqlite3
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
CLASS_MAPPING = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
class HAM10000NodeDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, target_size=(256, 256)):
        self.target_size = target_size
        if not os.path.exists(db_path):
            self.samples = []
            return
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT image_path, mask_path, ground_truth FROM pacs_dermatology WHERE image_path IS NOT NULL")
            self.samples = cursor.fetchall()
        except:
            self.samples = []
        finally:
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
        return image_tensor, mask_tensor.float(), torch.tensor(CLASS_MAPPING.get(label_str.upper(), 0), dtype=torch.long)
