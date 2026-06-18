# backend_modules/api_pacs.py
import traceback
import base64
import os
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# 引入抽离后的模型架构
try:
    from module_fl.fl_models import BareMetalUnet
except ImportError:
    # 兼容没有安装特定模块时的测试
    import sys
    import torch.nn as nn
    from torchvision import models
    class BareMetalUnet(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            resnet = models.resnet34(weights=None)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
            )
            self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        def forward(self, x):
            feat = self.encoder(x)
            return self.cls_head(feat), self.decoder(feat)

# 皮肤病类别映射 (反向映射用于输出展示)
REV_CLASS_MAPPING = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AKIEC', 4: 'BKL', 5: 'DF', 6: 'VASC'}
CHECKPOINT_PATH = "./checkpoints/fl_model_best.pth"  # 优先加载联邦学习历史最优模型

def handle_action(action, business_data, node_db_conn):
    """皮肤科影像 PACS 模块的报文处理中枢"""
    if action == "get_all":
        return _get_all_records(node_db_conn)
    elif action == "get_image_data":
        return _get_image_base64(business_data)
    elif action == "ai_predict":
        return _handle_ai_predict(business_data, node_db_conn)
    else:
        return {"status": "error", "message": f"未知的 PACS 操作指令: {action}"}

def _get_image_base64(business_data):
    """在 Linux 后端读取图片并转码为 Base64 流传给前端"""
    img_path = business_data.get("image_path")
    mask_path = business_data.get("mask_path")

    def encode_img(path):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"[PACS 转码异常] 无法读取文件 {path}: {e}")
            return None

    return {
        "status": "success",
        "data": {
            "image_b64": encode_img(img_path),
            "mask_b64": encode_img(mask_path)
        }
    }

def _get_all_records(db_conn):
    """获取所有皮肤科影像和患者基本关联记录"""
    cursor = db_conn.cursor()
    try:
        sql = '''
              SELECT
                  pd.record_id, p.name, p.gender, p.age,
                  pd.image_id, pd.image_path, pd.mask_path, pd.ground_truth,
                  pd.ai_prediction, pd.ai_confidence, pd.upload_time
              FROM pacs_dermatology pd
                       JOIN patients p ON pd.patient_id = p.patient_id
              ORDER BY pd.upload_time DESC \
              '''
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"status": "success", "data": records}
    except Exception as e:
        error_msg = str(e)
        print(f"[PACS 模块异常] 数据库查询失败:\n{traceback.format_exc()}")
        return {"status": "error", "message": f"数据库查询异常: {error_msg}"}

def _handle_ai_predict(business_data, db_conn):
    """核心功能：执行真实的 AI 辅助诊断推理并同步回写数据库"""
    img_path = business_data.get("image_path")
    record_id = business_data.get("record_id")

    if not img_path or not os.path.exists(img_path):
        return {"status": "error", "message": "医院本地无源影像文件，无法发起 AI 诊断"}

    if not os.path.exists(CHECKPOINT_PATH):
        return {"status": "error", "message": "尚未检测到训练完成的联邦学习聚合参数包 (fl_model_best.pth)，请先参与联合训练！"}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 实例化解耦后的模型，并加载最新的联邦最优权重
        model = BareMetalUnet(num_classes=7).to(device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval()

        # 2. 图像读取与标准预处理流程 (与联邦训练环境保持高度一致)
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        resized_img = TF.resize(image, (256, 256))

        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = norm(resized_img).unsqueeze(0).to(device) # 增加 Batch 维度

        # 3. 闭灯前向推理
        with torch.no_grad():
            cls_out, seg_out = model(img_tensor)

            # --- 分类多任务分支处理 ---
            probabilities = F.softmax(cls_out, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            pred_class = REV_CLASS_MAPPING.get(predicted_idx.item(), "未知")
            conf_val = float(confidence.item())

            # --- 分割多任务分支处理 (生成动态预测掩码图) ---
            seg_prob = torch.sigmoid(seg_out)[0, 0]  # 取单通道特征概率图
            seg_mask = (seg_prob > 0.5).cpu().numpy() * 255  # 阈值二值化转换为 0 / 255 像素值

            # 还原为原图尺寸并转码为 Base64 字节流
            mask_img = Image.fromarray(seg_mask.astype('uint8'), mode='L')
            mask_img = mask_img.resize(original_size, Image.Resampling.NEAREST)

            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            ai_mask_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 4. 本地医疗数据闭环：将 AI 推理结论回写落盘至当前节点数据库
        cursor = db_conn.cursor()
        cursor.execute('''
                       UPDATE pacs_dermatology
                       SET ai_prediction = ?, ai_confidence = ?
                       WHERE record_id = ?
                       ''', (pred_class, conf_val, record_id))
        db_conn.commit()

        # 5. 核弹级垃圾回收，杜绝常驻推理常态化侵占显存
        del model, img_tensor, cls_out, seg_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "data": {
                "prediction": pred_class,
                "confidence": conf_val,
                "ai_mask_b64": ai_mask_b64
            }
        }

    except Exception as e:
        print(f"[PACS 推理崩溃异常]:\n{traceback.format_exc()}")
        return {"status": "error", "message": f"边缘节点端 AI 推理失败: {str(e)}"}