# module_fl/fl_core.py
import os
import json
import time
import copy
import gc
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from PIL import Image
import sqlite3
from module_fl.fl_models import BareMetalUnet

from module_fl.fl_state import FL_STATUS
from server_dispatcher import get_node_database

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAPPING = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
MODEL_SIZE_MB = 142.5
CHECKPOINT_DIR = "./checkpoints"
LOG_FILE_PATH = "fl_training_history.log"
PROGRESS_JSON = "./checkpoints/fl_progress.json"

def write_log(msg):
    global FL_STATUS
    FL_STATUS["traffic_logs"].append(msg)
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
    except:
        pass

def save_progress_json():
    """将核心联邦状态持久化到磁盘，预防断电"""
    try:
        state_to_save = {
            "current_round": FL_STATUS["current_round"],
            "total_traffic_mb": FL_STATUS["total_traffic_mb"],
            "metrics_history": FL_STATUS["metrics_history"],
            "best_metrics": FL_STATUS["best_metrics"],
            "round_participants": FL_STATUS.get("round_participants", {}) # 核心：落盘双重保险
        }
        with open(PROGRESS_JSON, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"进度状态保存失败: {e}")

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

def calculate_dice(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

# === 联邦训练核心循环 ===
def federated_train_loop(target_rounds, local_epochs, node_list, start_round=0):
    global FL_STATUS
    torch.backends.cudnn.enabled = False
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    global_model = None
    local_model = None
    global_weights = None
    fed_avg_weights = None

    write_log(f"\n=======================================================")
    write_log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔥 核心引擎点火：总规划终点为第 {target_rounds} 轮，当前起跑点：第 {start_round+1} 轮")

    try:
        global_model = BareMetalUnet(num_classes=7).to(DEVICE)

        # 判断是恢复训练还是全新开始
        latest_pth = None
        if start_round > 0:
            # 【核心逻辑】：严格解析匹配带有节点信息的断点 pth 文件
            search_pattern = os.path.join(CHECKPOINT_DIR, f"fl_model_round_{start_round}_nodes_*.pth")
            matched_files = glob.glob(search_pattern)
            if matched_files:
                latest_pth = matched_files[0]
            else:
                # 兼容旧版本的回退方案
                fallback_pth = os.path.join(CHECKPOINT_DIR, "fl_checkpoint_latest.pth")
                if os.path.exists(fallback_pth):
                    latest_pth = fallback_pth

        if latest_pth and os.path.exists(latest_pth):
            global_model.load_state_dict(torch.load(latest_pth, map_location=DEVICE))
            write_log(f"[{time.strftime('%H:%M:%S')}] 🔐 成功从磁盘载入第 {start_round} 轮断点权重包: {os.path.basename(latest_pth)}")
        elif start_round > 0:
            write_log(f"[{time.strftime('%H:%M:%S')}] ⚠️ 未找到第 {start_round} 轮对应的权重包，引擎将从头初始化！")
            start_round = 0

        if start_round == 0:
            write_log(f"[{time.strftime('%H:%M:%S')}] 🌱 全新冷启动：全局中心骨干网随机参数初始化完成。")

        global_weights = global_model.state_dict()
        criterion_cls = nn.CrossEntropyLoss()
        criterion_seg = nn.BCEWithLogitsLoss()

        for r in range(start_round, target_rounds):
            if not FL_STATUS["is_training"]:
                write_log(f"[{time.strftime('%H:%M:%S')}] ⏸ 收到上层系统安全挂起指令，平稳保存进度退出。")
                break

            FL_STATUS["current_round"] = r + 1

            # 安全对齐 metrics_history 长度，防止直接越界
            while len(FL_STATUS["metrics_history"]) < r + 1:
                FL_STATUS["metrics_history"].append({"round": len(FL_STATUS["metrics_history"])+1, "loss": 0.0, "acc": 0.0, "dice": 0.0})

            local_updates = []
            total_samples_this_round = 0
            round_loss, round_acc, round_dice = 0, 0, 0
            valid_nodes_count = 0
            participating_node_ids = []

            for node in node_list:
                node_name = node["name"]
                node_id = str(node["id"])

                db_path = get_node_database(node["id"])
                if not db_path or not os.path.exists(db_path):
                    continue

                FL_STATUS["current_stage"] = "参数下发"
                FL_STATUS["current_node"] = node_name
                FL_STATUS["total_traffic_mb"] += MODEL_SIZE_MB
                write_log(f"[{time.strftime('%H:%M:%S')}] 下发全局参数至【{node_name}】，下行流量: {MODEL_SIZE_MB}MB")
                time.sleep(0.3)

                dataset = HAM10000NodeDataset(db_path)
                if len(dataset) < 5:
                    write_log(f"⚠️ 节点【{node_name}】数据量为 {len(dataset)}，拒绝参与本轮联邦。")
                    continue

                valid_nodes_count += 1
                total_samples_this_round += len(dataset)
                participating_node_ids.append(node_id) # 记录有效参与的节点ID

                loader = DataLoader(dataset, batch_size=4, shuffle=True)

                FL_STATUS["current_stage"] = "本地训练"
                local_model = BareMetalUnet(num_classes=7).to(DEVICE)
                local_model.load_state_dict(copy.deepcopy(global_weights))
                optimizer = optim.Adam(local_model.parameters(), lr=1e-4)

                local_model.train()
                node_loss, node_acc, node_dice = 0, 0, 0

                for epoch in range(local_epochs):
                    for imgs, masks, lbls in loader:
                        imgs, masks, lbls = imgs.to(DEVICE), masks.to(DEVICE), lbls.to(DEVICE)
                        optimizer.zero_grad()
                        cls_out, seg_out = local_model(imgs)
                        loss = 0.5 * criterion_cls(cls_out, lbls) + 0.5 * criterion_seg(seg_out, masks)
                        loss.backward()
                        optimizer.step()

                        node_loss += loss.item()
                        node_acc += (cls_out.argmax(dim=1) == lbls).sum().item() / len(lbls)
                        node_dice += calculate_dice(seg_out, masks).item()

                total_batches = len(loader) * local_epochs
                round_loss += (node_loss / total_batches)
                round_acc += (node_acc / total_batches)
                round_dice += (node_dice / total_batches)

                local_updates.append({"weights": copy.deepcopy(local_model.state_dict()), "size": len(dataset)})

                FL_STATUS["current_stage"] = "权重上传"
                FL_STATUS["total_traffic_mb"] += MODEL_SIZE_MB
                write_log(f"[{time.strftime('%H:%M:%S')}] 接收【{node_name}】梯度包，上行流量: {MODEL_SIZE_MB}MB")

                del local_model
                local_model = None
                torch.cuda.empty_cache()

            # 中枢聚合
            if local_updates and total_samples_this_round > 0:
                FL_STATUS["current_stage"] = "中枢FedAvg聚合"
                FL_STATUS["current_node"] = "中枢网关"

                # 【核心逻辑】：记录本轮真正参与的节点到状态表中
                if "round_participants" not in FL_STATUS:
                    FL_STATUS["round_participants"] = {}
                FL_STATUS["round_participants"][str(r + 1)] = participating_node_ids

                fed_avg_weights = copy.deepcopy(local_updates[0]["weights"])
                for key in fed_avg_weights.keys():
                    fed_avg_weights[key] = torch.zeros_like(fed_avg_weights[key], dtype=torch.float)

                for update in local_updates:
                    weight_factor = update["size"] / total_samples_this_round
                    for key in fed_avg_weights.keys():
                        fed_avg_weights[key] += update["weights"][key].float() * weight_factor

                global_weights = fed_avg_weights

                # 计算并记录当前轮次指标
                if valid_nodes_count > 0:
                    cur_loss = round_loss / valid_nodes_count
                    cur_acc = round_acc / valid_nodes_count
                    cur_dice = round_dice / valid_nodes_count

                    FL_STATUS["metrics_history"][r] = {
                        "round": r + 1, "loss": cur_loss, "acc": cur_acc, "dice": cur_dice
                    }
                    write_log(f"====== 🔔 第 [{r+1}] 轮全局收敛指标 -> Loss: {cur_loss:.4f} | Acc: {cur_acc*100:.2f}% | Dice: {cur_dice:.4f} ======")

                    # 🏆 【全历史最优模型比对机制】：不再局限于单次，而是和 JSON 读取出来的最佳记录 PK
                    if cur_dice > FL_STATUS["best_metrics"]["dice"]:
                        FL_STATUS["best_metrics"] = {
                            "round": r + 1, "loss": cur_loss, "acc": cur_acc, "dice": cur_dice
                        }
                        best_pth = os.path.join(CHECKPOINT_DIR, "fl_model_best.pth")
                        torch.save(global_weights, best_pth)
                        write_log(f"🏆 [纪录刷新] 检出当前第 [{r+1}] 轮为【全历史最优】，硬核锁存权重包至: {best_pth}")

                # 【命名规范升级】：落盘细化模型版本，名称严格带入节点标识
                nodes_str = "_".join(participating_node_ids)
                round_pth_name = f"fl_model_round_{r+1}_nodes_{nodes_str}.pth"
                round_pth = os.path.join(CHECKPOINT_DIR, round_pth_name)
                torch.save(global_weights, round_pth)

                # 覆盖更新最新物理断点
                latest_pth_path = os.path.join(CHECKPOINT_DIR, "fl_checkpoint_latest.pth")
                torch.save(global_weights, latest_pth_path)

                # 触发断电保护机制：同步将进度状态写入磁盘 JSON
                save_progress_json()
            else:
                write_log(f"⚠️ 本轮无有效节点参与，跳过聚合。")

            del local_updates
            gc.collect()

        FL_STATUS["model_saved_path"] = os.path.join(CHECKPOINT_DIR, "fl_checkpoint_latest.pth")

    except Exception as e:
        write_log(f"❌ 训练中断发生严重突发错误: {e}")

    finally:
        # ==========================================
        # 👑 绝对无敌的显存与内存核弹级清扫机制
        # ==========================================
        FL_STATUS["is_training"] = False
        FL_STATUS["current_stage"] = "已完成"
        FL_STATUS["current_node"] = "无"
        write_log(f"[{time.strftime('%H:%M:%S')}] 🧹 协同调度收尾，正在物理销毁显存内的模型实例、梯度包与张量引用...")

        if 'global_model' in locals() and global_model is not None: del global_model
        if 'local_model' in locals() and local_model is not None: del local_model
        if 'global_weights' in locals() and global_weights is not None: del global_weights
        if 'fed_avg_weights' in locals() and fed_avg_weights is not None: del fed_avg_weights

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_log(f"[{time.strftime('%H:%M:%S')}] ✅ 内存/显存完全解绑释空，GPU 已退回冷待机状态。")