# module_fl/fl_api.py
import os
import json
import threading
import time
from flask import Blueprint, jsonify, request
from module_fl.fl_state import FL_STATUS
from module_fl.fl_core import federated_train_loop

fl_bp = Blueprint('fl_bp', __name__)
LOG_FILE_PATH = "fl_training_history.log"
PROGRESS_JSON = "./checkpoints/fl_progress.json"

@fl_bp.route('/api/fl/start', methods=['POST'])
def start_fl():
    if FL_STATUS["is_training"]:
        return jsonify({"status": "error", "message": "已有一个联邦学习任务在后台运行，请勿重复开启！"}), 400

    req_data = request.json or {}
    rounds = req_data.get("rounds", 5)
    epochs = req_data.get("epochs", 1)
    nodes = req_data.get("nodes", [])
    resume = req_data.get("resume", False) # 是否为断点恢复训练

    if not nodes:
        return jsonify({"status": "error", "message": "启动失败：无活跃节点"}), 400

    if resume:
        # 断点恢复模式：尝试从本地 progress.json 中恢复状态
        if not os.path.exists(PROGRESS_JSON):
            return jsonify({"status": "error", "message": "未检测到本地历史断点存档，无法恢复！"}), 400
        try:
            with open(PROGRESS_JSON, 'r', encoding='utf-8') as f:
                saved_status = json.load(f)

            FL_STATUS["metrics_history"] = saved_status.get("metrics_history", [])
            FL_STATUS["best_metrics"] = saved_status.get("best_metrics", {"round": 0, "loss": float('inf'), "acc": 0.0, "dice": 0.0})
            FL_STATUS["total_traffic_mb"] = saved_status.get("total_traffic_mb", 0.0)
            FL_STATUS["round_participants"] = saved_status.get("round_participants", {})
            start_round = saved_status.get("current_round", 0)

            FL_STATUS["traffic_logs"].append(f"[{time.strftime('%H:%M:%S')}] 🔄 成功读取断点存档，准备从第 [{start_round + 1}] 轮恢复训练...")
        except Exception as e:
            return jsonify({"status": "error", "message": f"断点文件解析失败: {e}"}), 500
    else:
        # 全新启动模式：重置所有运行时状态
        start_round = 0
        FL_STATUS["metrics_history"] = []
        FL_STATUS["best_metrics"] = {"round": 0, "loss": float('inf'), "acc": 0.0, "dice": 0.0}
        FL_STATUS["traffic_logs"] = []
        FL_STATUS["total_traffic_mb"] = 0.0
        FL_STATUS["round_participants"] = {}

    # 【核心修复】：真正的终点轮次 = 当前已跑轮次 + 界面请求轮次
    target_rounds = start_round + rounds

    FL_STATUS["is_training"] = True
    FL_STATUS["total_rounds"] = target_rounds
    FL_STATUS["current_epochs"] = epochs
    FL_STATUS["active_nodes"] = [node.get("name", f"Node-{node.get('id')}") for node in nodes]
    FL_STATUS["model_saved_path"] = ""

    # 拉起后台联邦线程，传入起始轮次和目标终点
    t = threading.Thread(target=federated_train_loop, args=(target_rounds, epochs, nodes, start_round))
    t.start()

    return jsonify({"status": "success", "message": "联邦训练引擎已拉起", "resume": resume})

@fl_bp.route('/api/fl/status', methods=['GET'])
def get_fl_status():
    return jsonify(FL_STATUS)

@fl_bp.route('/api/fl/logs', methods=['GET'])
def get_fl_logs():
    """读取后端安全日志文件"""
    if not os.path.exists(LOG_FILE_PATH):
        return jsonify({"status": "success", "logs": "暂无历史全局日志记录。"})
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"status": "success", "logs": content})
    except Exception as e:
        return jsonify({"status": "error", "message": f"读取日志文件失败: {e}"}), 500