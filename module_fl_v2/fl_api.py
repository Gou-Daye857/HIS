# module_fl/fl_api.py
import subprocess
import time
import os
from flask import Blueprint, jsonify, request
from server_dispatcher import get_node_database
from module_fl.fl_state import get_fl_status, update_fl_status, reset_fl_status, append_traffic_log

fl_bp = Blueprint('fl_bp', __name__)
active_processes = []

@fl_bp.route('/api/fl/start', methods=['POST'])
def start_fl():
    global active_processes
    status = get_fl_status()
    if status.get("is_training", False) or active_processes:
        return jsonify({"status": "error", "message": "联邦学习核心引擎正在后台高速运转中，请勿重复点火"}), 400

    req_data = request.json or {}
    rounds = req_data.get("rounds", 5)       # UI 传递的【总目标轮次】
    epochs = req_data.get("epochs", 1)
    nodes = req_data.get("nodes", [])
    is_resume = req_data.get("is_resume", False) # 前端明确传递：是恢复断点还是重头开始
    port = 8080

    if len(nodes) < 2:
        return jsonify({"status": "error", "message": "根据联邦通信标准，至少需要 2 个节点"}), 400

    try:
        if not is_resume:
            # 全新启动：格式化物理状态
            reset_fl_status()
            latest_pth = "./checkpoints/fl_checkpoint_latest.pth"
            if os.path.exists(latest_pth):
                try: os.remove(latest_pth)
                except: pass
            start_round = 0
        else:
            # 断点恢复：查出当前跑到了第几轮
            start_round = status.get("current_round", 0)

        # 提取医院名称供UI拓扑图使用
        active_node_names = [node.get("name", f"Node-{node.get('id')}") for node in nodes]

        # 🚨 同步全部前端所需的参数
        update_fl_status({
            "is_training": True,
            "total_rounds": rounds,
            "current_epochs": epochs,
            "active_nodes": active_node_names,
            "current_stage": "正在拉起分布式网关..."
        })
        append_traffic_log(f"🚀 核心引擎点火：总规划 {rounds} 轮，当前起跑点：第 {start_round} 轮")

        # 1. 拉起 Server 中枢进程
        server_cmd = ["python", "module_fl/fl_server.py",
                      "--rounds", str(rounds),
                      "--min_clients", str(len(nodes)),
                      "--port", str(port)]
        server_process = subprocess.Popen(server_cmd)
        active_processes.append(server_process)

        time.sleep(3) # 预留时间给 gRPC 端口监听

        # 2. 拉起各个医院 Client 边缘进程
        for node in nodes:
            db_path = get_node_database(node["id"])
            client_cmd = ["python", "module_fl/fl_client.py",
                          "--node_id", str(node["id"]),
                          "--db_path", db_path,
                          "--server_address", f"127.0.0.1:{port}",
                          "--epochs", str(epochs)]

            client_proc = subprocess.Popen(client_cmd)
            active_processes.append(client_proc)

        return jsonify({"status": "success", "message": f"成功唤醒中枢与 {len(nodes)} 家参训医院独立进程。"})

    except Exception as e:
        for p in active_processes:
            try: p.kill()
            except: pass
        active_processes.clear()
        update_fl_status({"is_training": False, "current_stage": f"启动失败：{str(e)}"})
        return jsonify({"status": "error", "message": str(e)}), 500

@fl_bp.route('/api/fl/status', methods=['GET'])
def get_fl_realtime_status():
    return jsonify(get_fl_status())

@fl_bp.route('/api/fl/stop', methods=['POST'])
def stop_fl():
    global active_processes
    for p in active_processes:
        try: p.kill()
        except: pass
    active_processes.clear()
    update_fl_status({"is_training": False, "current_stage": "已被安全挂起（中止）"})
    append_traffic_log("🛑 训练被系统管理层安全平稳挂起，物理内存已释放。")
    return jsonify({"status": "success", "message": "全网子进程已强制执行物理回收"})