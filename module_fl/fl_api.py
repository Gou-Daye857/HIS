# module_fl/fl_api.py
import subprocess
import time
import os
import threading
from flask import Blueprint, jsonify, request
from server_dispatcher import get_node_database
from module_fl.fl_state import get_fl_status, update_fl_status, reset_fl_status, append_traffic_log

fl_bp = Blueprint('fl_bp', __name__)
active_processes = []
LOG_FILE_PATH = "fl_training_history.log"

def _unsafe_launch_processes(rounds, epochs, nodes, port=8080):
    global active_processes
    server_cmd = ["python", "module_fl/fl_server.py",
                  "--rounds", str(rounds),
                  "--min_clients", str(len(nodes)),
                  "--port", str(port)]
    server_process = subprocess.Popen(server_cmd)
    active_processes.append(server_process)

    time.sleep(3)

    for node in nodes:
        db_path = get_node_database(node["id"])
        client_cmd = ["python", "module_fl/fl_client.py",
                      "--node_id", str(node["id"]),
                      "--db_path", db_path,
                      "--server_address", f"127.0.0.1:{port}",
                      "--epochs", str(epochs)]

        client_proc = subprocess.Popen(client_cmd)
        active_processes.append(client_proc)

@fl_bp.route('/api/fl/start', methods=['POST'])
def start_fl():
    global active_processes
    status = get_fl_status()
    req_data = request.json or {}
    is_resume = req_data.get("is_resume", False)

    # 🛡️ 前端重连与防暴击拦截系统
    # 如果后台正在运行，且前端申请 resume（比如前端关掉重开，或者自检时请求），直接绿灯放行接管UI！
    if active_processes or status.get("is_training", False):
        if is_resume:
            return jsonify({"status": "success", "message": "已成功无缝接管后台运转中的联邦网络流！"})
        else:
            return jsonify({"status": "error", "message": "联邦训练引擎已在后台高速运转中，请勿重复点火"}), 400

    # 提取参数
    if is_resume:
        cfg = status.get("runtime_config", {})
        rounds = req_data.get("rounds") or cfg.get("rounds", 5)
        epochs = req_data.get("epochs") or cfg.get("epochs", 1)
        nodes = req_data.get("nodes") or cfg.get("nodes", [])
    else:
        rounds = req_data.get("rounds", 5)
        epochs = req_data.get("epochs", 1)
        nodes = req_data.get("nodes", [])

    port = 8080
    if len(nodes) < 2:
        return jsonify({"status": "error", "message": "至少需要 2 家医院参与联邦网络"}), 400

    try:
        if not is_resume:
            reset_fl_status()
            latest_pth = "./checkpoints/fl_checkpoint_latest.pth"
            if os.path.exists(latest_pth):
                try: os.remove(latest_pth)
                except: pass
            if os.path.exists(LOG_FILE_PATH):
                try: os.remove(LOG_FILE_PATH)
                except: pass
            start_round = 0
        else:
            # 严格依靠已保存落盘的真实历史数组计算断点！
            start_round = len(status.get("metrics_history", []))
            if start_round >= rounds:
                return jsonify({"status": "error", "message": "历史训练已达目标，无进度可恢复，请选择全新启动！"}), 400

        active_node_names = [node.get("name", f"Node-{node.get('id')}") for node in nodes]
        runtime_config = {"rounds": rounds, "epochs": epochs, "nodes": nodes, "port": port}

        update_fl_status({
            "is_training": True,
            "total_rounds": rounds,
            "current_epochs": epochs,
            "active_nodes": active_node_names,
            "current_stage": "正在拉起分布式组网网关...",
            "runtime_config": runtime_config
        })
        append_traffic_log(f"🚀 核心引擎点火：总规划 {rounds} 轮，当前起跑点：第 {start_round} 轮")

        _unsafe_launch_processes(rounds, epochs, nodes, port)

        return jsonify({"status": "success", "message": f"成功建立协同网络！已并发唤醒中枢与 {len(nodes)} 个自治子进程。"})

    except Exception as e:
        for p in active_processes:
            try: p.kill()
            except: pass
        active_processes.clear()
        update_fl_status({"is_training": False, "current_stage": f"启动失败：{str(e)}"})
        return jsonify({"status": "error", "message": str(e)}), 500

@fl_bp.route('/api/fl/status', methods=['GET'])
def get_fl_realtime_status():
    """彻底无缓存读盘：只要前端打开请求这个接口，必定获得最新的联邦状态"""
    return jsonify(get_fl_status())

@fl_bp.route('/api/fl/stop', methods=['POST'])
def stop_fl():
    global active_processes
    for p in active_processes:
        try: p.kill()
        except: pass
    active_processes.clear()
    update_fl_status({"is_training": False, "current_stage": "已被管理层主动挂起中止"})
    append_traffic_log("🛑 联邦系统被系统管理层安全平稳挂起，物理内存已完全回收。")
    return jsonify({"status": "success", "message": "全网子进程已强制执行物理回收解绑"})

@fl_bp.route('/api/fl/logs', methods=['GET'])
def get_fl_logs():
    try:
        status = get_fl_status()
        state_logs = status.get("traffic_logs", [])
        final_logs = []
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                final_logs = [line.strip() for line in f.readlines() if line.strip()]
        for log in state_logs:
            if log not in final_logs:
                final_logs.append(log)
        return jsonify({"status": "success", "logs": final_logs})
    except Exception as e:
        return jsonify({"status": "error", "message": f"日志解析失败: {str(e)}"}), 500

def _check_and_heal_orphaned_fl_session():
    """闪断自愈守护进程"""
    global active_processes
    time.sleep(2)
    status = get_fl_status()
    if status.get("is_training", False) and not active_processes:
        cfg = status.get("runtime_config", {})
        rounds = cfg.get("rounds")
        epochs = cfg.get("epochs")
        nodes = cfg.get("nodes")
        port = cfg.get("port", 8080)

        if rounds and nodes:
            start_round = len(status.get("metrics_history", []))
            if start_round >= rounds:
                update_fl_status({"is_training": False, "current_stage": "联邦训练圆满完成"})
                return

            update_fl_status({"current_stage": f"服务器闪断自愈中：正在拉起第 {start_round + 1} 轮进程..."})
            append_traffic_log(f"⚠️ 监测到主服务产生闪断，自愈引擎介入：正在追溯至第 {start_round} 轮断点...")
            try:
                _unsafe_launch_processes(rounds, epochs, nodes, port)
            except Exception as e:
                update_fl_status({"is_training": False, "current_stage": f"闪断自愈自修复失败: {e}"})
        else:
            update_fl_status({"is_training": False, "current_stage": "空闲"})

threading.Thread(target=_check_and_heal_orphaned_fl_session, daemon=True).start()