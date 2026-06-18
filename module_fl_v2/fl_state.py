# module_fl/fl_state.py
import os
import json
import time

STATE_FILE = "./checkpoints/fl_progress.json"

# 严格对齐前端 UI 期望的全部字段名，解决 KeyError 崩溃
DEFAULT_STATE = {
    "is_training": False,
    "current_round": 0,
    "total_rounds": 0,          # 🚨 修复重点：原来是 target_rounds，UI需要 total_rounds
    "current_epochs": 1,
    "current_stage": "空闲",
    "current_node": "无",
    "active_nodes": [],         # 供 UI 拓扑图渲染参与的医院名称
    "total_traffic_mb": 0.0,
    "metrics_history": [],
    "best_metrics": {"round": 0, "loss": 999.0, "acc": 0.0, "dice": 0.0},
    "round_participants": {},
    "traffic_logs": [],         # 供 UI 面板下方黑色终端滚动显示
    "model_saved_path": ""
}

def _read_state():
    """安全读取物理状态文件"""
    if not os.path.exists(STATE_FILE):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        _write_state(DEFAULT_STATE)
        return DEFAULT_STATE
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_STATE

def _write_state(state):
    """采用临时文件原子替换机制，防止多进程并发写入导致物理文件损坏"""
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp_file = STATE_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        os.replace(tmp_file, STATE_FILE)
    except Exception as e:
        pass

def get_fl_status():
    return _read_state()

def update_fl_status(updates: dict):
    state = _read_state()
    state.update(updates)
    _write_state(state)

def append_traffic_log(msg):
    """专门用于向前端终端追加带时间戳的日志"""
    state = _read_state()
    time_str = time.strftime('%H:%M:%S')
    full_msg = f"[{time_str}] {msg}"
    logs = state.get("traffic_logs", [])
    logs.append(full_msg)
    # 限制日志长度防止传输卡顿
    if len(logs) > 150:
        logs = logs[-150:]
    state["traffic_logs"] = logs
    _write_state(state)

def reset_fl_status():
    _write_state(DEFAULT_STATE)