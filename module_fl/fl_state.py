# module_fl/fl_state.py
import os
import json
import time
import uuid

STATE_FILE = "./checkpoints/fl_progress.json"

DEFAULT_STATE = {
    "is_training": False,
    "current_round": 0,
    "total_rounds": 0,
    "current_epochs": 1,
    "current_stage": "空闲",
    "current_node": "无",
    "active_nodes": [],
    "total_traffic_mb": 0.0,
    "metrics_history": [],
    "best_metrics": {"round": 0, "loss": 999.0, "acc": 0.0, "dice": 0.0},
    "round_participants": {},
    "traffic_logs": [],
    "model_saved_path": "",
    "runtime_config": {
        "rounds": 5,
        "epochs": 1,
        "nodes": [],
        "port": 8080
    }
}

def _read_state():
    """彻底废弃内存缓存：强制物理读盘，配合高频微秒级重试，解决 Windows 并发锁问题"""
    if not os.path.exists(STATE_FILE):
        _write_state(DEFAULT_STATE)
        return DEFAULT_STATE.copy()

    for _ in range(50):  # 最高允许 2.5 秒的并发等待
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.05)

    return DEFAULT_STATE.copy()

def _write_state(state):
    """绝对安全的原子写入：保证中枢随时断电，状态文件也绝不损坏"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    # 使用 uuid 确保多进程下临时文件名绝对不冲突
    tmp_file = f"{STATE_FILE}.{uuid.uuid4().hex}.tmp"

    for _ in range(50):
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
            os.replace(tmp_file, STATE_FILE) # OS 级别的原子替换
            return
        except (PermissionError, OSError):
            time.sleep(0.05)
        finally:
            if os.path.exists(tmp_file):
                try: os.remove(tmp_file)
                except: pass

def get_fl_status():
    return _read_state()

def update_fl_status(updates: dict):
    state = _read_state()
    state.update(updates)
    _write_state(state)

def append_traffic_log(msg):
    state = _read_state()
    time_str = time.strftime('%H:%M:%S')
    full_msg = f"[{time_str}] {msg}"

    logs = state.get("traffic_logs", [])
    logs.append(full_msg)
    if len(logs) > 150:
        logs = logs[-150:]
    state["traffic_logs"] = logs
    _write_state(state)

    try:
        with open("fl_training_history.log", "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
    except:
        pass

def reset_fl_status():
    _write_state(DEFAULT_STATE)