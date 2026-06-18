# module_fl/fl_state.py

FL_STATUS = {
    "is_training": False,
    "current_round": 0,
    "total_rounds": 0,
    "current_epochs": 1,
    "current_stage": "空闲",
    "current_node": "无",
    "active_nodes": [],    # 记录参与训练的节点名称，供拓扑图渲染
    "metrics_history": [], # 格式: [{"round": 1, "loss": 0.5, "acc": 0.7, "dice": 0.6}, ...]
    "best_metrics": {      # 👑 历史最优性能指标锁 (全局跨断点比较)
        "round": 0,
        "loss": float('inf'),
        "acc": 0.0,
        "dice": 0.0
    },
    "round_participants": {}, # 🛡️ 双重保险：记录每轮实际参与训练的节点ID字典 {"1": ["NODE_001", "NODE_002"]}
    "traffic_logs": [],
    "total_traffic_mb": 0.0,
    "model_saved_path": ""
}