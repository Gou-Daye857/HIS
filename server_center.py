# server_center.py
from flask import Flask, jsonify, request
import os
import time
import sqlite3
import logging  # 1. 引入 logging 模块

# 导入调度器蓝图 (确保 server_dispatcher.py 在同级目录)
from server_dispatcher import dispatcher_bp
from module_fl.fl_api import fl_bp

app = Flask(__name__)

# =====================================================================
# 🔇 新增：日志消音器 (屏蔽高频 UI 轮询日志，保持终端清爽)
# =====================================================================
class PollingLogFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        # 如果日志里包含以下高频轮询接口，就丢弃不打印
        if '/api/fl/status' in message or '/api/center/health' in message:
            return False
        return True

# 捕获 Flask 底层的 werkzeug 记录器，并给它戴上消音器
log = logging.getLogger('werkzeug')
log.addFilter(PollingLogFilter())
# =====================================================================

# 注册统一调度网关
app.register_blueprint(dispatcher_bp)
app.register_blueprint(fl_bp)

# 全局实例状态池 (维持你原有的设计，用于监控和FL)
CENTER_RESOURCES = {
    "instances": {}, # 存储各 NodeID 在线状态
    "fl_metrics": {"round": 0, "status": "Waiting"}
}

# 动态获取中心数据库的路径适配
DB_PATH = "./client_database/center_management.db" if os.path.exists("./client_database/center_management.db") else "center_management.db"

@app.route('/api/center/health', methods=['GET'])
def health():
    """管理中心连通性自检"""
    return jsonify({
        "status": "success",
        "center_db_online": os.path.exists(DB_PATH),
        "active_nodes": len(CENTER_RESOURCES["instances"])
    })

@app.route('/api/center/get_nodes', methods=['GET'])
def get_nodes():
    """【新增】动态获取已注册的合法医院节点列表"""
    try:
        if not os.path.exists(DB_PATH):
            return jsonify({"status": "error", "message": "中心数据库文件不存在"}), 404

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 只查询状态为 active 的节点
        cursor.execute("SELECT NodeID, NodeName FROM Registered_Nodes WHERE Status = 'active'")
        nodes = cursor.fetchall()
        conn.close()

        # 将查询结果打包成 JSON 列表
        node_list = [{"id": row[0], "name": row[1]} for row in nodes]
        return jsonify({"status": "success", "nodes": node_list})
    except Exception as e:
        return jsonify({"status": "error", "message": f"数据库查询失败: {str(e)}"}), 500

@app.route('/api/center/register_node', methods=['POST'])
def register():
    """实例上线登记 (心跳)"""
    node_id = request.json.get("node_id")
    CENTER_RESOURCES["instances"][node_id] = {
        "ip": request.remote_addr,
        "last_seen": time.time()
    }
    print(f">>> [实例调度] 节点 {node_id} 已上线并接入中枢")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # 启动后端中心
    app.run(host='0.0.0.0', port=5000)