# server_dispatcher.py
import sqlite3
from flask import Blueprint, request, jsonify
# 核心改变：只对接与 client_interface 对等的后端面板调度程序
import api_interface as api_interface

dispatcher_bp = Blueprint('dispatcher', __name__)

def get_node_database(node_id):
    """根据 node_id 从中心数据库动态映射对应的物理业务库文件"""
    try:
        center_conn = sqlite3.connect("./client_database/center_management.db")
        cursor = center_conn.cursor()
        cursor.execute("SELECT DatabasePath FROM Registered_Nodes WHERE NodeID = ?", (node_id,))
        result = cursor.fetchone()
        center_conn.close()
        return result[0] if result else None
    except Exception as e:
        print(f"中枢元数据库查询错误: {e}")
        return None

@dispatcher_bp.route('/api/dispatch', methods=['POST'])
def handle_dispatch():
    """
    统一报文传输通道 (只负责根据 NodeID 穿透到对应的物理隔离数据库)
    """
    req_data = request.json
    node_id = req_data.get("node_id")
    module = req_data.get("module")
    action = req_data.get("action")
    business_data = req_data.get("data", {})

    if not all([node_id, module, action]):
        return jsonify({"status": "error", "message": "网关拒绝：缺少标准报文要素"}), 400

    # 1. 通道鉴权与物理库路径匹配
    db_path = get_node_database(node_id)
    if not db_path:
        return jsonify({"status": "error", "message": f"非法接入：未注册的节点 [{node_id}]"}), 403

    # 2. 建立该节点专属的隔离数据库实例连接
    try:
        node_db_conn = sqlite3.connect(db_path)
    except Exception as e:
        return jsonify({"status": "error", "message": f"节点物理隔离库连接失败: {str(e)}"}), 500

    # 3. 核心解耦点：将四要素报文整体直接向下透传给面板调度程序 (api_interface)
    try:
        response_data = api_interface.dispatch_to_module(module, action, business_data, node_db_conn)
        node_db_conn.commit() # 统一在此处做事务提交
        return jsonify(response_data)
    except Exception as e:
        node_db_conn.rollback() # 异常时统一回滚
        return jsonify({"status": "error", "message": f"报文在业务流转中发生故障: {str(e)}"}), 500
    finally:
        node_db_conn.close() # 统一释放连接池