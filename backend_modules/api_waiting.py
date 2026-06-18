# backend_modules/api_waiting.py
import sqlite3

def handle_action(action, business_data, node_db_conn):
    if action == "get_queue":
        return _get_queue(node_db_conn)
    elif action == "update_status":
        return _update_status(node_db_conn, business_data)
    else:
        return {"status": "error", "message": f"waiting 模块不支持的操作: {action}"}

def _get_queue(db_conn):
    """获取候诊队列，包含待诊、优先待诊、过号，并按优先级排序"""
    cursor = db_conn.cursor()
    try:
        # 跨表联查获取姓名，使用 CASE WHEN 进行优先级排序：优先待诊(1) > 待诊(2) > 过号(3)
        cursor.execute('''
                       SELECT r.queue_num, p.name, r.dept_name, r.doctor_name, r.reg_type, r.status, r.reg_id
                       FROM registrations r
                                JOIN patients p ON r.patient_id = p.patient_id
                       WHERE r.status IN ('待诊', '优先待诊', '过号') AND date(r.reg_time) = date('now')
                       ORDER BY
                           CASE r.status WHEN '优先待诊' THEN 1 WHEN '待诊' THEN 2 ELSE 3 END,
                r.queue_num ASC
                       ''')
        rows = cursor.fetchall()
        data_list = []
        for r in rows:
            data_list.append({
                "queue_num": r[0], "patient_name": r[1], "dept_name": r[2],
                "doctor_name": r[3], "reg_type": r[4], "status": r[5], "reg_id": r[6]
            })
        return {"status": "success", "data": data_list}
    except Exception as e:
        return {"status": "success", "data": []} # 如果表没建好，返回空队列

def _update_status(db_conn, data):
    """更新挂号单的就诊状态"""
    reg_id = data.get("reg_id")
    new_status = data.get("status")
    cursor = db_conn.cursor()
    try:
        cursor.execute("UPDATE registrations SET status = ? WHERE reg_id = ?", (new_status, reg_id))
        return {"status": "success", "message": f"状态已更新为 {new_status}"}
    except Exception as e:
        return {"status": "error", "message": f"状态更新失败: {str(e)}"}