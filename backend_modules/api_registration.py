# backend_modules/api_registration.py
import sqlite3

def handle_action(action, business_data, node_db_conn):
    """
    挂号管理模块后端核心分发器 (严格适配真实数据库字段)
    """
    if action == "search_patient_for_reg":
        return _search_patient(node_db_conn, business_data)
    elif action == "add_registration":
        return _add_registration(node_db_conn, business_data)
    elif action == "get_registrations":
        return _get_registrations(node_db_conn)
    elif action == "cancel_registration":
        return _cancel_registration(node_db_conn, business_data)
    else:
        return {"status": "error", "message": f"registration 模块不支持的操作: {action}"}

def _search_patient(db_conn, data):
    """用于挂号时的患者检索"""
    kw = data.get("keyword", "").strip()
    cursor = db_conn.cursor()

    try:
        # 严格匹配 patients 表字段
        cursor.execute(
            "SELECT patient_id, name, gender, age, phone, id_card FROM patients WHERE phone = ? OR id_card = ?",
            (kw, kw)
        )
        row = cursor.fetchone()

        if row:
            patient_data = {
                "patient_id": row[0], "name": row[1], "gender": row[2],
                "age": row[3], "phone": row[4], "id_card": row[5]
            }
            return {"status": "success", "data": patient_data}
        else:
            return {"status": "error", "message": "未找到匹配的患者档案，请核对证件号或先进行建档！"}
    except Exception as e:
        return {"status": "error", "message": f"查询患者档案失败: {str(e)} (可能是未初始化数据库)"}

def _add_registration(db_conn, data):
    """生成挂号单"""
    cursor = db_conn.cursor()
    try:
        # 自动生成当天的科室排队号 (queue_num)
        cursor.execute("SELECT COUNT(*) FROM registrations WHERE dept_name = ? AND date(reg_time) = date('now')", (data["dept_name"],))
        current_count = cursor.fetchone()[0]
        queue_num = current_count + 1

        # 严格遵守 registrations 字段: patient_id, dept_name, doctor_name, reg_type, queue_num
        cursor.execute('''
                       INSERT INTO registrations (patient_id, dept_name, doctor_name, reg_type, queue_num, status)
                       VALUES (?, ?, ?, ?, ?, '待诊')
                       ''', (data["patient_id"], data["dept_name"], data["doctor_name"], data["reg_type"], queue_num))

        return {"status": "success", "message": f"挂号成功！排队号为: {queue_num} 号，请引导患者候诊。"}
    except Exception as e:
        return {"status": "error", "message": f"挂号写入失败: {str(e)}"}

def _get_registrations(db_conn):
    """获取挂号列表 (核心：使用 JOIN 跨表获取姓名)"""
    cursor = db_conn.cursor()
    try:
        # 使用 JOIN 关联 registrations(r) 和 patients(p) 表
        cursor.execute('''
                       SELECT r.reg_id, p.name, r.dept_name, r.doctor_name, r.reg_type, r.queue_num, r.reg_time, r.status
                       FROM registrations r
                                JOIN patients p ON r.patient_id = p.patient_id
                       ORDER BY r.reg_id DESC
                       ''')
        rows = cursor.fetchall()

        data_list = []
        for r in rows:
            data_list.append({
                "reg_id": r[0], "patient_name": r[1], "dept_name": r[2],
                "doctor_name": r[3], "reg_type": r[4], "queue_num": r[5],
                "reg_time": r[6], "status": r[7]
            })
        return {"status": "success", "data": data_list}
    except Exception as e:
        # 如果表为空或还未建表，不抛出500，优雅返回空列表
        return {"status": "success", "data": []}

def _cancel_registration(db_conn, data):
    """处理退号逻辑"""
    reg_id = data.get("reg_id")
    cursor = db_conn.cursor()
    cursor.execute("UPDATE registrations SET status = '已取消' WHERE reg_id = ?", (reg_id,))
    return {"status": "success", "message": "挂号单已成功取消并标记。"}