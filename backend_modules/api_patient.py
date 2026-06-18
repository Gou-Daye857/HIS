# backend_modules/api_patient.py
import sqlite3

def handle_action(action, business_data, node_db_conn):
    """
    患者管理模块后端核心分发器
    :param action: 操作指令
    :param business_data: 前端提供的数据体
    :param node_db_conn: 对应物理节点的专属 SQLite 连接对象
    """
    # 1. 确保表结构存在 (模块自愈)
    _ensure_table_exists(node_db_conn)

    # 2. 路由到具体操作
    if action == "get_all_patients":
        return _get_all_patients(node_db_conn)
    elif action == "search_patients":
        return _search_patients(node_db_conn, business_data)
    elif action == "add_patient":
        return _add_patient(node_db_conn, business_data)
    elif action == "update_patient":
        return _update_patient(node_db_conn, business_data)
    elif action == "delete_patient":
        return _delete_patient(node_db_conn, business_data)
    else:
        return {"status": "error", "message": f"patient 模块不支持的操作: {action}"}

def _ensure_table_exists(db_conn):
    cursor = db_conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS patients (
                                                           patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                           name VARCHAR(50) NOT NULL,
                       gender VARCHAR(10),
                       age INTEGER,
                       id_card VARCHAR(20) UNIQUE,
                       phone VARCHAR(20),
                       allergy_history TEXT,
                       create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                   ''')
    db_conn.commit()

def _get_all_patients(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT patient_id, name, gender, age, id_card, phone, allergy_history FROM patients ORDER BY patient_id DESC")
    rows = cursor.fetchall()

    data_list = []
    for r in rows:
        data_list.append({
            "patient_id": r[0], "name": r[1], "gender": r[2],
            "age": r[3], "id_card": r[4], "phone": r[5], "allergy_history": r[6]
        })
    return {"status": "success", "data": data_list}

def _search_patients(db_conn, data):
    kw = data.get("keyword", "")
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT patient_id, name, gender, age, id_card, phone, allergy_history FROM patients WHERE name LIKE ? OR id_card LIKE ?",
        (f"%{kw}%", f"%{kw}%")
    )
    rows = cursor.fetchall()
    data_list = []
    for r in rows:
        data_list.append({
            "patient_id": r[0], "name": r[1], "gender": r[2],
            "age": r[3], "id_card": r[4], "phone": r[5], "allergy_history": r[6]
        })
    return {"status": "success", "data": data_list}

def _add_patient(db_conn, data):
    cursor = db_conn.cursor()
    try:
        cursor.execute('''
                       INSERT INTO patients (name, gender, age, id_card, phone, allergy_history)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (data["name"], data["gender"], data["age"], data["id_card"], data["phone"], data["allergy_history"]))
        return {"status": "success", "message": f"患者【{data['name']}】电子健康档案建立成功"}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "错误：该身份证号已被登记，不可重复建档！"}

def _update_patient(db_conn, data):
    cursor = db_conn.cursor()
    try:
        cursor.execute('''
                       UPDATE patients
                       SET name=?, gender=?, age=?, id_card=?, phone=?, allergy_history=?
                       WHERE patient_id=?
                       ''', (data["name"], data["gender"], data["age"], data["id_card"], data["phone"], data["allergy_history"], data["patient_id"]))
        return {"status": "success", "message": "患者健康档案更新成功"}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "错误：修改后的身份证号与其他患者冲突！"}

def _delete_patient(db_conn, data):
    cursor = db_conn.cursor()
    cursor.execute("DELETE FROM patients WHERE patient_id = ?", (data["patient_id"],))
    return {"status": "success", "message": "档案注销成功"}