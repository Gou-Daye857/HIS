# backend_modules/api_visit.py
import sqlite3

def handle_action(action, business_data, node_db_conn):
    if action == "get_waiting_patients":
        return _get_waiting_patients(node_db_conn)
    elif action == "get_patient_history":
        return _get_patient_history(node_db_conn, business_data)
    elif action == "save_visit":
        return _save_visit(node_db_conn, business_data)
    elif action == "get_drugs":
        return _get_drugs(node_db_conn)
    elif action == "add_pres_item":
        return _add_pres_item(node_db_conn, business_data)
    elif action == "get_pres_items":
        return _get_pres_items(node_db_conn, business_data)
    else:
        return {"status": "error", "message": f"visit 模块不支持: {action}"}

def _get_waiting_patients(db_conn):
    """查询已被分诊台叫号，状态为'就诊中'的患者"""
    cursor = db_conn.cursor()
    try:
        cursor.execute('''
                       SELECT r.reg_id, r.patient_id, p.name, p.gender, p.age, p.allergy_history
                       FROM registrations r
                                JOIN patients p ON r.patient_id = p.patient_id
                       WHERE r.status = '就诊中'
                       ''')
        data = [{"reg_id": r[0], "patient_id": r[1], "name": r[2], "gender": r[3], "age": r[4], "allergy": r[5]} for r in cursor.fetchall()]
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "success", "data": []}

def _get_patient_history(db_conn, data):
    """获取该患者过往的就诊病历"""
    patient_id = data.get("patient_id")
    cursor = db_conn.cursor()
    cursor.execute('''
                   SELECT v.visit_time, r.dept_name, v.diagnosis
                   FROM visits v
                            JOIN registrations r ON v.reg_id = r.reg_id
                   WHERE r.patient_id = ?
                   ORDER BY v.visit_time DESC
                   ''', (patient_id,))
    data = [{"time": r[0], "dept": r[1], "diag": r[2]} for r in cursor.fetchall()]
    return {"status": "success", "data": data}

def _save_visit(db_conn, data):
    """保存病历 -> 结束就诊 -> 创建空白处方单 (原子事务)"""
    cursor = db_conn.cursor()
    try:
        # 1. 插入病历
        cursor.execute("INSERT INTO visits (reg_id, chief_complaint, diagnosis) VALUES (?,?,?)",
                       (data["reg_id"], data["complaint"], data["diag"]))
        visit_id = cursor.lastrowid

        # 2. 更新挂号单状态为已就诊
        cursor.execute("UPDATE registrations SET status='已就诊' WHERE reg_id=?", (data["reg_id"],))

        # 3. 初始化对应的空白处方主表
        cursor.execute("INSERT INTO prescriptions (visit_id, patient_id, total_price, status) VALUES (?, ?, 0.0, '已开方')",
                       (visit_id, data["patient_id"]))
        pres_id = cursor.lastrowid

        return {"status": "success", "message": "病历已保存，请在右侧开具处方", "data": {"pres_id": pres_id}}
    except Exception as e:
        return {"status": "error", "message": f"病历保存失败: {str(e)}"}

def _get_drugs(db_conn):
    """获取药品字典库"""
    cursor = db_conn.cursor()
    try:
        cursor.execute("SELECT drug_id, drug_name, spec, price FROM drugs WHERE stock > 0")
        data = [{"drug_id": r[0], "name": r[1], "spec": r[2], "price": r[3]} for r in cursor.fetchall()]
        return {"status": "success", "data": data}
    except Exception:
        # 如果药品表没数据或没建好
        return {"status": "success", "data": []}

def _add_pres_item(db_conn, data):
    """向处方中添加药品"""
    cursor = db_conn.cursor()
    try:
        pres_id = data["pres_id"]
        drug_id = data["drug_id"]
        quantity = int(data["quantity"])
        dosage = data["dosage"]

        # 1. 查询该药的单价计算小计
        cursor.execute("SELECT price FROM drugs WHERE drug_id = ?", (drug_id,))
        price = cursor.fetchone()[0]
        subtotal = float(price) * quantity

        # 2. 插入明细表
        cursor.execute("INSERT INTO prescription_items (pres_id, drug_id, quantity, dosage, subtotal) VALUES (?,?,?,?,?)",
                       (pres_id, drug_id, quantity, dosage, subtotal))

        # 3. 更新主处方表的总金额
        cursor.execute("UPDATE prescriptions SET total_price = total_price + ? WHERE pres_id = ?", (subtotal, pres_id))

        return {"status": "success", "message": "药品已添加"}
    except Exception as e:
        return {"status": "error", "message": f"开药失败: {str(e)}"}

def _get_pres_items(db_conn, data):
    """获取当前处方的药品明细列表"""
    cursor = db_conn.cursor()
    cursor.execute('''
                   SELECT d.drug_name, d.spec, d.price, i.quantity, i.dosage, i.subtotal
                   FROM prescription_items i
                            JOIN drugs d ON i.drug_id = d.drug_id
                   WHERE i.pres_id = ?
                   ''', (data["pres_id"],))
    items = [{"name": r[0], "spec": r[1], "price": r[2], "qty": r[3], "dosage": r[4], "subtotal": r[5]} for r in cursor.fetchall()]
    return {"status": "success", "data": items}