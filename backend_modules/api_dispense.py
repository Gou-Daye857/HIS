# backend_modules/api_dispense.py
import sqlite3

def handle_action(action, business_data, node_db_conn):
    if action == "get_paid":
        return _get_paid(node_db_conn)
    elif action == "get_details":
        return _get_details(node_db_conn, business_data)
    elif action == "dispense":
        return _dispense(node_db_conn, business_data)
    return {"status": "error", "message": f"不支持的操作: {action}"}

def _get_paid(db_conn):
    cursor = db_conn.cursor()
    cursor.execute('''
                   SELECT pr.pres_id, p.name, pr.create_time
                   FROM prescriptions pr
                            JOIN patients p ON pr.patient_id = p.patient_id
                   WHERE pr.status = '已收费'
                   ''')
    data = [{"pres_id": r[0], "name": r[1], "time": r[2]} for r in cursor.fetchall()]
    return {"status": "success", "data": data}

def _get_details(db_conn, data):
    cursor = db_conn.cursor()
    cursor.execute('''
                   SELECT d.drug_name, d.spec, i.quantity, i.dosage
                   FROM prescription_items i
                            JOIN drugs d ON i.drug_id = d.drug_id
                   WHERE i.pres_id = ?
                   ''', (data["pres_id"],))
    data = [{"name": r[0], "spec": r[1], "qty": r[2], "dosage": r[3]} for r in cursor.fetchall()]
    return {"status": "success", "data": data}

def _dispense(db_conn, data):
    """发药闭环：扣减真实库存并完成业务流"""
    pres_id = data["pres_id"]
    cursor = db_conn.cursor()
    try:
        # 1. 扣库存
        cursor.execute("SELECT drug_id, quantity FROM prescription_items WHERE pres_id = ?", (pres_id,))
        items = cursor.fetchall()
        for drug_id, qty in items:
            cursor.execute("UPDATE drugs SET stock = stock - ? WHERE drug_id = ?", (qty, drug_id))

        # 2. 插入发药记录
        cursor.execute("INSERT INTO dispense_records (pres_id, dispenser_name) VALUES (?, '系统网关药师')", (pres_id,))
        # 3. 终结处方单生命周期
        cursor.execute("UPDATE prescriptions SET status = '已发药' WHERE pres_id = ?", (pres_id,))
        return {"status": "success", "message": "发药成功！库存已自动扣减，该患者就诊流程圆满结束。"}
    except Exception as e:
        return {"status": "error", "message": str(e)}