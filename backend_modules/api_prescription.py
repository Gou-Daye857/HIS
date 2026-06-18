# backend_modules/api_prescription.py
import sqlite3

def handle_action(action, business_data, node_db_conn):
    if action == "get_pending_pres":
        return _get_pending_pres(node_db_conn)
    elif action == "get_drugs":
        return _get_drugs(node_db_conn)
    elif action == "submit_pres":
        return _submit_pres(node_db_conn, business_data)
    return {"status": "error", "message": f"不支持的操作: {action}"}

def _get_pending_pres(db_conn):
    """获取所有在医生工作站已就诊、等待开方(已开方初始化)的记录"""
    cursor = db_conn.cursor()
    cursor.execute('''
                   SELECT pr.pres_id, p.name, p.gender, p.age, v.diagnosis
                   FROM prescriptions pr
                            JOIN visits v ON pr.visit_id = v.visit_id
                            JOIN patients p ON pr.patient_id = p.patient_id
                   WHERE pr.status = '已开方'
                   ''')
    data = [{"pres_id": r[0], "name": r[1], "gender": r[2], "age": r[3], "diag": r[4]} for r in cursor.fetchall()]
    return {"status": "success", "data": data}

def _get_drugs(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT drug_id, drug_name, spec, price, stock FROM drugs WHERE stock > 0")
    data = [{"drug_id": r[0], "name": r[1], "spec": r[2], "price": r[3], "stock": r[4]} for r in cursor.fetchall()]
    return {"status": "success", "data": data}

def _submit_pres(db_conn, data):
    """将开具好的处方明细入库，并将状态推向未收费"""
    pres_id = data["pres_id"]
    cart_items = data["cart_items"]
    cursor = db_conn.cursor()
    try:
        total_price = 0.0
        for item in cart_items:
            # 插入处方明细
            cursor.execute("INSERT INTO prescription_items (pres_id, drug_id, quantity, dosage, subtotal) VALUES (?,?,?,?,?)",
                           (pres_id, item["drug_id"], item["qty"], item["dosage"], item["subtotal"]))
            total_price += float(item["subtotal"])

        # 更新主表金额与状态
        cursor.execute("UPDATE prescriptions SET total_price = ?, status = '未收费' WHERE pres_id = ?", (total_price, pres_id))
        return {"status": "success", "message": f"处方提交成功，总计 ￥{total_price:.2f}。请患者前往收费处。"}
    except Exception as e:
        return {"status": "error", "message": str(e)}