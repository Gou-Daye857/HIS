# backend_modules/api_payment.py
import sqlite3

def handle_action(action, business_data, node_db_conn):
    if action == "get_unpaid":
        return _get_unpaid(node_db_conn)
    elif action == "pay":
        return _pay(node_db_conn, business_data)
    return {"status": "error", "message": f"不支持的操作: {action}"}

def _get_unpaid(db_conn):
    cursor = db_conn.cursor()
    cursor.execute('''
                   SELECT pr.pres_id, p.name, pr.total_price, pr.create_time
                   FROM prescriptions pr
                            JOIN patients p ON pr.patient_id = p.patient_id
                   WHERE pr.status = '未收费'
                   ''')
    data = [{"pres_id": r[0], "name": r[1], "amount": r[2], "time": r[3]} for r in cursor.fetchall()]
    return {"status": "success", "data": data}

def _pay(db_conn, data):
    """资金入账闭环"""
    pres_id, method, amount = data["pres_id"], data["method"], data["amount"]
    cursor = db_conn.cursor()
    try:
        cursor.execute("INSERT INTO payments (pres_id, amount, pay_method) VALUES (?,?,?)", (pres_id, amount, method))
        cursor.execute("UPDATE prescriptions SET status = '已收费' WHERE pres_id = ?", (pres_id,))
        return {"status": "success", "message": "缴费成功！系统已通知药房配药。"}
    except Exception as e:
        return {"status": "error", "message": str(e)}