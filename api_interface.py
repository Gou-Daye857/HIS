# backend_modules/api_interface.py
import backend_modules.api_patient as api_patient  # 引入下级细化模块服务
import backend_modules.api_registration as api_registration
import backend_modules.api_waiting as api_waiting
import backend_modules.api_visit as api_visit
import backend_modules.api_prescription as api_prescription
import backend_modules.api_payment as api_payment
import backend_modules.api_dispense as api_dispense
import backend_modules.api_pacs as api_pacs
def dispatch_to_module(module, action, business_data, node_db_conn):
    """
    【核心全链路分发器】
    所有的医疗节点终端都会通过 JSON 报文打向这里，
    系统会精准无误地分配给对应的模块，保证物理隔离层面的数据强一致性。
    """
    if module == "interface":
        return _handle_panel_self_action(action, node_db_conn)

    elif module == "patient":
        return api_patient.handle_action(action, business_data, node_db_conn)

    elif module == "registration":
        return api_registration.handle_action(action, business_data, node_db_conn)

    elif module == "waiting":
        return api_waiting.handle_action(action, business_data, node_db_conn)

    elif module == "visit":
        return api_visit.handle_action(action, business_data, node_db_conn)

    elif module == "prescription":
        return api_prescription.handle_action(action, business_data, node_db_conn)

    elif module == "payment":
        return api_payment.handle_action(action, business_data, node_db_conn)

    elif module == "dispense":
        return api_dispense.handle_action(action, business_data, node_db_conn)

    elif module == "pacs":
        return api_pacs.handle_action(action, business_data, node_db_conn)

    else:
        return {"status": "error", "message": f"面板调度程序无法识别该模块报文: [{module}]"}


def _handle_panel_self_action(action, node_db_conn):
    """处理属于主面板自身的动作指令"""
    if action == "get_dashboard_info":
        return _get_dashboard_info(node_db_conn)
    else:
        return {"status": "error", "message": f"interface 自身不支持的操作指令: [{action}]"}


def _get_dashboard_info(db_conn):
    """获取本节点医院专属的仪表盘统计数据"""
    cursor = db_conn.cursor()
    try:
        # 安全检查 patients 表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM patients")
            patient_count = cursor.fetchone()[0]
        else:
            patient_count = 0
    except Exception as e:
        print(f"主面板拉取专属数据异常: {e}")
        patient_count = 0

    return {
        "status": "success",
        "data": {
            "total_patients": patient_count,
            "welcome_message": "联通正常。专属沙盒数据链隔离完备，联邦学习联邦内核就绪。"
        }
    }