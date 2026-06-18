import sqlite3
import os


def init_db():
    # 确保数据库文件在当前目录
    db_path = os.path.join(os.path.dirname(__file__), 'db_node_003.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. patients（患者表）- 约束：身份证号唯一
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        gender TEXT,
        age INTEGER,
        id_card TEXT UNIQUE NOT NULL, 
        phone TEXT,
        address TEXT,
        allergy_history TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # 2. registrations（挂号表）
    cursor.execute('''CREATE TABLE IF NOT EXISTS registrations (
        reg_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        dept_name TEXT,
        doctor_name TEXT,
        reg_type TEXT,
        queue_num INTEGER,
        status TEXT DEFAULT '待诊', -- 状态：待诊、就诊中、已就诊、已取消
        reg_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )''')

    # 3. visits（就诊表）
    cursor.execute('''CREATE TABLE IF NOT EXISTS visits (
        visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        reg_id INTEGER,
        chief_complaint TEXT,
        diagnosis TEXT,
        visit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(reg_id) REFERENCES registrations(reg_id)
    )''')

    # 4. drugs（药品表）
    cursor.execute('''CREATE TABLE IF NOT EXISTS drugs (
        drug_id INTEGER PRIMARY KEY AUTOINCREMENT,
        drug_name TEXT NOT NULL,
        spec TEXT,
        unit TEXT,
        price REAL,
        stock INTEGER DEFAULT 0
    )''')

    # 5. prescriptions（处方表）
    cursor.execute('''CREATE TABLE IF NOT EXISTS prescriptions (
        pres_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id INTEGER,
        patient_id INTEGER,
        total_price REAL DEFAULT 0.0,
        status TEXT DEFAULT '已开方', -- 状态：已开方、已收费、已发药
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(visit_id) REFERENCES visits(visit_id)
    )''')

    # 6. prescription_items（处方明细表）
    cursor.execute('''CREATE TABLE IF NOT EXISTS prescription_items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pres_id INTEGER,
        drug_id INTEGER,
        quantity INTEGER,
        dosage TEXT,
        subtotal REAL,
        FOREIGN KEY(pres_id) REFERENCES prescriptions(pres_id),
        FOREIGN KEY(drug_id) REFERENCES drugs(drug_id)
    )''')

    # 7. payments（收费表）
    cursor.execute('''CREATE TABLE IF NOT EXISTS payments (
        payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pres_id INTEGER,
        amount REAL,
        pay_method TEXT,
        pay_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(pres_id) REFERENCES prescriptions(pres_id)
    )''')

    # 8. dispense_records（发药表）
    cursor.execute('''CREATE TABLE IF NOT EXISTS dispense_records (
        dispense_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pres_id INTEGER,
        dispenser_name TEXT,
        dispense_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(pres_id) REFERENCES prescriptions(pres_id)
    )''')

    # 9. pacs_dermatology（皮肤科影像与AI诊断表 - 联邦学习核心数据池）
    cursor.execute('''CREATE TABLE IF NOT EXISTS pacs_dermatology (
                                                                      record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                      patient_id INTEGER NOT NULL,
                                                                      visit_id INTEGER,
                                                                      image_id TEXT UNIQUE NOT NULL,
                                                                      image_path TEXT NOT NULL,
                                                                      mask_path TEXT,
                                                                      ground_truth TEXT NOT NULL,
                                                                      ai_prediction TEXT,
                                                                      ai_confidence REAL,
                                                                      upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                                      FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
        FOREIGN KEY(visit_id) REFERENCES visits(visit_id)
        )''')
    conn.commit()
    conn.close()
    print("数据库及其业务表初始化成功！")



if __name__ == "__main__":
    init_db()