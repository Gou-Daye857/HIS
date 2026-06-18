import sqlite3
import csv
import os
import random
from faker import Faker
from datetime import datetime, timedelta

# 初始化 Faker（使用中文生成逼真患者数据）
fake = Faker('zh_CN')

# 设定的三个节点数据库文件名 (请确保它们在当前目录下已经由 init_db() 创建好)
NODE_DBS = ['db_node_001.db', 'db_node_002.db', 'db_node_003.db']
CSV_FILE = '/home/sunjingbo/py/HIS/client_database/GroundTruth_database.csv'
DISEASE_LABELS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

def get_node_for_label(label):
    """
    【联邦学习核心逻辑：Non-IID 数据分配】
    根据疾病标签分配节点，保证同一个样本只去一个节点（无重叠），
    但不同节点的数据类别极度不平衡，完美模拟真实医院的数据孤岛。
    """
    if label in ['MEL', 'BCC']:
        # 偏恶性肿瘤：70% 概率进入 NODE_001（模拟肿瘤专科医院）
        weights = [0.70, 0.10, 0.20]
    elif label == 'NV':
        # 普通黑色素痣：80% 概率进入 NODE_002（模拟社区门诊，良性最多）
        weights = [0.10, 0.80, 0.10]
    else:
        # 其他病种 (AKIEC, BKL, DF, VASC)：60% 概率进入 NODE_003（模拟综合医院皮肤科）
        weights = [0.20, 0.20, 0.60]

    return random.choices(NODE_DBS, weights=weights, k=1)[0]

def simulate_data():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到文件 {CSV_FILE}，请确保它在当前目录下。")
        return

    # 检查数据库文件是否存在
    for db in NODE_DBS:
        if not os.path.exists(db):
            print(f"❌ 找不到数据库文件 {db}，请先运行你的 init_db() 建表！")
            return

    # 1. 建立所有节点的数据库连接池
    connections = {db: sqlite3.connect(db) for db in NODE_DBS}
    counters = {db: 0 for db in NODE_DBS}

    # 跟踪已生成的身份证号，防止极其罕见的随机重复导致 UNIQUE 约束报错
    generated_id_cards = set()

    print("🚀 开始解析 GroundTruth.csv 并向现有表中插入联邦节点数据...")

    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = row['image']

            # 解析 One-Hot 编码得出实际疾病单标签 (如 'NV')
            actual_label = "UNKNOWN"
            for label in DISEASE_LABELS:
                if int(float(row.get(label, 0))) == 1:
                    actual_label = label
                    break

            # 2. 根据疾病标签，加权随机选择目标节点 (制造 Non-IID)
            target_db = get_node_for_label(actual_label)
            conn = connections[target_db]
            cursor = conn.cursor()

            # 3. 生成虚拟患者数据 (保证身份证号全局绝对唯一)
            while True:
                id_card = fake.ssn()
                if id_card not in generated_id_cards:
                    generated_id_cards.add(id_card)
                    break

            name = fake.name()
            gender = random.choice(['男', '女'])
            age = random.randint(15, 85)
            phone = fake.phone_number()
            address = fake.address()
            allergy = random.choice(['无', '无', '无', '青霉素过敏', '海鲜过敏', '酒精过敏'])

            # 随机生成过去两年的建档时间
            random_days = random.randint(0, 730)
            create_time = (datetime.now() - timedelta(days=random_days)).strftime("%Y-%m-%d %H:%M:%S")

            # 4. 插入患者表 (严格遵循你提供的字段)
            cursor.execute('''
                           INSERT INTO patients (name, gender, age, id_card, phone, address, allergy_history, create_time)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (name, gender, age, id_card, phone, address, allergy, create_time))

            # 获取刚刚由 SQLite 自动生成的 patient_id，作为外键
            patient_id = cursor.lastrowid

            # 5. 生成物理图片和掩码的相对路径 (假设图片和掩码存放在后端的 dataset/ 目录下)
            image_path = f"/data/sunjingbo/HAM10000/images/{image_id}.jpg"
            mask_path = f"/data/sunjingbo/HAM10000/masks/{image_id}_segmentation.png"

            # 6. 插入皮肤科影像表 (visit_id 留空，ai相关预留字段留空)
            cursor.execute('''
                           INSERT INTO pacs_dermatology (patient_id, visit_id, image_id, image_path, mask_path, ground_truth, ai_prediction, ai_confidence, upload_time)
                           VALUES (?, NULL, ?, ?, ?, ?, NULL, NULL, ?)
                           ''', (patient_id, image_id, image_path, mask_path, actual_label, create_time))

            counters[target_db] += 1

    # 7. 提交事务并关闭连接
    for db, conn in connections.items():
        conn.commit()
        conn.close()

    print("\n✅ 数据模拟与分发完成！")
    print("📊 联邦节点数据分布统计 (Non-IID 结果):")
    for db, count in counters.items():
        print(f"  - {db}: 成功插入 {count} 条独立的患者与影像记录")

if __name__ == '__main__':
    simulate_data()