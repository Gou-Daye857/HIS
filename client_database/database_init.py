# create_center_db.py (在后端运行一次即可)
import sqlite3

def init_center_db():
    conn = sqlite3.connect("center_management.db")
    cursor = conn.cursor()

    # 创建注册节点表
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS Registered_Nodes (
                                                                   NodeID VARCHAR(20) PRIMARY KEY,
                       NodeName VARCHAR(50),
                       DatabasePath VARCHAR(100),
                       Status VARCHAR(20)
                       )
                   ''')

    # 预先插入你的三个模拟医院节点，并为它们分配独立的业务数据库文件
    nodes_data = [
        ("NODE_001", "市第一医院", "db_node_001.db", "active"),
        ("NODE_002", "社区诊所A", "db_node_002.db", "active"),
        ("NODE_003", "专科医院B", "db_node_003.db", "active")
    ]

    cursor.executemany('''
                       INSERT OR IGNORE INTO Registered_Nodes (NodeID, NodeName, DatabasePath, Status)
        VALUES (?, ?, ?, ?)
                       ''', nodes_data)

    conn.commit()
    conn.close()
    print("中心管理数据库 (center_management.db) 初始化完成！各节点业务库已映射。")

if __name__ == "__main__":
    init_center_db()