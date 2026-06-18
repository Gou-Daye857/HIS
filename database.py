# database.py
import sqlite3

DB_FILE = 'his_system.db'

def get_db_connection():
    """获取数据库连接，并设置返回字典格式"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # 这一步非常关键，它让查询结果可以直接转为 JSON 字典
    return conn