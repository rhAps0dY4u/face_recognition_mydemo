import sqlite3
import face_recognition
import numpy as np

# 创建或连接到SQLite数据库
conn = sqlite3.connect('face_recognition.db')
c = conn.cursor()

# 创建表格存储人脸编码和姓名
c.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
''')
conn.commit()
conn.close()
