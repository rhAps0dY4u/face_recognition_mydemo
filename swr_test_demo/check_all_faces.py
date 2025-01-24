import sqlite3
import numpy as np

def view_all_faces():
    # 连接到SQLite数据库
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()

    # 查询所有人脸记录
    c.execute('SELECT name, encoding FROM faces')
    rows = c.fetchall()

    # 打印所有记录
    for row in rows:
        name = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float64)  # 将存储的字节转换为人脸编码数组
        print(f"姓名: {name}, 编码: {encoding[:5]}...")  # 编码打印前5个值作为示例

    conn.close()

if __name__ == "__main__":
    view_all_faces()
