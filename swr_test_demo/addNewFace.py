import sqlite3
import face_recognition
import numpy as np

def add_new_face(image_path, name):
    # 连接到SQLite数据库
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()

    # 加载人脸图像并获取编码
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]

    # 将编码转换为字节
    encoding_blob = encoding.tobytes()

    # 插入新的人脸记录
    c.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', (name, encoding_blob))
    conn.commit()
    conn.close()
    print(f"已添加人脸: {name}")

if __name__ == "__main__":
    while True:
        add_face = input("是否要添加新的人脸？(y/n): ")
        if add_face.lower() == 'y':
            name = input("请输入姓名: ")
            image_path = input("请输入人脸图像文件路径: ")
            add_new_face(image_path, name)
        else:
            break
