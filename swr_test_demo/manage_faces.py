import sqlite3
import face_recognition
import numpy as np
import os

# 显示数据库中的所有人脸名称
def display_all_faces():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()

    c.execute('SELECT name FROM faces')
    faces = c.fetchall()
    conn.close()

    print("\n当前已知人脸列表:")
    for i, (name,) in enumerate(faces, 1):
        print(f"{i}. {name}")

    return [name[0] for name in faces]  # 返回名称列表

# 添加人脸到数据库
def add_new_face(image_path, name):
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

# 删除人脸记录
def delete_face_by_index(index, names):
    name = names[index]
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()

    # 删除指定姓名的记录
    c.execute('DELETE FROM faces WHERE name = ?', (name,))
    conn.commit()
    conn.close()
    print(f"已删除人脸: {name}")

# 修改人脸姓名
def rename_face(old_name, new_name):
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()

    # 更新姓名
    c.execute('UPDATE faces SET name = ? WHERE name = ?', (new_name, old_name))
    conn.commit()
    conn.close()
    print(f"已将人脸名称从 {old_name} 修改为 {new_name}")

# 主函数
if __name__ == "__main__":
    while True:
        print("\n选择操作: ")
        print("1. 添加人脸")
        print("2. 删除人脸")
        print("3. 修改人脸名称")
        print("4. 退出")

        choice = input("请输入操作编号: ")

        if choice == '1':
            name = input("请输入姓名: ")
            image_path = input("请输入人脸图像文件路径: ")
            if os.path.exists(image_path):
                add_new_face(image_path, name)
                display_all_faces()  # 显示添加后的列表
            else:
                print("图像文件路径无效。")

        elif choice == '2':
            names = display_all_faces()  # 显示所有人脸名称并获取列表
            if names:
                try:
                    delete_index = int(input("请输入要删除的人脸编号: ")) - 1
                    if 0 <= delete_index < len(names):
                        delete_face_by_index(delete_index, names)
                        display_all_faces()  # 显示删除后的列表
                    else:
                        print("无效的编号。")
                except ValueError:
                    print("请输入有效的编号。")
            else:
                print("没有可删除的人脸。")

        elif choice == '3':
            names = display_all_faces()  # 显示所有人脸名称并获取列表
            if names:
                try:
                    rename_index = int(input("请输入要修改名称的人脸编号: ")) - 1
                    if 0 <= rename_index < len(names):
                        old_name = names[rename_index]
                        new_name = input("请输入新姓名: ")
                        rename_face(old_name, new_name)
                        display_all_faces()  # 显示修改后的列表
                    else:
                        print("无效的编号。")
                except ValueError:
                    print("请输入有效的编号。")
            else:
                print("没有可修改的人脸。")

        elif choice == '4':
            print("已退出程序")
            break

        else:
            print("无效的操作编号，请重新输入。")
