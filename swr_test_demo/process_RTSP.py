import sqlite3
import face_recognition.api as face_api  # 使用 face_recognition.api
import cv2
import numpy as np

# 从数据库加载已知人脸
def load_known_faces():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute('SELECT name, encoding FROM faces')
    
    known_face_encodings = []
    known_face_names = []
    
    for row in c.fetchall():
        known_face_names.append(row[0])
        encoding = np.frombuffer(row[1], dtype=np.float64)
        known_face_encodings.append(encoding)
        print(f"Loaded face: {row[0]}")  # 显示已加载的每个人脸名称

    conn.close()
    return known_face_encodings, known_face_names

# 动态调整曝光度
def adjust_exposure(video_capture, avg_brightness):
    if avg_brightness < 50:  # 低亮度环境
        print("Increasing exposure due to low brightness")
        video_capture.set(cv2.CAP_PROP_EXPOSURE, 0.1)  # 适当增加曝光度值
    elif avg_brightness > 200:  # 高亮度环境
        print("Decreasing exposure due to high brightness")
        video_capture.set(cv2.CAP_PROP_EXPOSURE, -3.0)  # 适当减少曝光度值
    else:
        print("Normal brightness detected, keeping current exposure")

# 主程序
def main():
    video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置分辨率宽度
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)  # 设置分辨率高度
    video_capture.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率为摄像头支持的30 FPS
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))  # 设置视频编码格式为 YUYV

    if not video_capture.isOpened():
        print("无法打开摄像头")
        return

    known_face_encodings, known_face_names = load_known_faces()

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    frame_count = 0  # 用于计数帧数

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("无法读取视频帧")
            break

        # 调整曝光度(暂不启用)
        if frame_count % 100 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_frame)
            print(f"Average brightness: {avg_brightness}")
            # adjust_exposure(video_capture, avg_brightness)  # 根据亮度调整曝光度

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB

            # 使用异常处理来避免崩溃
            try:
                face_locations = face_api.face_locations(rgb_small_frame)
                face_encodings = face_api.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
            except TypeError as e:
                print("Error computing face descriptor:", e)
                face_encodings = []  # 跳过当前帧的编码

            face_names = []
            for face_encoding in face_encodings:
                # 计算距离来确定匹配的最佳人脸
                face_distances = face_api.face_distance(known_face_encodings, face_encoding)
                print(f"Face distances: {face_distances}")

                best_match_index = np.argmin(face_distances)
                name = "Unknown"
                
                # 根据距离设定阈值
                if face_distances[best_match_index] < 0.5:  # 可以调整为更严格的值如 0.4
                    name = known_face_names[best_match_index]

                face_names.append(name)
                
                # 打印检测到的面部信息和距离
                print(f"Detected face: {name} with distance {face_distances[best_match_index]}")

        process_this_frame = not process_this_frame
        frame_count += 1  # 增加帧计数

        # 绘制每个人脸的 bounding box（包括已知和未知人脸）
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 绘制 bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # 已知人脸用绿色，陌生人用红色
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # 打印 bounding box 坐标
            print(f"Bounding box for {name}: Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")

        # 显示视频流
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
