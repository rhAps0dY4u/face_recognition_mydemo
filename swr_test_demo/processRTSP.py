import sqlite3
import face_recognition
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
        # 将字节转换为numpy数组
        encoding = np.frombuffer(row[1], dtype=np.float64)
        known_face_encodings.append(encoding)

    conn.close()
    return known_face_encodings, known_face_names

# 主程序
def main():
    video_capture = cv2.VideoCapture("rtsp://本机ip:9555/ds-test")
    known_face_encodings, known_face_names = load_known_faces()

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("无法读取视频流")
            break

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
