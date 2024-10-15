# Importing required packages
from keras.models import load_model         #để tải mô hình nhận diện cảm xúc đã được huấn luyện trước.
import numpy as np          # thực hiện các phép toán số học.
import argparse         #phân tích các tham số từ dòng lệnh.
import dlib     #phát hiện khuôn mặt và xác định các điểm đặc trưng trên khuôn mặt.
import cv2      #xử lý video và hình ảnh với OpenCV.   

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

#tham số dòng lệnh --isVideoWriter, tham số này quyết định xem video đầu ra có được lưu vào file hay không

emotion_offsets = (20, 40)
#Định nghĩa các cảm xúc có thể được nhận diện và màu sắc tương ứng của các khung bao quanh.
emotions = {
    0: {
        "emotion": "Angry",             #giận dữ
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",           #ghê tởm
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",              #sợ
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",             #vui
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",               #buồn
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",           #ngạc nhiên
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",           #bình thường
        "color": (108, 72, 200)
    }
}

#Nó sử dụng bộ phát hiện khuôn mặt dựa trên HOG 
#(Histogram of Oriented Gradients) và SVM (Support Vector Machine) để phát hiện khuôn mặt.

#Thư viện cũng bao gồm một mô hình dự đoán landmark khuôn mặt (68 điểm mốc),
#  giúp xác định vị trí các đặc trưng trên khuôn mặt như mắt, mũi, miệng.


def shapePoints(shape):         #Chuyển đổi các điểm đặc trưng khuôn mặt của dlib thành mảng numpy
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):           #Trích xuất tọa độ khung bao quanh từ một đối tượng hình chữ nhật của dlib
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

#mô hình dlib để phát hiện khuôn mặt và dự đoán các điểm đặc trưng.
faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)
#mô hình nhận diện cảm xúc đã được huấn luyện trước.
emotionModelPath = 'models/emotionModel.hdf5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

cap = cv2.VideoCapture(0)
#Khởi tạo trình ghi video nếu tham số --isVideoWriter là True.
if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22,
                                 (capWidth, capHeight))

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))
 #Đọc các khung hình từ webcam.
    if not ret:
        break
       #Chuyển đổi từng khung hình sang màu xám.
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue
#Phát hiện các khuôn mặt trong khung hình sử dụng dlib.
        grayFace = grayFace.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotion_prediction = emotionClassifier.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)
        #Trích xuất các điểm đặc trưng khuôn mặt và khung bao quanh.
        if (emotion_probability > 0.36):
            emotion_label_arg = np.argmax(emotion_prediction)
            color = emotions[emotion_label_arg]['color']
            #Trích xuất và tiền xử lý vùng khuôn mặt.
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #Dự đoán cảm xúc sử dụng mô hình đã được huấn luyện.
            cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                     color,
                     thickness=2)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40),
                          color, -1)
            cv2.putText(frame, emotions[emotion_label_arg]['emotion'],
                        (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            #Hiển thị khung hình đã xử lý.
        else:
            color = (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if args["isVideoWriter"] == True:
        videoWrite.write(frame)

    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
#phương pháp phát hiện khuôn mặt được triển khai trong tệp models.faceDetectionMethods giúp dễ dàng so sánh hiệu suất
cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
