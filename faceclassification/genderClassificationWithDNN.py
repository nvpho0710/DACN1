# Importing required packages
from tensorflow.keras.models import load_model #mô hình phân loại giới tính đã được huấn luyện trước.
import numpy as np
import cv2 #xử lý video và hình ảnh với OpenCV.
import os       #thao tác với các đường dẫn file và kiểm tra sự tồn tại của file.

# Đường dẫn tới mô hình phân loại giới tính
genderModelPath = 'models/genderModel_VGG16.hdf5'
if not os.path.exists(genderModelPath):
    raise FileNotFoundError(f"Model file not found: {genderModelPath}")

#mô hình phân loại giới tính đã được huấn luyện trước.
genderClassifier = load_model(genderModelPath, compile=False)
genderTargetSize = genderClassifier.input_shape[1:3]

genders = {
    0: {
        "label": "Female",
        "color": (245, 215, 130)
    },
    1: {
        "label": "Male",
        "color": (148, 181, 192)
    },
}

# Đường dẫn tới mô hình phát hiện khuôn mặt
modelFile = "faceDetection/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "faceDetection/models/dnn/deploy.prototxt"

if not os.path.exists(modelFile) or not os.path.exists(configFile):
    raise FileNotFoundError("Face detection model files not found.")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#Hàm phát hiện khuôn mặt và phân loại giới tính trong một khung hình.
def detectFacesWithDNN(frame):
    size = (300, 300)
    scalefactor = 1.0
    swapRB = (104.0, 117.0, 123.0)

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor, size, swapRB)
    net.setInput(blob)
    dnnFaces = net.forward()
    
    for i in range(dnnFaces.shape[2]):
        confidence = dnnFaces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnnFaces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            resized = frame[max(0, y - 20):min(height, y1 + 30), max(0, x - 10):min(width, x1 + 10)]

            if resized.size == 0:
                continue

            try:
                frame_resize = cv2.resize(resized, genderTargetSize)
            except Exception as e:
                continue

            frame_resize = frame_resize.astype("float32") / 255.0
            frame_reshape = np.expand_dims(frame_resize, axis=0)
            gender_prediction = genderClassifier.predict(frame_reshape)
            gender_probability = np.max(gender_prediction)

            if gender_probability > 0.4:
                gender_label = np.argmax(gender_prediction)
                gender_result = genders[gender_label]["label"]
                color = genders[gender_label]["color"]
            else:
                gender_result = "Unknown"
                color = (255, 255, 255)

            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            cv2.rectangle(frame, (x + 20, y1 + 20), (x + 130, y1 + 55), color, -1)
            cv2.putText(frame, gender_result, (x + 25, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detectFacesWithDNN(frame)
    cv2.imshow("Gender Classification", frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:  
        break

cap.release()
cv2.destroyAllWindows()
