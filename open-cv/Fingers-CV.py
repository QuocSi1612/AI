import cv2
import os
import numpy as np

from jax.example_libraries.stax import serial

import hand as htm
import serial

ad=serial.Serial('COM6',9600)
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
FolderPath = "Fingers"
lst = os.listdir(FolderPath)
print(lst)

lst_2 = []
for i in lst:
    image = cv2.imread(f"{FolderPath}/{i}")
    lst_2.append(image)

detector = htm.handDetector(detectionCon=int(0.55))  # Giữ giá trị detectionCon là số thực

# Khởi tạo đối tượng ghi video
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0  # Số frame trên giây

# Tạo đối tượng VideoWriter với tên file "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while True:
    ret2, frame2 = cap2.read()
    ret, frame = cap1.read()

    height = int(cap1.get(4))
    width = int(cap1.get(3))

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
    h, w, c = lst_2[0].shape

    small_frame = detector.findHands(small_frame)
    lmList = detector.findPosition(small_frame, draw=False)

    fingersID = [4, 8, 12, 16, 20]
    fingersCount = []

    if len(lmList) != 0:
        # Kiểm tra ngón cái
        if len(lmList) > fingersID[0] and len(lmList) > (fingersID[0] - 1):
            if lmList[fingersID[0]][1] > lmList[fingersID[0] - 1][1]:
                fingersCount.append(1)
            else:
                fingersCount.append(0)

        # Kiểm tra 4 ngón còn lại
        for i in range(1, 5):
            if len(lmList) > fingersID[i] and len(lmList) > (fingersID[i] - 2):
                if lmList[fingersID[i]][2] < lmList[fingersID[i] - 2][2]:
                    fingersCount.append(1)
                else:
                    fingersCount.append(0)

    numFingers = fingersCount.count(1)  # Số ngón tay được giơ lên
    print(numFingers)

    if numFingers == 1:
        ad.write(bytes('ON1' + '\r', 'utf-8'))
    elif numFingers == 2:
        ad.write(bytes('ON2' + '\r', 'utf-8'))
    elif numFingers == 3:
        ad.write(bytes('ON3' + '\r', 'utf-8'))
    elif numFingers == 4:
        ad.write(bytes('ON4' + '\r', 'utf-8'))
    elif numFingers == 5:
        ad.write(bytes('ON5' + '\r', 'utf-8'))
    elif numFingers == 0:
        ad.write(bytes('OFF' + '\r', 'utf-8'))

    image = np.zeros(frame.shape, np.uint8)
    image[:height // 2, :width // 2] = small_frame
    image[height // 2:, width // 2:] = small_frame
    image[height // 2:, :width // 2] = small_frame2
    image[:height // 2, width // 2:] = small_frame2

    # Ghi lại khung hình hiện tại vào file video
    out.write(image)

    cv2.imshow('Fingers window', image)

    if cv2.waitKey(1) == ord('s'):
        break

# Giải phóng video ghi và các thiết bị camera
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
