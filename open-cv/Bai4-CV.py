import cv2
import numpy as np

cap=cv2.VideoCapture(0)
cap2=cv2.VideoCapture(1)
while True:
    ret2,frame2=cap2.read()
    ret,frame=cap.read()
    height=int(cap.get(4))
    width=int(cap.get(3))

    small_frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    small_frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
    image=np.zeros(frame.shape,np.uint8)
    image[:height//2,:width//2]=small_frame
    image[height // 2:, width // 2:] = small_frame
    image[height // 2:, :width // 2] = small_frame2
    image[:height // 2, width // 2:] = small_frame2
    cv2.imshow('Window camera',image)
    if cv2.waitKey(1)==ord('s'):
        break
cap.release()
cv2.destroyWindow()