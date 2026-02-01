

import cv2

img=cv2.imread('D:\Pythor\OpenCV\Screenshot 2024-09-21 100336.png',1)
#img=cv2.resize(img,(400,200))# dai,rong
img=cv2.resize(img,(0,0),fx=0.5,fy=0.5) # thu theo ti le ban dau
img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Picture',img)
k=cv2.waitKey()

if k==ord('s'):
    cv2.imwrite('Anh_moi.png',img)