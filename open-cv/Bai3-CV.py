from random import randint

import cv2
from numpy.random import random

img=cv2.imread('Screenshot 2024-09-21 100336.png',1)

# cv2.imshow('Hinh anh',img)
print(img)
print(type(img))

print(img.shape)
# print(img[499]) xuat dong cuoi cung
"""for i in range (img.shape[0]):
    for j in range(img.shape[1]):
        img[i][j]=[randint(0,255),randint(0,255),randint(0,255)]"""
vungCopy=img[0:100,300:400]
img[100:200,400:500]=vungCopy
cv2.imshow("Image",img)

cv2.waitKey()
