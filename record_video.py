# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:54:35 2022

@author: 2021-11
"""

import cv2
 
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
width = 640
ret = cap.set(3, width)
ret = cap1.set(3, width)
height = 480
ret = cap.set(4, height)
ret = cap1.set(4, height)
 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
out1 = cv2.VideoWriter('out1.mp4', fourcc, 20.0, (width, height))
out2 = cv2.VideoWriter('out2.mp4', fourcc, 20.0, (width, height))
 
while cap.isOpened():
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if ret and ret1 is True:
        frame = cv2.resize(frame, (640, 480))
        frame1 = cv2.resize(frame1, (640, 480))
        
        out1.write(frame)
        out2.write(frame1)
 
        cv2.imshow('frame', frame)
        cv2.imshow('frame1', frame1)
 
    else:
        break
 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
 
cap.release()
cap1.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
