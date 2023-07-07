import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(1)
offset = 20
imgSize = 250
folder = "data/validate/A"
counter = 0
while True:
    ret, img = cap.read()
    hands,img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[0:imgResizeShape[0], wGap : wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap : hCal + hGap, :] = imgResize

        cv2.imshow('imgCrop', imgCrop)
        cv2.imshow('imgWhite', imgWhite)

    cv2.imshow('Video',img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite(f'{folder}/{counter}.jpg',imgWhite)
        counter+=1
        print(counter)
        if counter > 20:
            counter = 0
            break