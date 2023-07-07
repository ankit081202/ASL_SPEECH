import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from classifier import Classifier
from text_to_speech import *
import math
detector = HandDetector(maxHands=1)
classifier = Classifier("model/model.h5", "model/labels.txt")
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 250
text = ""
prev = 0
while True:
    ret, img = cap.read()
    hands, img = detector.findHands(img)
    imgWhite = img.copy()
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[0:imgResizeShape[0], wGap: wCal + wGap] = imgResize
            imgWhite = cv2.GaussianBlur(imgWhite, (5, 5), 0.5)
            prediction, index = classifier.getPrediction(imgWhite)
            prev = index
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            prev = index

        cv2.imshow('imgCrop', imgCrop)
        cv2.imshow('imgWhite', imgWhite)

    cv2.imshow('Video', imgWhite)
    key = cv2.waitKey(1)
    if key == ord("s"):
        text = text + chr(prev + 97)
    elif key == ord('e'):
        play(text)
        print(text)
        break
