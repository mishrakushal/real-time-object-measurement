import cv2
import numpy as np

webcam = False

# PATH FOR THE IMAGE IN CASE NO WEBCAM
path = "../img/1.jpg"

# PARAMETERS FOR VIDEO CAPTURE
cap = cv2.VideoCapture(0)

# Setting the brightness using id:10 and value 160
cap.set(10, 160)
# Setting the width using id:3 and value 1920
cap.set(3, 1920)
# Setting the width using id:4 and value 1080
cap.set(4, 1080)

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)
        # Shrink the image by factor of 2 so that it fits the frame
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Original', img)
        cv2.waitKey(1)