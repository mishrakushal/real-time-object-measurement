import cv2
from utils import getContours, warpImage, reorder, findDistance

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

SCALE = 3

# Setting the dimension of the first contour to be that of a standard A4-size sheet
wPaper = 210 * SCALE
hPaper = 297 * SCALE

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

        # Shrink the image by factor of 2 so that it fits the frame
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    img, contours = getContours(img, thresholdArea=50000, filter=4)

    if len(contours) > 0:
        maxContour = contours[0]
        biggest = maxContour[2]
        
        warpImg = warpImage(img, biggest, wPaper, hPaper)
        cv2.imshow('A4', warpImg)

        imgIN, contoursIN = getContours(warpImg, thresholdArea=2000, filter=4,cannyThreshold=[50, 50], draw=False)

        if len(contoursIN) > 0:
            for obj in contoursIN:
                cv2.polylines(imgIN, [obj[2]], True, (0, 255, 0), 2)
                newPoints = reorder(obj[2])

                # we get width in millimeters
                widthMM = findDistance(newPoints[0][0] // SCALE, newPoints[1][0] // SCALE)

                # convert the width to centimeteres and round to 1 decimal place
                widthCM = round(widthMM / 10, 1)
                
                # we get the height in millimeters
                heightMM = findDistance(newPoints[0][0] // SCALE, newPoints[2][0] // SCALE)

                # convert the height to centimeteres and round to 1 decimal place
                heightCM = round(heightMM / 10, 1)

        cv2.imshow('A4', imgIN)

    cv2.imshow('Original', img)
    cv2.waitKey(1)