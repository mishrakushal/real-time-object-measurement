"""
Idea:
    - first extract the A4-sized paper itself
    - we use the standard dimensions of the paper to determine the dimensions of the contained objects
    - the function takes in an image and applies filters to it
    - it finds the positions of the objects within the image and
    - returns it in a suitable format
"""
import cv2
import numpy as np


"""
function to get the edges from the image
"""
def getContours(img, cannyThreshold=[100, 100], showCanny=False, thresholdArea=1000, filter=0, draw=False):
    # Convert the image to greyscale
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
        Apply Gaussian Blur to the image to remove the Gaussian noise
        Doing so will reduce the high-frequency components from the image
    """
    blurImg = cv2.GaussianBlur(greyImg, (5, 5), 1)

    # Get the edges from the grey-scale image using the Canny edge-detection algorithm
    cannyImg = cv2.Canny(blurImg, cannyThreshold[0], cannyThreshold[1])

    # We chose a 5x5 kernel with its center as the anchor
    kernel = np.ones((5, 5))

    """
        Apply dilation to the image to ensure thicker edges
        The dilate function will compute the maximum pixel brightness after overlapping original image with the kernel
        In doing so, the final image that we get has broader edges
    """
    dilateImg = cv2.dilate(cannyImg, kernel, iterations=3)

    """
        Apply erosions so that the edges become thin again
        The erode function will compute the minimum pixel brightness after overlapping original image with the kernel
        In doing so, the final image that we get has thinner edges
    """
    thresholdImg = cv2.erode(dilateImg, kernel, iterations=2)

    # Display if the  user wants to see the canny image
    if showCanny:
        cv2.imshow('Canny', cannyImg)

    """
        we need to find the EXTREME OUTER border of the shape in the current frame
        hence, we use RETR_EXTERNAL mode and
        CHAIN_APPROX_SIMPLE method to remove the redundant vertices along a single-line chain
    """
    contours, heirarchy = cv2.findContours(thresholdImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > thresholdArea:
            # Get the perimeter of the contour
            perimeter = cv2.arcLength(i, True)

            """
                We smoothen out the current contour using the approxPolyDP function but
                we try to keep the value of the approximation within 0.2% of the perimeter value
            """
            approx = cv2.approxPolyDP(i, 0.002 * perimeter, True)

            # Get the top-left (x, y) coordinates of the bounding-rectangle and its width and height
            boundingBox = cv2.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, boundingBox, i])
            else:
                finalContours.append([len(approx), area, approx, boundingBox, i])

    # Sort the contours array in decreasing order of the contour area
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)

    # Draw the image of the user wants the image to be drawn
    if draw:
        for contour in finalContours:
            # Add a red border to each shape
            cv2.drawContours(img, contour[4], -1, (0, 0, 255), 3)

    return img, finalContours


"""
    helps appropriately label the corners for processing
    eg: if 2->3->1->4 conv to 1->2->3->4 in clockwise order
"""
def reorder(points):
    newPoints = np.zeros_like(points)
    points = points.reshape((4, 2))
    add = points.sum(1)

    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)

    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints


def warpImage(img, points, w, h, pad=20):
    points = reorder(points)

    points1 = np.float32(points)
    points2 = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    matrix = cv2.getPerspectiveTransform(points1, points2)
    warpImg = cv2.warpPerspective(img, matrix, (w, h))

    # REMOVE PADDING FROM THE IMAGE, IF ANY
    warpImg = warpImg[pad:warpImg.shape[0] - pad, pad:warpImg.shape[1] - pad]

    return warpImg


"""
function to calculate the distance between two points
"""
def findDistance(points1, points2):
    x = points2[0] - points1[0]
    y = points2[1] - points1[1]

    return ((x ** 2) + (y ** 2)) ** 0.5

