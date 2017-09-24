import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt




cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('unblemished.jpg', frame)
    #edges = cv2.Canny(frame, 100, 200)
    # cv2.imshow('edges', edges)

    ###########

    im = frame
    ret, thresh = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 175, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)\

    img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    #cv2.imshow("contours", img)

    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    #     if len(approx) == 4:
    #         print "square"
    img = cv2.drawContours(image, contours, 0, (0, 255, 0), 3)
    #cv2.imshow('img',img)

    # largeContour = None
    # for cnt in contours:
    #     if (cv2.arcLength(cnt, True) > cv2.arcLength(largeContour, True)):
    #         largeContour = cnt
    #     rect = cv2.minAreaRect(largeContour)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # im3 = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
    if (contours):
        rect = cv2.minAreaRect(contours[0])
        boxV = cv2.boxPoints(rect)
        print(cv2.boxPoints(rect))
        box = np.int0(boxV)
        im3 = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

    cv2.imshow('im3',im3)

    if (contours):
        xVals = []
        yVals = []
        maxX = cv2.boxPoints(rect)[0][0]
        maxY = cv2.boxPoints(rect)[0][1]
        minX = cv2.boxPoints(rect)[0][0]
        minY = cv2.boxPoints(rect)[0][1]


        xVals.append(cv2.boxPoints(rect)[0][0])
        yVals.append(cv2.boxPoints(rect)[0][1])
        xVals.append(cv2.boxPoints(rect)[1][0])
        yVals.append(cv2.boxPoints(rect)[1][1])
        xVals.append(cv2.boxPoints(rect)[2][0])
        yVals.append(cv2.boxPoints(rect)[2][1])
        xVals.append(cv2.boxPoints(rect)[3][0])
        yVals.append(cv2.boxPoints(rect)[3][1])

        for index in range(0,4):
            if (xVals[index] > maxX):
                maxX = xVals[index]
            if (xVals[index] < minX):
                minX = xVals[index]

        for index in range(0, 4):
            if (yVals[index] > maxY):
                maxY = yVals[index]
            if (yVals[index] < minY):
                minY = yVals[index]

        minX = int(minX)
        maxX = int(maxX)
        minY = int(minY)
        maxY = int(maxY)
        if (minX < 0):
            minX = 0
        if (minY < 0):
            minY = 0

        cropped = cv2.imread('unblemished.jpg')[minY:maxY , minX:maxX]
        if not (maxY-minY <= 0 or maxX-minX <= 0):
            cv2.imshow('hope',cropped)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite('image.jpg', cropped)
        break




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()