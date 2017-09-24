import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt




cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    edges = cv2.Canny(frame, 100, 200)

    # Our operations on the frame come here


    # Display the resulting frame


    #cv2.imshow('edges', edges)

    ###########

    im = frame
    ret, thresh = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    #     if len(approx) == 4:
    #         print "square"
    #         img = cv2.drawContours(image, contours, 0, (0, 255, 0), 3)
    #         #cv2.imshow('img',img)

    lengths = [cv2.arcLength(c, True) for c in contours]
    if not(lengths == None):
        max_Index = np.argmax(lengths)
        cnt = contours[max_Index]
        img2 = cv2.drawContours(image, cnt, 0, (0, 255, 0), 3)
        cv2.imshow('img2',img2)




    ###########

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()