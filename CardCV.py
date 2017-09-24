import requests
import cv2
import numpy as np
import json
import heapq

url = "https://southcentralus.api.cognitive.microsoft.com/customvision/v1.0/Prediction/2a4d1acc-9398-42e8-b762-c024e3cb0292/image?iterationId=8792e79d-d251-4a4c-b53a-accbe4663052"
querystring = {"iterationId": "8792e79d-d251-4a4c-b53a-accbe4663052"}
headers = {
    'prediction-key': "4720d0607b2b4526bd4807c37ff9214d",
    'cache-control': "no-cache",
    'postman-token': "bb68f26e-4c00-c4d9-3fd2-c72abc729eda"
}


def getProbs(r):
    parsed = json.loads(r.text)
    tags = [parsed['Predictions'][n]['Tag']
            for n in range(0, len(parsed['Predictions']))]
    probabilities = [parsed['Predictions'][n]['Probability']
                     for n in range(0, len(parsed['Predictions']))]
    return tags, probabilities


if __name__ == '__main__':

    cap = cv2.VideoCapture(1)

    font = cv2.FONT_HERSHEY_PLAIN
    cnt = 0

    while True:
        cnt += 1
        ret, frame = cap.read()
        cv2.imshow("frame", frame)

        if cnt % 30 == 0:
            # isolation code here
            cv2.imwrite('unblemished.jpg', frame)

            ###########

            im = frame
            ret, thresh = cv2.threshold(cv2.cvtColor(
                im, cv2.COLOR_BGR2GRAY), 175, 255, 0)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)\

            img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            img = cv2.drawContours(image, contours, 0, (0, 255, 0), 3)

            if (contours):
                rect = cv2.minAreaRect(contours[0])
                boxV = cv2.boxPoints(rect)
                # print(cv2.boxPoints(rect))
                box = np.int0(boxV)
                im3 = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

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

                for index in range(0, 4):
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

                cropped = cv2.imread('unblemished.jpg')[minY:maxY, minX:maxX]
                if not (maxY - minY <= 0 or maxX - minX <= 0):
                    cv2.imwrite('crop.jpg', cropped)
                    cv2.imshow('crop', cropped)
                    # end isolation code

                    cv2.imshow('pass', cv2.imread('crop.jpg'))
                    with open("crop.jpg", "rb") as imageFile:
                        f = imageFile.read()
                        b = bytearray(f)
                    response = requests.request(
                        "POST", url, data=b, headers=headers, params=querystring)

                    tag, val = getProbs(response)
                    largeval = heapq.nlargest(4, val)
                    largetagidx = reversed(np.argsort(val)[:4])
                    largetag = [tag[n] for n in range(4)]

                    n = 0
                    for key, val in zip(largetag, largeval):
                        cv2.putText(im3, str((key, val))[2:len(str((key, val))) - 1], (10, 50 + n), font,
                                    2, (0, 0, 0), 1, cv2.LINE_AA)
                        n += 50
                    # cv2.imshow("every30", frame)
                    cv2.imshow('im3', im3)


        k = cv2.waitKey(1)
        if k == 27:
            break
