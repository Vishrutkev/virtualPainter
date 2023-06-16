import cv2
import numpy as np
import time
import os   # to access Header file
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 50

folderPath = "Header"
myList = os.listdir(folderPath)
myList.sort()
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[1]    # it will overlay on top of the original image
drawColor = (0, 0, 255)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)   #width
cap.set(4, 720)    #height
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)   #using numpy to draw on campus using zeros method, unit8(unsigned int of 8 bits -> (0, 255)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  #it will flip the image because if you draw on left it will draw on the right

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if(len(lmList) != 0):
        #print(lmList)

        x1, y1 = lmList[8][1:] #tip of index finger
        x2, y2 = lmList[12][1:] #tip of middle finger

        fingers = detector.fingersUp()
        #print(fingers)   it will show which fingers/thumb are up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("Selection Mode")
            #Checking for the click
            if y1 < 125:
                if 250<x1<375:
                    header = overlayList[1]
                    drawColor = (255, 0, 255)

                elif 425<x1<550:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)

                elif 600<x1<725:
                    header = overlayList[3]
                    drawColor = (0, 200, 255)

                elif 775<x1<900:
                    header = overlayList[4]
                    drawColor = (255, 255, 0)

                elif 1000<x1<1250:
                    header = overlayList[5]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        elif fingers[1]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor,eraserThickness)  # it will draw a line from previous point(xp, yp) to new points(x1, y1)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor,brushThickness)  # it will draw a line from previous point(xp, yp) to new points(x1, y1)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)  #Converting canvas img to gray
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  #Converting into binary image and inversing(black to white and any other color to black) it
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)  #Converting it back to so that we can add to original image
    img = cv2.bitwise_and(img, imgInv)    #adding inversed image(black drawing) to the original image
    img = cv2.bitwise_or(img, imgCanvas)   #merging the above created image to the imgCanvas so that we get the colors we wanted

    img[0:125, 0:1280] = header   #slicing the image because we know that our header img is 125x1280
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)  #merging original image to canvas image
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
