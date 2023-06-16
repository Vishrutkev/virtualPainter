import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # to draw points on the hand (used mediapipe)



class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon



        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw="True"):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert img from BGR to RGB because hands only uses RGB images
        self.results = hands.process(imgRGB)  # process frame and give's the result(does not display anything yet)

        # print(results.multi_hand_landmarks)  it will detect the hand and print the results accordingly

        # below loop will show us the dots on the hands

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(handLMS.landmark):
                       h, w, c = img.shape
                       cx, cy = int(lm.x * w), int(lm.y * h)
                       #if id == 3:
                            #cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                       #if id == 5:
                            #cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
                       #if id == 9:
                            #cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                       #if id == 13:
                            #cv2.circle(img, (cx, cy), 15, (128, 0, 128), cv2.FILLED)
                       #if id == 17:
                            #cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                    mpDraw.draw_landmarks(img, handLMS,
                                               mpHands.HAND_CONNECTIONS)  # display BGR image with dots and connections between dots

        return img


    # we're finding the position of 1 hand, so bascically will get first hand and within that hand we will get all the landmarks.

    def findPosition(self, img, handNo = 0, draw = True):

        self.lmList = []
        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)    #it will print x and y co-ordinates
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)   will print the value in pixels for all the 21 values
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return self.lmList


    def fingersUp(self):
        fingers = []

        #thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:  #checking if the tip of our thumb is on right(close) or left(open)
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range (1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:   #checking if tip of the finger is above other landmark which is 2 steps below it
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(False, 2, 0.5, 0.5)
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        #if len(lmList) != 0:
            #print(lmList[4])     it will only print handlandmark position at 4.
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)  # it will show fps on the screen



        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
