# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils    #to draw points on the hand (used mediapipe)

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert img from BGR to RGB because hands only uses RGB images
    results = hands.process(imgRGB)   #process frame and give's the result(does not display anything yet)

    #print(results.multi_hand_landmarks)  it will detect the hand and print the results accordingly


    #below loop will ahow us the dots on the hands

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                #print(id, lm)    it will print x and y co-ordinates
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)    #will print the value in pixels for all the 21 values
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                elif id == 6:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)   #display BGR image with dots and connections between dots


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # it will show fps on the screen

    cv2.imshow("Image", img)
    cv2.waitKey(1)
