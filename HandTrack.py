import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture(0)

cTime = 0
pTime = 0
fps = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()

# mpFace = mp.solutions.face_mesh
# face = mpFace.FaceMesh()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resultsHands = hands.process(imgRGB)
    #resultsFace = face.process(imgRGB)

    if resultsHands.multi_hand_landmarks:
        for handLms in resultsHands.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    # if resultsFace.multi_face_landmarks:
    #     for face in resultsFace.multi_face_landmarks:
    #         mpDraw.draw_landmarks(img, face, mpFace.FACE_CONNECTIONS)

    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime

    cv2.putText(img, str(fps), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)
