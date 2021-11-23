import cv2
import mediapipe as mp
import numpy as np
import cvzone


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4,480) # height

# cap.set(cv2.CAP_PROP_FPS, 60)
# fpsReader = cvzone.FPS()

while True:
    ret,img = cap.read()
    # img = cv2.resize(img,(600, 400))

    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_draw.DrawingSpec((255,0,0),2, 2),
                            mp_draw.DrawingSpec((255,0,255), 2, 2)
                            )

    cv2.imshow("Pose Estimation", img)


    h, w, c = img.shape
    opImg = np.zeros([h, w,c])
    opImg.fill(255)

    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_draw.DrawingSpec((255,0,0), 2, 2), 
                            mp_draw.DrawingSpec((255,0,255), 2, 2)
                            )
    cv2.imshow("Extracted Pose", opImg)

    # imgStacked = cvzone.stackImages([img, opImg],2,1)
    # _, imgStacked = fpsReader.update(imgStacked, color= (0, 0, 255))

    # cv2.imshow("Image stacked", imgStacked)

    print(results.pose_landmarks)


    key = cv2.waitKey(1)

    if key == ord("q"):
        break