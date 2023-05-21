import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import lm_utils


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

img = cv2.imread('./data/training/016z051pf.JPG')


imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(imgRGB)


mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


for id, lm in enumerate(results.pose_landmarks.landmark):
    h, w,c = img.shape
    #print(id, lm)
    cx, cy = int(lm.x*w), int(lm.y*h)
    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

center_x, center_y = lm_utils.find_center_of_mass(results.pose_landmarks)
cx, cy = int(center_x*w), int(center_y*h)
cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)



aspect_ratio = img.shape[1] / img.shape[0]

width = 700
height = int(width / aspect_ratio)

resized_img = cv2.resize(img, (width,height))

cv2.imshow("Image", resized_img)
cv2.waitKey(10000)