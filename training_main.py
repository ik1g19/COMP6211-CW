import cv2
import mediapipe as mp
import numpy as np
import csv
import os

def lm_distance(lm1, lm2):
    lm1_coordinates = np.array([results.pose_landmarks.landmark[lm1].x, results.pose_landmarks.landmark[lm1].y])
    lm2_coordinates = np.array([results.pose_landmarks.landmark[lm2].x, results.pose_landmarks.landmark[lm2].y])

    return np.linalg.norm(lm2_coordinates - lm1_coordinates)

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)



training_dir = './data/training'

for filename in os.listdir(training_dir):
    if os.path.isfile(os.path.join(training_dir, filename)):
        # Process the file here

        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        mpDraw = mp.solutions.drawing_utils

        img = cv2.imread(training_dir + '/' + filename)


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


            if "f" in filename:
                shoulder_width = lm_distance(12,11)

                right_arm = lm_distance(12,14) + lm_distance(14,16)

                right_leg = lm_distance(24,26) + lm_distance(26,28)

                left_leg = lm_distance(23,25) + lm_distance(25,27)

                avg_torso = (lm_distance(12,24) + lm_distance(11,23)) / 2

            else:
                left_arm = lm_distance(11, 13) + lm_distance(13, 15)

                data = [filename, shoulder_width, right_arm, left_arm, right_leg, left_leg, avg_torso]

                write_to_csv('./vectors/vectors2.csv', [data])

            # for id, lm in enumerate(results.pose_landmarks.landmark):
            #     h, w,c = img.shape
            #     #print(id, lm)
            #     cx, cy = int(lm.x*w), int(lm.y*h)
            #     cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)


            # aspect_ratio = img.shape[1] / img.shape[0]
            #
            # width = 700
            # height = int(width / aspect_ratio)
            #
            # resized_img = cv2.resize(img, (width,height))
            #
            # cv2.imshow("Image", resized_img)
            # cv2.waitKey(10000)