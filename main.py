import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import lm_utils
import knn

def lm_distance(lm1, lm2):
    lm1_coordinates = np.array([results.pose_landmarks.landmark[lm1].x, results.pose_landmarks.landmark[lm1].y])
    lm2_coordinates = np.array([results.pose_landmarks.landmark[lm2].x, results.pose_landmarks.landmark[lm2].y])

    return np.linalg.norm(lm2_coordinates - lm1_coordinates)

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def data_from_csv(filename):
    d = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            float_row = [row[0]] + [float(value) for value in row[1:]]
            d.append(float_row)
    return d


testing_dir = './data/test'
trained_data = './vectors/vectors2.csv'


data = data_from_csv(trained_data)


answers = ["021z001","021z002","021z003","021z004","021z005","021z006","021z007","021z008","021z009","021z010","024z011"]



count = 0
correct = 0
for filename in os.listdir(testing_dir):
    if os.path.isfile(os.path.join(testing_dir, filename)):
        # Process the file here
        count = count + 1

        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        mpDraw = mp.solutions.drawing_utils

        img = cv2.imread(testing_dir + '/' + filename)


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


            if count % 2 == 0:
                front_vals = lm_utils.front_calcs(results.pose_landmarks)

                query = front_vals + side_vals

                k_labels, prediction = knn.knn(data, query, k=1, distance_fn=knn.euclidean_distance, choice_fn=knn.mode)

                print(filename + ": Real ID - " + answers[(int(count / 2)) - 1])
                print("Prediction: " + prediction)
                if ((answers[(int(count / 2)) - 1] + "ps.jpg") == prediction): correct = correct + 1
                print("")
                print("")

            else:
                side_vals = lm_utils.side_calcs(results.pose_landmarks)

print("Correct: " + str(correct) + "/11")