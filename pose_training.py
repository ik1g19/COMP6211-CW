import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import lm_utils
import math
import argparse
import csv_utils



def read_side_measurements(path):
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    img = cv2.imread(path)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    lm_utils.side_calcs(results.pose_landmarks)




def read_front_measurements(path):
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    img = cv2.imread(path)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    return lm_utils.front_calcs(results.pose_landmarks)





def main(arg1, arg2, arg3):
    training_dir = arg1
    csv_path = arg2
    front_first = int(arg3)

    if (front_first == 1): front = True
    else: front = False

    front_taken = False
    side_taken = False
    initial = True

    for index, filename in enumerate(os.listdir(training_dir)):
        if os.path.isfile(os.path.join(training_dir, filename)):
            # Process the file here

            path = training_dir + '/' + filename

            mpPose = mp.solutions.pose
            pose = mpPose.Pose()
            mpDraw = mp.solutions.drawing_utils

            img = cv2.imread(path)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_landmarks:

                if front:
                    front_vals = lm_utils.front_calcs(results.pose_landmarks)
                    front_taken = True

                else:
                    side_vals = lm_utils.side_calcs(results.pose_landmarks)
                    com_values = lm_utils.com(results.pose_landmarks)
                    side_taken = True

            if front_taken & side_taken:
                data = [filename] + com_values + front_vals + side_vals
                if initial:
                    csv_utils.write_to_csv(csv_path, [data])
                    initial = False
                else:
                    csv_utils.append_to_csv(csv_path, [data])
                front_taken = False
                side_taken = False

            front = not front


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add command-line argument options
    parser.add_argument("arg1", help="Description of arg1")
    parser.add_argument("arg2", help="Description of arg2")
    parser.add_argument("arg3", help="Description of arg3")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.arg1, args.arg2, args.arg3)
