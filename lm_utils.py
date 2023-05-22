import numpy as np
import math




def lm_distance(lms, lm1, lm2):
    lm1_coordinates = np.array([lms.landmark[lm1].x, lms.landmark[lm1].y])
    lm2_coordinates = np.array([lms.landmark[lm2].x, lms.landmark[lm2].y])

    return np.linalg.norm(lm2_coordinates - lm1_coordinates)

def lm_distance_(lms, x, y, lm2):
    lm1_coordinates = np.array([x, y])
    lm2_coordinates = np.array([lms.landmark[lm2].x, lms.landmark[lm2].y])

    return np.linalg.norm(lm2_coordinates - lm1_coordinates)


def front_calcs(lms):
    shoulder_width = lm_distance(lms,12,11)

    right_arm = lm_distance(lms,12,14) + lm_distance(lms,14,16)

    right_leg = lm_distance(lms,24,26) + lm_distance(lms,26,28)

    left_leg = lm_distance(lms,23,25) + lm_distance(lms,25,27)

    avg_torso = (lm_distance(lms,12,24) + lm_distance(lms,11,23)) / 2

    left_torso = lm_distance(lms,12,24)

    right_torso = lm_distance(lms,11,23)

    avg_leg = (right_leg + left_leg) / 2

    total_leg = right_leg + left_leg

    upper_to_lower = avg_torso / avg_leg

    body_length = avg_torso + avg_leg

    body_to_leg = body_length / avg_leg

    center_x, center_y = find_center_of_mass(lms)

    center_to_ankle = lm_distance_(lms, center_x, center_y, 28)

    knee_to_knee = lm_distance(lms, 26, 25)

    avg_shoulder_height = (lms.landmark[12].y + lms.landmark[11].y) / 2

    shoulder_nose_height = lms.landmark[0].y - avg_shoulder_height

    waist = lm_distance(lms, 24, 23)

    shoulder_to_waist = waist / shoulder_width

    left_torso_angle = calculate_angle_3(lms, 12, 24, 23)

    right_torso_angle = calculate_angle_3(lms, 12, 23, 24)

    left_shoulder_angle = calculate_angle_3(lms, 24, 12, 11)

    right_shoulder_angle = calculate_angle_3(lms, 12, 11, 23)

    left_leg_angle = calculate_angle_3(lms, 28, 26, 24)

    right_leg_angle = calculate_angle_3(lms, 27, 25, 23)

    torso_left_leg_angle = calculate_angle_3(lms, 12, 24, 26)

    torso_right_leg_angle = calculate_angle_3(lms, 11, 23, 25)

    com_left_angle = calculate_angle_3_(lms, 12, center_x, center_y, 24)

    com_right_angle = calculate_angle_3_(lms, 11, center_x, center_y, 23)

    ear_ear = lm_distance(lms, 8, 7)

    cross_torso = lm_distance(lms, 12, 23)

    cross_torso_2 = lm_distance(lms, 11, 24)

    cross_torso_angle = calculate_angle(lms, 12, 23)

    cross_torso_angle_2 = calculate_angle(lms, 11, 24)

    return [shoulder_width, left_leg, right_leg, waist, knee_to_knee, ear_ear,
            left_torso, right_torso, cross_torso, cross_torso_2, cross_torso_angle,
            cross_torso_angle_2]
    #return [com_left_angle, com_right_angle]
    #return [left_shoulder_angle, right_shoulder_angle, left_torso_angle, right_torso_angle]
    # return []




def side_calcs(lms):
    left_arm = lm_distance(lms,11, 13) + lm_distance(lms,13, 15)

    height = lm_distance(lms, 11, 27)

    arm_to_height = left_arm / height

    leg_waist_angle = calculate_angle(lms, 25, 23)

    waist_shoulder_angle = calculate_angle(lms, 23, 11)

    neck_angle = calculate_angle(lms, 11, 0)

    neck = lm_distance(lms, 11, 0)

    nose_left_ear = lm_distance(lms, 0, 7)

    #return [height, arm_to_height]
    #return [height, leg_waist_angle, waist_shoulder_angle, neck_angle, left_arm]
    return [leg_waist_angle, waist_shoulder_angle, neck_angle, neck, nose_left_ear, height]
    #return [neck_angle]
    return []




def com(lms):
    center_x, center_y = find_center_of_mass(lms)
    return [center_x,center_y]



def find_center_of_mass(landmark_list):
    total_landmarks = len(landmark_list.landmark)
    sum_x = sum([lm.x for lm in landmark_list.landmark])
    sum_y = sum([lm.y for lm in landmark_list.landmark])
    avg_x = sum_x / total_landmarks
    avg_y = sum_y / total_landmarks
    return avg_x, avg_y




def calculate_angle(lms, lm1, lm2):
    x1 = lms.landmark[lm1].x
    y1 = lms.landmark[lm1].y
    x2 = lms.landmark[lm2].x
    y2 = lms.landmark[lm2].y
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle


import math


def calculate_angle_3(lms, lm1, lm2, lm3):
    x1 = lms.landmark[lm1].x
    y1 = lms.landmark[lm1].y
    x2 = lms.landmark[lm2].x
    y2 = lms.landmark[lm2].y
    x3 = lms.landmark[lm3].x
    y3 = lms.landmark[lm3].y

    # Calculate the lengths of the three sides of the triangle formed by the landmarks
    a = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)  # Length of side a
    b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)  # Length of side b
    c = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Length of side c

    # Calculate the angle using the law of cosines
    angle = math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))

    return angle

def calculate_angle_3_(lms, lm1, lm2x, lm2y, lm3):
    x1 = lms.landmark[lm1].x
    y1 = lms.landmark[lm1].y
    x2 = lm2x
    y2 = lm2y
    x3 = lms.landmark[lm3].x
    y3 = lms.landmark[lm3].y

    # Calculate the lengths of the three sides of the triangle formed by the landmarks
    a = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)  # Length of side a
    b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)  # Length of side b
    c = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Length of side c

    # Calculate the angle using the law of cosines
    angle = math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))

    return angle