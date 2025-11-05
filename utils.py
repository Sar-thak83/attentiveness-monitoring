import csv
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
predictions = []
timestamps = []

# Set CSV file path (replace with your desired path)
csv_file_path = "attentiveness_data.csv"

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results,font_scale=2, font_thickness=2):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
# Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    prediction = predict_attentiveness(results)
    text_x, text_y = (10, 30)  # Adjust coordinates as needed
    text = f"{prediction}"
    color = (0, 255, 0) if prediction == "Attentive" else (0, 0, 255)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, color, 1)  # Reduced font_scale to 0.6 and thickness to 1

    return image
def predict_attentiveness(results):
    """Predicts attentiveness based on head pose and eye aspect ratio."""

    pose_landmarks = results.pose_landmarks
    face_landmarks = results.face_landmarks

    # Thresholds
    head_yaw_threshold = 25  # degrees (left/right turn)
    head_pitch_threshold = 20  # degrees (up/down tilt)
    eye_aspect_ratio_threshold = 0.2  # Eyes closed threshold

    head_yaw = 0
    head_pitch = 0
    eye_aspect_ratio = 0.5

    # Calculate head pose angles using face landmarks
    if face_landmarks:
        head_yaw, head_pitch = calculate_head_pose(face_landmarks)
        eye_aspect_ratio = calculate_eye_aspect_ratio(face_landmarks)

    # Determine attentiveness:
    # Attentive if: head is centered (low yaw/pitch) AND eyes are open
    if (
        abs(head_yaw) < head_yaw_threshold
        and abs(head_pitch) < head_pitch_threshold
        and eye_aspect_ratio > eye_aspect_ratio_threshold
    ):
        return "Attentive"
    else:
        return "Not attentive"

def calculate_head_pose(face_landmarks):
    """
    Calculate head yaw and pitch angles from face landmarks.
    Returns: (yaw, pitch) in degrees
    """
    # Key face landmarks
    nose = face_landmarks.landmark[1]  # Nose tip
    left_eye = face_landmarks.landmark[33]  # Left eye outer
    right_eye = face_landmarks.landmark[263]  # Right eye outer
    mouth_left = face_landmarks.landmark[61]  # Left mouth corner
    mouth_right = face_landmarks.landmark[291]  # Right mouth corner
    chin = face_landmarks.landmark[152]  # Chin

    # Convert to numpy arrays
    nose_pos = np.array([nose.x, nose.y])
    left_eye_pos = np.array([left_eye.x, left_eye.y])
    right_eye_pos = np.array([right_eye.x, right_eye.y])
    mouth_left_pos = np.array([mouth_left.x, mouth_left.y])
    mouth_right_pos = np.array([mouth_right.x, mouth_right.y])
    chin_pos = np.array([chin.x, chin.y])

    # Calculate yaw (left-right head turn)
    eyes_center = (left_eye_pos + right_eye_pos) / 2
    yaw_vector = right_eye_pos - left_eye_pos
    nose_yaw_offset = nose_pos[0] - eyes_center[0]
    yaw_angle = np.arctan2(nose_yaw_offset, np.linalg.norm(yaw_vector)) * 180 / np.pi

    # Calculate pitch (up-down head tilt)
    mouth_center = (mouth_left_pos + mouth_right_pos) / 2
    face_height = np.linalg.norm(chin_pos - eyes_center)
    nose_pitch_offset = nose_pos[1] - eyes_center[1]
    pitch_angle = np.arctan2(nose_pitch_offset, face_height) * 180 / np.pi

    return yaw_angle, pitch_angle

def calculate_eye_aspect_ratio(face_landmarks):
    """
    Calculates eye aspect ratio using MediaPipe face landmarks (468 points).
    Returns average EAR for both eyes.
    """
    # MediaPipe face mesh eye landmarks
    left_eye_indices = [362, 385, 387, 263, 373, 380]  # Left eye
    right_eye_indices = [33, 160, 158, 133, 153, 144]  # Right eye

    left_eye_points = [face_landmarks.landmark[i] for i in left_eye_indices]
    right_eye_points = [face_landmarks.landmark[i] for i in right_eye_indices]

    left_ear = calculate_individual_ear(left_eye_points)
    right_ear = calculate_individual_ear(right_eye_points)

    return (left_ear + right_ear) / 2

def calculate_individual_ear(eye_landmarks):
    """
    Calculates EAR (eye aspect ratio) for a single eye.
    eye_landmarks: 6 points [P1, P2, P3, P4, P5, P6]
    """
    # Extract points
    P1 = np.array([eye_landmarks[0].x, eye_landmarks[0].y])
    P2 = np.array([eye_landmarks[1].x, eye_landmarks[1].y])
    P3 = np.array([eye_landmarks[2].x, eye_landmarks[2].y])
    P4 = np.array([eye_landmarks[3].x, eye_landmarks[3].y])
    P5 = np.array([eye_landmarks[4].x, eye_landmarks[4].y])
    P6 = np.array([eye_landmarks[5].x, eye_landmarks[5].y])

    # Vertical distances
    vertical_dist1 = np.linalg.norm(P2 - P6)
    vertical_dist2 = np.linalg.norm(P3 - P5)

    # Horizontal distance
    horizontal_dist = np.linalg.norm(P1 - P4)

    # Calculate EAR
    if horizontal_dist == 0:
        return 0
    ear = (vertical_dist1 + vertical_dist2) / (2 * horizontal_dist)
    return ear