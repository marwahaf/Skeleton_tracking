import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize variables for metrics
mje_values = []
pck_values = []
frame_times = []

def calculate_mje(predicted_joints, ground_truth_joints):
    errors = np.linalg.norm(predicted_joints - ground_truth_joints, axis=1)
    mje = np.mean(errors)
    return mje

def calculate_pck(predicted_joints, ground_truth_joints, threshold):
    errors = np.linalg.norm(predicted_joints - ground_truth_joints, axis=1)
    correct = np.sum(errors <= threshold)
    pck = (correct / len(predicted_joints)) * 100
    return pck

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")
        raise TypeError

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Measure frame processing time
        start_time = time.time()

        results = pose.process(image)

        # Calculate frame processing time
        frame_time = time.time() - start_time
        frame_times.append(frame_time)

        if results.pose_landmarks is None:
            continue

        # Get predicted joint positions
        predicted_joints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])

        # Generate imaginary ground truth joint positions for demonstration
        ground_truth_joints = np.random.rand(predicted_joints.shape[0], 3)  # Replace with your own ground truth data

        # Calculate accuracy metrics
        mje = calculate_mje(predicted_joints, ground_truth_joints)
        pck = calculate_pck(predicted_joints, ground_truth_joints, threshold=0.1)

        # Update accuracy metric lists
        mje_values.append(mje)
        pck_values.append(pck)

        # Draw landmarks on the frame
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the metrics on the frame
        cv2.putText(image, "MJE: {:.2f}".format(mje), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "PCK: {:.2f}".format(pck), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Avg MJE: {:.2f}".format(np.mean(mje_values)), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Avg PCK: {:.2f}".format(np.mean(pck_values)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Avg Frame Time: {:.2f} s".format(np.mean(frame_times)), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "FPS: {:.2f}".format(1 / np.mean(frame_times)), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Frame', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
