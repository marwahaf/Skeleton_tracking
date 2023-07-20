import torchvision.transforms as transforms
import torchvision.models.detection as detection
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn
import time
import numpy as np

# Define the skeleton connections
skeleton_connections = [
    (5, 6), (6, 7), (7, 8), (8, 11),  # Right arm
    (5, 4), (4, 3), (3, 2), (2, 0),  # Left arm
    (11, 12), (12, 13), (13, 14), (14, 16),  # Right leg
    (11, 10), (10, 9), (9, 8), (8, 6)  # Left leg
]

# Initialize variables for metrics
num_frames = 0
total_fps = 0
mje_values = []
pck_values = []

def calculate_mje(predicted_joints, ground_truth_joints):
    errors = np.linalg.norm(predicted_joints - ground_truth_joints, axis=1)
    mje = np.mean(errors)
    return mje

def calculate_pck(predicted_joints, ground_truth_joints, threshold):
    errors = np.linalg.norm(predicted_joints - ground_truth_joints, axis=1)
    correct = np.sum(errors <= threshold)
    pck = (correct / len(predicted_joints)) * 100
    return pck

while True:
    ret, frame = cap.read()
    cv2.imshow('Skeleton Tracking', frame)

    # Preprocess the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = transforms.ToTensor()(frame_rgb)

    # Run the frame through the model
    with torch.no_grad():
        predictions = model([input_image])

    # Get the keypoints and draw them on the frame
    keypoints = predictions[0]['keypoints']
    for person in keypoints:
        for kp in person:
            x, y, conf = kp
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Draw skeleton lines
        for connection in skeleton_connections:
            kp1, kp2 = connection
            x1, y1, conf1 = person[kp1]
            x2, y2, conf2 = person[kp2]
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Get the predicted joint positions
        predicted_joints = np.array([[kp[0], kp[1]] for kp in person if kp[2] > 0.5])
        # Generate imaginary ground truth joint positions for demonstration
        ground_truth_joints = np.random.rand(predicted_joints.shape[0], 2)  # Replace with your own ground truth data

        # Calculate accuracy metrics
        mje = calculate_mje(predicted_joints, ground_truth_joints)
        pck = calculate_pck(predicted_joints, ground_truth_joints, threshold=0.1)

        # Update accuracy metric lists
        mje_values.append(mje)
        pck_values.append(pck)

    # Calculate and display metrics
    num_frames += 1
    current_fps = 1.0 / time.time()
    total_fps += current_fps
    avg_fps = total_fps / num_frames if num_frames > 0 else 0
    avg_mje = np.mean(mje_values) if len(mje_values) > 0 else 0
    avg_pck = np.mean(pck_values) if len(pck_values) > 0 else 0

    cv2.putText(frame, "FPS: {:.2f}".format(avg_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Avg MJE: {:.2f}".format(avg_mje), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Avg PCK: {:.2f}".format(avg_pck), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with keypoints, skeleton, and metrics
    cv2.imshow('Skeleton Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
