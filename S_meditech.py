import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])  # First point
    b = np.array([b.x, b.y])  # Middle point
    c = np.array([c.x, c.y])  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Initialize variables
reps = 0
points = 0
stage = None
exercise = "Squats"  # Default exercise. Options: "Finger Twirling", "Head Rotation", "Fist Rotation", "Squats"
fullscreen = False  # Track fullscreen mode

# Start video capture
cap = cv2.VideoCapture(0)
screen_width = 1920  # Replace with your screen width
screen_height = 1080  # Replace with your screen height

cv2.namedWindow("Exercise Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Exercise Tracker", 1280, 720)

print("Press 'Q' to quit. Use keys 1-4 to switch exercises.")
print("1: Squats, 2: Finger Twirling, 3: Head Rotation, 4: Fist Rotation")
print("Press 'F' to toggle fullscreen.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Display selected exercise
    cv2.putText(frame, f"Exercise: {exercise}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Squat Tracking
    if exercise == "Squats" and results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results_pose.pose_landmarks.landmark

        left_hip = mp_pose.PoseLandmark.LEFT_HIP
        left_knee = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle = mp_pose.PoseLandmark.LEFT_ANKLE

        # Calculate knee angle
        knee_angle = calculate_angle(landmarks[left_hip], landmarks[left_knee], landmarks[left_ankle])

        # Detect the squat movement
        if knee_angle > 160:  # Standing
            stage = "up"
        elif knee_angle < 90 and stage == "up":  # Squatted
            stage = "down"
            reps += 1
            points += 10  # Award points for each squat

    # Finger Twirling Tracking
    elif exercise == "Finger Twirling" and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance between thumb and index tips
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if distance < 0.03:  # Threshold for closed position
                stage = "closed"
            elif distance > 0.06 and stage == "closed":  # Full twirl
                stage = "open"
                reps += 1
                points += 5  # Award points for each twirl

    # Head Rotation Tracking
    elif exercise == "Head Rotation" and results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results_pose.pose_landmarks.landmark

        left_ear = mp_pose.PoseLandmark.LEFT_EAR
        nose = mp_pose.PoseLandmark.NOSE
        right_ear = mp_pose.PoseLandmark.RIGHT_EAR

        # Calculate head rotation angle
        angle = calculate_angle(landmarks[left_ear], landmarks[nose], landmarks[right_ear])

        if angle > 140:  # Rotate fully to one side
            stage = "rotated"
        elif angle < 100 and stage == "rotated":  # Return to neutral
            stage = "neutral"
            reps += 1
            points += 7  # Award points for each full rotation

    # Fist Rotation Tracking
    elif exercise == "Fist Rotation" and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate vertical movement for fist rotation
            distance = abs(wrist.y - pinky_tip.y)
            if distance > 0.1:  # Threshold for fist down
                stage = "down"
            elif distance < 0.05 and stage == "down":  # Fist up
                stage = "up"
                reps += 1
                points += 8  # Award points for each fist rotation

    # Display feedback
    cv2.putText(frame, f"Reps: {reps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Points: {points}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Resize and crop frame for fullscreen handling
    h, w, _ = frame.shape
    scale_width = screen_width / w
    scale_height = screen_height / h
    scale = max(scale_width, scale_height)

    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    start_x = (new_width - screen_width) // 2
    start_y = (new_height - screen_height) // 2
    cropped_frame = resized_frame[start_y:start_y + screen_height, start_x:start_x + screen_width]

    if fullscreen:
        cv2.imshow("Exercise Tracker", cropped_frame)
    else:
        cv2.imshow("Exercise Tracker", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('1'):
        exercise = "Squats"
        reps, points = 0, 0  # Reset counters
    elif key == ord('2'):
        exercise = "Finger Twirling"
        reps, points = 0, 0
    elif key == ord('3'):
        exercise = "Head Rotation"
        reps, points = 0, 0
    elif key == ord('4'):
        exercise = "Fist Rotation"
        reps, points = 0, 0
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Exercise Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Exercise Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
