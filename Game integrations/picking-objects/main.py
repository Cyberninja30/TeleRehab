import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Game window settings
width, height = 800, 600
objects = {  # Object positions and types
    'ball': [random.randint(100, 700), random.randint(100, 500)],
    'chair': [random.randint(100, 700), random.randint(100, 500)],
    'table': [random.randint(100, 700), random.randint(100, 500)]
}

# Target zones for objects
targets = {
    'ball': (100, 100),
    'chair': (400, 300),
    'table': (600, 450)
}

# Capture video
cap = cv2.VideoCapture(0)
selected_object = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger and thumb tip positions
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            ix, iy = int(index_finger.x * w), int(index_finger.y * h)
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            
            # Check for pinch gesture (distance between thumb and index finger is small)
            if abs(ix - tx) < 30 and abs(iy - ty) < 30:
                for obj_name, pos in objects.items():
                    if abs(ix - pos[0]) < 40 and abs(iy - pos[1]) < 40:
                        selected_object = obj_name
                        break
            
            if selected_object:
                objects[selected_object] = [ix, iy]
                
            if selected_object and abs(ix - targets[selected_object][0]) < 50 and abs(iy - targets[selected_object][1]) < 50:
                cv2.putText(frame, f'{selected_object} placed correctly!', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                selected_object = None

    # Draw objects
    for obj_name, pos in objects.items():
        color = (0, 0, 255) if selected_object == obj_name else (255, 0, 0)
        cv2.circle(frame, tuple(pos), 30, color, -1)
        cv2.putText(frame, obj_name, (pos[0]-20, pos[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw target zones
    for obj_name, pos in targets.items():
        cv2.rectangle(frame, (pos[0]-20, pos[1]-20), (pos[0]+20, pos[1]+20), (0, 255, 0), 2)
        cv2.putText(frame, f'Target: {obj_name}', (pos[0]-30, pos[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Object Game", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()