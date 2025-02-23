import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Game Variables
w, h = 640, 480  # Screen size
ball_x, ball_y = w // 2, h // 2
ball_dx, ball_dy = random.choice([-6, 6]), random.choice([-6, 6])
ball_speed = 6
ball_radius = 15
score1, score2 = 0, 0  # Initial scores

# Paddle Variables
paddle_w, paddle_h = 20, 120
paddle1_x, paddle1_y = 60, h // 2
paddle2_x, paddle2_y = w - 60, h // 2
prev_paddle1_y, prev_paddle2_y = paddle1_y, paddle2_y
prev_time = time.time()

# Create a window
cv2.namedWindow("AI Table Tennis", cv2.WINDOW_NORMAL)
fullscreen = False  # Track fullscreen state

# Function to draw scoreboard
def draw_scoreboard(frame):
    cv2.putText(frame, f"Player 1: {score1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Player 2: {score2}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Game Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process Hands
    result = hands.process(rgb_frame)
    curr_time = time.time()
    dt = max(curr_time - prev_time, 0.01)
    prev_time = curr_time

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:
            wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
            x, y = int(wrist.x * w), int(wrist.y * h)
            
            # Detect paddle movement
            if x < w // 2:
                paddle1_y = max(60, min(y, h - 60))
            
            else:
                paddle2_y = max(60, min(y, h - 60))
            
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    
    # Ball Movement
    ball_x += ball_dx
    ball_y += ball_dy
    
    # Ball Collision with Walls
    if ball_y - ball_radius <= 0 or ball_y + ball_radius >= h:
        ball_dy *= -1
    
    # Paddle Collision Detection
    if (paddle1_x < ball_x < paddle1_x + paddle_w and paddle1_y - 60 < ball_y < paddle1_y + 60) or \
       (paddle2_x - paddle_w < ball_x < paddle2_x and paddle2_y - 60 < ball_y < paddle2_y + 60):
        ball_dx *= -1

    # Check if a player misses the ball
    if ball_x < 0:
        score2 += 1  # Player 2 scores when Player 1 misses
        ball_x, ball_y = w // 2, h // 2
        ball_dx, ball_dy = random.choice([-6, 6]), random.choice([-6, 6])
    
    elif ball_x > w:
        score1 += 1  # Player 1 scores when Player 2 misses
        ball_x, ball_y = w // 2, h // 2
        ball_dx, ball_dy = random.choice([-6, 6]), random.choice([-6, 6])

    # Draw Paddles
    cv2.rectangle(frame, (paddle1_x, paddle1_y - 60), (paddle1_x + paddle_w, paddle1_y + 60), (255, 0, 0), -1)
    cv2.rectangle(frame, (paddle2_x - paddle_w, paddle2_y - 60), (paddle2_x, paddle2_y + 60), (0, 255, 0), -1)
    
    # Draw Ball
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)
    
    # Draw Middle Line
    cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 2)
    
    # Draw Scoreboard
    draw_scoreboard(frame)
    
    # Show Frame
    cv2.imshow("AI Table Tennis", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty("AI Table Tennis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
