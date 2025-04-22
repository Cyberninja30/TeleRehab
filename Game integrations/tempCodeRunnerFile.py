import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Game Settings
screen_width, screen_height = 800, 600
snake_size = 20
snake_speed = 15  # Reduced speed for smoother gameplay
snake = [(screen_width // 2, screen_height // 2)]
direction = "RIGHT"

food_x, food_y = random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)
score = 0

# Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip & Resize Frame
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (screen_width, screen_height))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand Detection
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            index_finger = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger.x * screen_width), int(index_finger.y * screen_height)

            # Gesture-based Control (Reduced sensitivity for better control)
            if abs(x - snake[0][0]) > 50:  # Horizontal Movement
                if x > snake[0][0] and direction != "LEFT":
                    direction = "RIGHT"
                elif x < snake[0][0] and direction != "RIGHT":
                    direction = "LEFT"
            elif abs(y - snake[0][1]) > 50:  # Vertical Movement
                if y > snake[0][1] and direction != "UP":
                    direction = "DOWN"
                elif y < snake[0][1] and direction != "DOWN":
                    direction = "UP"

            # Draw Hand Landmarks
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Move Snake
    head_x, head_y = snake[0]
    if direction == "UP":
        head_y -= snake_speed
    elif direction == "DOWN":
        head_y += snake_speed
    elif direction == "LEFT":
        head_x -= snake_speed
    elif direction == "RIGHT":
        head_x += snake_speed

    new_head = (head_x, head_y)
    snake.insert(0, new_head)

    # Food Collision Detection
    if abs(head_x - food_x) < snake_size and abs(head_y - food_y) < snake_size:
        score += 1
        food_x, food_y = random.randint(50, screen_width - 50), random.randint(50, screen_height - 50)
    else:
        snake.pop()

    # Wall / Self Collision Handling
    if head_x < 0 or head_x > screen_width or head_y < 0 or head_y > screen_height or new_head in snake[1:]:
        snake = [(screen_width // 2, screen_height // 2)]
        direction = "RIGHT"
        score = 0

    # Draw Snake
    for part in snake:
        cv2.rectangle(frame, (part[0] - snake_size // 2, part[1] - snake_size // 2),
                      (part[0] + snake_size // 2, part[1] + snake_size // 2), (0, 255, 0), -1)

    # Draw Food
    cv2.circle(frame, (food_x, food_y), snake_size // 2, (0, 0, 255), -1)

    # Display Score
    cv2.putText(frame, f"Score: {score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Show Frame
    cv2.imshow("Snake Game", frame)

    # Add Delay to Control Speed
    if cv2.waitKey(50) & 0xFF == ord('q'):  # Slower frame update
        break

cap.release()
cv2.destroyAllWindows()
