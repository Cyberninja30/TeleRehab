import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import random
import math

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Snake variables
snake_points = [(300, 300)]
snake_length = []
snake_max_length = 150
speed = 20

# Food variables
food_position = (400, 400)
food_size = 40
score = 0

# Load food image
food_img = cv2.imread('Images/donut.png', cv2.IMREAD_UNCHANGED)  # Ensure correct path and format (png or jpg)

# Game over flag
game_over = False

def random_food_position():
    return random.randint(100, 1180), random.randint(100, 620)

def is_collision(pt1, pt2):
    distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    return distance < food_size

def overlay_transparent(background, overlay, x, y, scale=1):
    """Helper function to overlay transparent images like the donut."""
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    rows, cols, _ = background.shape

    if x + w > cols or y + h > rows:
        return background

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        background[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] + alpha_l * background[y:y+h, x:x+w, c])
    return background

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Detect hands
    hands, img = detector.findHands(img, flipType=False)
    if hands:
        lmList = hands[0]['lmList']
        index_finger = lmList[8][0:2]  # x, y coordinates of the index finger
        hand_type = hands[0]['type']  # "Right" or "Left"

        if not game_over:
            # Add new position to the snake
            snake_points.append(index_finger)

            # Calculate distance between points
            if len(snake_points) > 1:
                dist = math.hypot(snake_points[-1][0] - snake_points[-2][0],snake_points[-1][1] - snake_points[-2][1])
                snake_length.append(dist)
                if sum(snake_length) > snake_max_length:
                    snake_length.pop(0)
                    snake_points.pop(0)

            # Draw snake
            for i in range(len(snake_points) - 1):
                cv2.line(img, snake_points[i], snake_points[i + 1], (0, 255, 0), 15)
            cv2.circle(img, snake_points[-1], 15, (0, 255, 0), cv2.FILLED)

            # Check collision with food
            if is_collision(snake_points[-1], food_position):
                score += 1
                food_position = random_food_position()
                snake_max_length += 20  # Increase snake length

            # Draw food with overlay
            img = overlay_transparent(img, food_img, food_position[0] - food_size // 2,food_position[1] - food_size // 2, scale=0.5)

            # Check collision with itself
            for i in range(len(snake_points) - 2):
                if is_collision(snake_points[-1], snake_points[i]):
                    game_over = True
        else:
            # Game over screen
            cv2.putText(img, "Game Over", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(img, f"Score: {score}", (450, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press 'R' to Restart", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display hand type
        cv2.putText(img, hand_type, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Display score
    cv2.putText(img, f"Your Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)

    # Show the image
    cv2.imshow("Snake Game", img)

    # Key controls
    key = cv2.waitKey(1)
    if key == ord('r'):  # Restart the game
        game_over = False
        snake_points = [(300, 300)]
        snake_length = []
        snake_max_length = 150
        food_position = random_food_position()
        score = 0
    elif key == 27:  # Escape key to exit
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
