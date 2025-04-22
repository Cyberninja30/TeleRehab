import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Video Capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Connect the Dots", cv2.WINDOW_NORMAL)

# Game Variables
initial_dots = 5  # Starting number of dots
num_dots = initial_dots
dots = []  # Stores dot positions
connected_lines = []  # Stores lines between connected dots
current_dot = 0  # Index of the dot to connect next
game_over = False
start_time = time.time()
next_level = False
level_time_limit = 10  # Time limit per level in seconds
score = 0

# Encouragement Messages
messages = ["Keep up the good work!", "You're doing great!", "Keep going!", "Nice job!", "Well done!"]

# Trophy System
trophies = {"Bronze": 10, "Silver": 20, "Gold": 30}

# Generate Random Dots
def generate_dots():
    global dots, connected_lines, current_dot, game_over, num_dots, next_level, start_time
    dots = [(random.randint(100, 500), random.randint(100, 400)) for _ in range(num_dots)]
    connected_lines = []
    current_dot = 0
    game_over = False
    next_level = False
    start_time = time.time()

def display_menu():
    print("Select a Mode:")
    print("1. Practice Match")
    print("2. Single Player")
    print("3. Multiplayer (vs Computer)")
    print("4. Other Features")
    print("5. Exit")
    choice = input("Enter your choice: ")
    return choice

while True:
    choice = display_menu()
    if choice == "5":
        break
    elif choice == "1":
        level_time_limit = 0  # No time limit in practice mode
    elif choice == "2":
        level_time_limit = 10  # Standard mode
    elif choice == "3":
        level_time_limit = 15  # Multiplayer mode
    elif choice == "4":
        print("Other features coming soon...")
        continue
    
    generate_dots()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror effect
        h, w, _ = frame.shape

        # Convert Frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Draw Dots
        for i, (x, y) in enumerate(dots):
            color = (0, 255, 0) if i == current_dot else (255, 255, 255)  # Green for the next dot
            cv2.circle(frame, (x, y), 15, color, -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw Connected Lines
        for (x1, y1), (x2, y2) in connected_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Display Time Left
        elapsed_time = time.time() - start_time
        time_left = max(0, level_time_limit - elapsed_time) if level_time_limit > 0 else "∞"
        cv2.putText(frame, f"Time Left: {time_left}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if time_left == 0 and level_time_limit > 0:
            game_over = True
            cv2.putText(frame, "Time's Up! Game Over", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            time.sleep(2)
            num_dots = initial_dots  # Reset difficulty
            generate_dots()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Draw fingertip indicator
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

                # Check if index finger touches the correct dot
                if not game_over and current_dot < len(dots):
                    dot_x, dot_y = dots[current_dot]
                    if np.linalg.norm(np.array([x, y]) - np.array([dot_x, dot_y])) < 20:
                        if current_dot > 0:
                            connected_lines.append((dots[current_dot - 1], dots[current_dot]))
                        current_dot += 1
                        score += 10
                        cv2.putText(frame, random.choice(messages), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Draw Hand Landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check if game is complete
        if current_dot == num_dots:
            cv2.putText(frame, "You Win!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            game_over = True
            next_level = True

        # Move to next level only after current one is completed
        if next_level:
            time.sleep(2)  # Pause before moving to the next level
            num_dots += 2  # Increase difficulty
            generate_dots()

        # Display the Window
        cv2.imshow("Connect the Dots", frame)

        # Key Press Actions
        # Key Press Actions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
           cap.release()
           cv2.destroyAllWindows()
           hands.close()
           break  # Immediate exit


    # Display trophy
    for trophy, threshold in trophies.items():
        if score >= threshold:
            print(f"Congratulations! You earned a {trophy} trophy!")

cap.release()
cv2.destroyAllWindows()
