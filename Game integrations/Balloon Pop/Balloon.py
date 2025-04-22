import cv2
import mediapipe as mp
import pygame
import random
import time

# Initialize Mediapipe Hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# OpenCV camera capture
cap = cv2.VideoCapture(0)

# Game Variables
game_mode = "balloon_pop"
balloons = []
score = 0
balloon_spawn_interval = 1.5  # Time interval for spawning new balloons
last_spawn_time = time.time()
fullscreen = False  # Track fullscreen state

# Function to toggle fullscreen
def toggle_fullscreen():
    global screen, fullscreen
    fullscreen = not fullscreen
    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((800, 600))

# Function to spawn balloons continuously
def spawn_balloon():
    x = random.randint(100, 700)
    y = 600  # Spawn at the bottom
    color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    balloons.append([x, y, color])

# Function to draw and update balloon game
def draw_balloon_game(hand_landmarks):
    global balloons, score, last_spawn_time

    screen.fill((0, 0, 0))  # Black background
    
    # Move balloons up gradually
    for balloon in balloons:
        balloon[1] -= 2  # Adjusted speed to make it more natural
        pygame.draw.circle(screen, balloon[2], (balloon[0], balloon[1]), 30)

    # Remove balloons that move out of screen
    balloons[:] = [balloon for balloon in balloons if balloon[1] > -30]

    # Spawn new balloons based on time interval
    if time.time() - last_spawn_time > balloon_spawn_interval:
        spawn_balloon()
        last_spawn_time = time.time()

    # Hand tracking
    if hand_landmarks:
        index_finger = hand_landmarks.landmark[8]
        x, y = int(index_finger.x * 800), int(index_finger.y * 600)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 10)  # Draw fingertip tracker

        for balloon in balloons[:]:
            if (x - balloon[0]) ** 2 + (y - balloon[1]) ** 2 < 900:
                balloons.remove(balloon)
                score += 1

    # Display score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.update()

# Spawn initial balloons
for _ in range(3):
    spawn_balloon()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if game_mode == "balloon_pop":
                draw_balloon_game(hand_landmarks)
    else:
        draw_balloon_game(None)  # Keep game running even if no hand is detected

    # Handle key presses
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                toggle_fullscreen()

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
