import cv2
import mediapipe as mp
import pygame
import random

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Player
player_x = WIDTH // 2
player_y = HEIGHT - 50
player_speed = 7
bullets = []

# Enemy
enemy_size = 40
enemies = [{'x': random.randint(50, WIDTH - 50), 'y': 50} for _ in range(5)]

# MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
blink_detected = False
blink_cooldown = 0  # Timer to prevent multiple blinks
blink_state = False  # Track if eyes were closed in the last frame

def detect_blink(landmarks):
    """Detects blink based on eye landmarks."""
    global blink_detected, blink_cooldown, blink_state

    left_eye = [landmarks[i] for i in [159, 145]]  # Upper and lower points of left eye
    right_eye = [landmarks[i] for i in [386, 374]]  # Upper and lower points of right eye

    left_eye_ratio = abs(left_eye[0].y - left_eye[1].y)
    right_eye_ratio = abs(right_eye[0].y - right_eye[1].y)

    blink_threshold = 0.018  # Adjust this value based on testing

    if left_eye_ratio < blink_threshold and right_eye_ratio < blink_threshold:
        if not blink_state:  # Only detect when state changes (open → closed)
            blink_detected = True
            blink_cooldown = 10  # Prevents multiple bullets
            blink_state = True  # Set state to closed
            return True
    else:
        blink_state = False  # Reset when eyes are open

    return False

def get_head_direction(landmarks):
    """Detects head tilt based on nose landmark."""
    nose = landmarks[1].x

    if nose > 0.55:
        return "Right"
    elif nose < 0.45:
        return "Left"
    return "Center"

frame_skip = 2  # Process every 2nd frame
frame_counter = 0

running = True
while running:
    screen.fill(WHITE)

    # Capture Video Frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Face Every 2nd Frame
    frame_counter += 1
    if frame_counter % frame_skip == 0:
        result = face_mesh.process(rgb_frame)
    else:
        result = None  # Skip processing

    # Player Movement & Shooting Based on Blink
    if result and result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            direction = get_head_direction(landmarks)

            if direction == "Left" and player_x > 20:
                player_x -= player_speed
            elif direction == "Right" and player_x < WIDTH - 20:
                player_x += player_speed
            
            # Fire bullet on blink (only once per blink)
            if detect_blink(landmarks):
                bullets.append({'x': player_x, 'y': player_y})

    # Decrease Blink Cooldown
    if blink_cooldown > 0:
        blink_cooldown -= 1

    # Draw Player
    pygame.draw.rect(screen, BLUE, (player_x - 20, player_y, 40, 20))

    # Move Bullets
    for bullet in bullets[:]:
        bullet['y'] -= 10
        pygame.draw.circle(screen, RED, (bullet['x'], bullet['y']), 5)
        if bullet['y'] < 0:
            bullets.remove(bullet)

    # Move & Draw Enemies
    for enemy in enemies:
        pygame.draw.rect(screen, RED, (enemy['x'], enemy['y'], enemy_size, enemy_size))
        enemy['y'] += 1
        if enemy['y'] > HEIGHT:
            enemy['y'] = 50
            enemy['x'] = random.randint(50, WIDTH - 50)

    # Collision Detection
    for bullet in bullets[:]:
        for enemy in enemies[:]:
            if enemy['x'] < bullet['x'] < enemy['x'] + enemy_size and enemy['y'] < bullet['y'] < enemy['y'] + enemy_size:
                enemies.remove(enemy)
                bullets.remove(bullet)
                enemies.append({'x': random.randint(50, WIDTH - 50), 'y': 50})
                break

    # Quit Event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
    clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
