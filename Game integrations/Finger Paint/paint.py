import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Hand Gesture Paint", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Gesture Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

canvas = None
prev_x, prev_y = None, None
fullscreen = False

# Tools and Colors
tool_index = 0
tool_names = ["Pencil", "Marker", "Brush", "Highlighter", "Eraser", "Custom"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 0), (255, 255, 255)]
size_options = [10, 20, 30, 40, 50]
current_size = size_options[0]
custom_color = (255, 255, 255)

# Button Positions
buttons = {
    "Pencil": (10, 10, 100, 50),
    "Marker": (120, 10, 100, 50),
    "Brush": (230, 10, 100, 50),
    "Highlighter": (340, 10, 130, 50),
    "Eraser": (480, 10, 100, 50),
    "Color": (600, 10, 100, 50),
    "Clear": (710, 10, 100, 50)
}

# Color Palette Position and Size
color_palette_x = 10
color_palette_y = 70
color_palette_height = 30
color_width = 30
color_spacing = 5

# Function to Draw Buttons
def draw_buttons(frame):
    for label, (x, y, w, h) in buttons.items():
        color = (200, 200, 200) if label != tool_names[tool_index] else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def draw_color_palette(frame):
    global color_palette_x, color_palette_y, color_width, color_spacing
    x = color_palette_x
    for color in colors:
        cv2.rectangle(frame, (x, color_palette_y), (x + color_width, color_palette_y + color_palette_height), color, -1)
        x += color_width + color_spacing

def check_color_selection(x, y):
    global color_palette_x, color_palette_y, color_width, color_spacing, colors, custom_color, color_palette_height
    x_start = color_palette_x
    for i, color in enumerate(colors):
        if x_start < x < x_start + color_width and color_palette_y < y < color_palette_y + color_palette_height:
            custom_color = color
            return True
        x_start += color_width + color_spacing
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None or canvas.shape[:2] != (h, w):

        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if canvas is None:

        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            x, y = int(index_tip.x * w), int(index_tip.y * h)
            distance = np.linalg.norm(
                np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            )
            drawing = distance > 0.04

            # Check Color Selection
            if check_color_selection(x, y):
                tool_index = 5

            # Button Interaction and Clear Functionality
            for label, (bx, by, bw, bh) in buttons.items():
                if bx < x < bx + bw and by < y < by + bh:
                    if label in tool_names:
                        tool_index = tool_names.index(label)
                    elif label == "Clear":
                        canvas.fill(0)

            # Adjust Brush Size Based on Finger Count
            finger_count = 0
            for finger in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                           mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                           mp_hands.HandLandmark.RING_FINGER_TIP,
                           mp_hands.HandLandmark.PINKY_TIP]:
                if hand_landmarks.landmark[finger].y < hand_landmarks.landmark[finger - 2].y:
                    finger_count += 1

            if finger_count > len(size_options) - 1:
                finger_count = len(size_options) - 1

            current_size = size_options[finger_count] if 0 <= finger_count < len(size_options) else current_size

            # Drawing on Canvas
            if drawing and y > 110:
                draw_color = custom_color if tool_index == 5 else colors[tool_index]
                if tool_index == 4:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), current_size * 2)
                elif prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, current_size)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    canvas_resized = cv2.resize(canvas, (w, h))
    blended_frame = cv2.addWeighted(frame, 0.5, canvas_resized, 0.5, 0)

    draw_buttons(blended_frame)
    draw_color_palette(blended_frame)

    cv2.putText(blended_frame, f"Tool: {tool_names[tool_index]} | Size: {current_size}", (10, color_palette_y + color_palette_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Paint", blended_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Hand Gesture Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("Hand Gesture Paint", w, h)
        else:
            cv2.setWindowProperty("Hand Gesture Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Hand Gesture Paint", 1920, 1080)

cap.release()
cv2.destroyAllWindows()
