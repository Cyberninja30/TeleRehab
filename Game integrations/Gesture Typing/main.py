import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep, time
import numpy as np
import cvzone
from pynput.keyboard import Controller
import os
from datetime import datetime

# Create a folder for saving files if it doesn't exist
if not os.path.exists("typed_texts"):
    os.makedirs("typed_texts")

# Initializing the video capture
cap = cv2.VideoCapture(0)

# Get screen resolution dynamically
screen_width = 1920  # Default value
screen_height = 1080  # Default value

cap.set(3, screen_width)  # Set width to full screen width
cap.set(4, screen_height)  # Set height to full screen height

# Increased detection confidence for better accuracy
detector = HandDetector(detectionCon=0.85, maxHands=2)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        ["SAVE", " ", "CLEAR"]]
finalText = ""

keyboard = Controller()

class Button():
    def __init__(self, pos, text, size=[100, 100]):  # Increased size for better visibility
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []

# Adjust button positions to be fully visible
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        x_pos = 100 * j + 200
        y_pos = 100 * i + 200
        size = [100, 100]  # Standard button size

        if key == " ":
            size = [400, 100]  # Bigger spacebar
        elif key == "SAVE":
            size = [150, 100]  # Wider save button
            x_pos = 100  # Move to left
        elif key == "CLEAR":
            size = [160, 100]  # Wider clear button
            x_pos = 500  # Move to right

        buttonList.append(Button([x_pos, y_pos], key, size))

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)

        # Different colors for special buttons
        if button.text == "SAVE":
            color = (0, 255, 0)  # Green for save
        elif button.text == "CLEAR":
            color = (0, 0, 255)  # Red for clear
        else:
            color = (50, 50, 50)  # Dark gray for regular keys
            
        cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    return img

def save_text_to_file(text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"typed_texts/typed_text_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(text)
    return filename

def check_fist(hand_landmarks):
    # Check if all fingers are closed (fist gesture)
    fingers = detector.fingersUp(hand_landmarks)
    return sum(fingers) == 0  # Returns True if all fingers are down (fist)

def check_pinch(lmList):
    # Check for pinch with increased sensitivity
    pinch_detected = False
    finger_tips = [4, 8, 12, 16, 20]

    for i in range(len(finger_tips)):
        for j in range(i + 1, len(finger_tips)):
            dist = detector.findDistance(
                (lmList[finger_tips[i]][0], lmList[finger_tips[i]][1]),
                (lmList[finger_tips[j]][0], lmList[finger_tips[j]][1])
            )[0]
            if dist < 40:
                pinch_detected = True
                break
    return pinch_detected

# Initialize last click time for debouncing
last_click_time = time()
CLICK_DELAY = 0.3  # Delay between clicks

# Set full screen display
cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)
        img = drawAll(img, buttonList)

        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]

            # Check for fist gesture to exit
            if check_fist(hand1):
                if finalText:
                    filename = save_text_to_file(finalText)
                    print(f"Final text saved to {filename}")
                cv2.putText(img, "Closing Program...", (600, 100), 
                           cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                cv2.imshow("Image", img)
                cv2.waitKey(1000)
                break

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList1[8][0] < x+w and y < lmList1[8][1] < y+h:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

                    # Check for pinch to register click
                    current_time = time()
                    if check_pinch(lmList1) and (current_time - last_click_time) > CLICK_DELAY:
                        if button.text == "SAVE":
                            if finalText:
                                filename = save_text_to_file(finalText)
                                print(f"Saved to {filename}")
                        elif button.text == "CLEAR":
                            finalText = ""
                        else:
                            keyboard.press(button.text)
                            finalText += button.text

                        last_click_time = current_time
                        sleep(0.3)

        # Display text input area
        cv2.rectangle(img, (100, 50), (1800, 150), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, finalText, (110, 130), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        # Display instructions
        cv2.putText(img, "Make a fist to close", (50, screen_height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(5) & 0xFF == 27:
            if finalText:
                save_text_to_file(finalText)
            break

    except Exception as e:
        print(f"Error: {e}")
        continue

cap.release()
cv2.destroyAllWindows()
