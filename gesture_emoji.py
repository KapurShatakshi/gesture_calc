import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set up webcam.
cap = cv2.VideoCapture(0)

# Update this path to a font that supports emoji on your system.
font_path = "C:/Windows/Fonts/seguiemj.ttf"  # Windows example; change as needed.
font = ImageFont.truetype(font_path, 64)      # Adjust size as needed.

def get_finger_status(hand_landmarks, handedness):
    """
    Determines if each finger is extended.
    For the thumb:
      - For a "Right" hand (mirrored view): extended if tip (landmark 4) is left of IP (landmark 3).
      - For a "Left" hand: extended if tip is right of IP.
    For other fingers, a finger is considered extended if its tip is above its PIP joint.
    Returns a dictionary with boolean values.
    """
    lm = hand_landmarks.landmark
    status = {}
    if handedness == "Right":
        status['thumb'] = lm[4].x < lm[3].x
    else:
        status['thumb'] = lm[4].x > lm[3].x
    status['index'] = lm[8].y < lm[6].y
    status['middle'] = lm[12].y < lm[10].y
    status['ring'] = lm[16].y < lm[14].y
    status['pinky'] = lm[20].y < lm[18].y
    return status

def recognize_gesture(hand_landmarks, handedness):
    """
    Recognizes a gesture based on the finger status.
    Returns an emoji string based on these simple rules:
      - Open Palm (all extended): 😄
      - Fist (none extended): ✊
      - Peace sign (only index and middle extended): ✌️
      - Thumbs up (only thumb extended): 👍
      - Otherwise, returns an empty string.
    """
    status = get_finger_status(hand_landmarks, handedness)
    if status['thumb'] and status['index'] and status['middle'] and status['ring'] and status['pinky']:
        return "😄"
    if not any(status.values()):
        return "✊"
    if status['index'] and status['middle'] and (not status['thumb']) and (not status['ring']) and (not status['pinky']):
        return "✌️"
    if status['thumb'] and (not status['index']) and (not status['middle']) and (not status['ring']) and (not status['pinky']):
        return "👍"
    return ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for natural interaction.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Convert the frame (as an array) to a PIL image.
    pil_im = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_im)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks using MediaPipe (optional for debugging).
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = handedness_info.classification[0].label  # "Left" or "Right"
            emoji = recognize_gesture(hand_landmarks, label)
            if emoji:
                # Get coordinates of index fingertip (landmark 8).
                x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                y = int(hand_landmarks.landmark[8].y * frame.shape[0])
                # Draw the emoji using PIL's draw.text.
                draw.text((x, y - 64), emoji, font=font, fill=(0, 255, 0))

    # Convert the PIL image (with emoji overlay) back to a numpy array and then BGR color space.
    output_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imshow("Hand Gesture Recognizer", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
