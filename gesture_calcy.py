import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands (allow up to 2 hands).
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam.
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to open camera.")
    cap.release()
    exit()

frame_height, frame_width, _ = frame.shape

# Define the Operator Window region.
# This window will contain the operator boxes.
op_win_x = 10
op_win_y = 10
op_win_w = 480   # width covering all operator boxes
op_win_h = 70    # height covering operator boxes

# Global arithmetic expression string.
expression = ""

# Variables for digit input stabilization.
stable_digit = None
stable_counter = 0
digit_cooldown = 0
COOLDOWN_FRAMES = 30    # frames to wait after a digit is registered
STABLE_THRESHOLD = 20   # frames to consider the gesture stable (~1 sec)

# Operator palette: each operator is drawn as a small box within the Operator Window.
operator_palette = [
    {"op": "+", "pos": (10, 10, 50, 50)},
    {"op": "-", "pos": (70, 10, 50, 50)},
    {"op": "*", "pos": (130, 10, 50, 50)},
    {"op": "/", "pos": (190, 10, 50, 50)},
    {"op": "(", "pos": (250, 10, 50, 50)},
    {"op": ")", "pos": (310, 10, 50, 50)},
    {"op": "=", "pos": (370, 10, 50, 50)},
    {"op": "C", "pos": (430, 10, 50, 50)}
]

def draw_operator_palette(img, operator_hover=None, hover_progress=0):
    """Draws the operator boxes and shows a progress bar for selection."""
    for box in operator_palette:
        x, y, w_box, h_box = box["pos"]
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (200, 200, 200), -1)
        cv2.putText(img, box["op"], (x + 10, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if operator_hover is not None and box["op"] == operator_hover:
            cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 0, 0), 3)
            progress_width = int((hover_progress / STABLE_THRESHOLD) * w_box)
            cv2.rectangle(img, (x, y + h_box - 5), (x + progress_width, y + h_box), (0, 255, 0), -1)

def count_fingers(hand_landmarks, handedness):
    """
    Counts the number of extended fingers in one hand.
    For the thumb:
      - "Right" hand: counts if thumb tip (landmark 4) is left of its IP (landmark 3).
      - "Left" hand: counts if thumb tip is right of its IP.
    For other fingers, the tip must be above the PIP joint.
    """
    lm = hand_landmarks.landmark
    count = 0

    # Thumb.
    if handedness == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:  # Left hand.
        if lm[4].x > lm[3].x:
            count += 1

    # Index finger.
    if lm[8].y < lm[6].y:
        count += 1
    # Middle finger.
    if lm[12].y < lm[10].y:
        count += 1
    # Ring finger.
    if lm[16].y < lm[14].y:
        count += 1
    # Pinky.
    if lm[20].y < lm[18].y:
        count += 1

    return count

def process_operator(op):
    """Processes the selected operator and updates the arithmetic expression."""
    global expression
    if op == "=":
        try:
            result = str(eval(expression))
            expression = result
        except Exception:
            expression = "Error"
    elif op == "C":
        expression = ""
    else:
        expression += op

# Global variables for operator hover.
operator_hover = None
operator_hover_counter = 0

print("Gesture Calculator with Operator Window:")
print("  - When your index fingertip is inside the Operator Window, digit input is disabled.")
print("  - Hover your index fingertip over an operator box for >1 sec to select that operator.")
print("  - Digit input is computed as the sum of finger counts (modulo 10) from all hands (using handedness).")
print("  - '=' evaluates the expression and 'C' clears it. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirror effect.
    frame = cv2.flip(frame, 1)
    output = frame.copy()

    # Draw the Operator Window (a rectangle surrounding operator boxes).
    cv2.rectangle(output, (op_win_x, op_win_y), (op_win_x + op_win_w, op_win_y + op_win_h), (255, 255, 255), 2)

    # Draw the operator palette (inside the Operator Window).
    draw_operator_palette(output, operator_hover, operator_hover_counter)

    # Display the current expression as text at the bottom.
    cv2.putText(output, f"Expr: {expression}", (50, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- Operator Hover Processing ---
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get the index fingertip coordinates.
            index_x = int(hand_landmarks.landmark[8].x * frame_width)
            index_y = int(hand_landmarks.landmark[8].y * frame_height)

            # Check if the index fingertip is inside the Operator Window.
            if op_win_x <= index_x <= op_win_x + op_win_w and op_win_y <= index_y <= op_win_y + op_win_h:
                # Now check for individual operator boxes within the window.
                for box in operator_palette:
                    bx, by, bw, bh = box["pos"]
                    if bx <= index_x <= bx + bw and by <= index_y <= by + bh:
                        if operator_hover == box["op"]:
                            operator_hover_counter += 1
                        else:
                            operator_hover = box["op"]
                            operator_hover_counter = 1
                        break
                else:
                    operator_hover = None
                    operator_hover_counter = 0

                # If hovered long enough, process the operator.
                if operator_hover is not None and operator_hover_counter >= STABLE_THRESHOLD:
                    process_operator(operator_hover)
                    operator_hover = None
                    operator_hover_counter = 0
                    digit_cooldown = COOLDOWN_FRAMES

    # --- Check if any index fingertip is inside the Operator Window to skip digit input ---
    skip_digit = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_x = int(hand_landmarks.landmark[8].x * frame_width)
            index_y = int(hand_landmarks.landmark[8].y * frame_height)
            if op_win_x <= index_x <= op_win_x + op_win_w and op_win_y <= index_y <= op_win_y + op_win_h:
                skip_digit = True
                break

    # --- Digit Input Processing ---
    digit = None
    if results.multi_hand_landmarks and not skip_digit:
        total_fingers = 0
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # "Left" or "Right"
            total_fingers += count_fingers(hand_landmarks, label)
        digit = total_fingers % 10  # Ensure the digit is between 0 and 9.
    else:
        digit = None

    # Stabilize digit input so that it is added only after a steady gesture.
    if digit is not None:
        if stable_digit == digit:
            stable_counter += 1
            if stable_counter >= STABLE_THRESHOLD and digit_cooldown == 0:
                expression += str(digit)
                digit_cooldown = COOLDOWN_FRAMES
                stable_counter = 0
        else:
            stable_digit = digit
            stable_counter = 1

    if digit_cooldown > 0:
        digit_cooldown -= 1

    cv2.imshow("Gesture Calculator", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
