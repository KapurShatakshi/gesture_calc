import os
# Suppress TensorFlow and Mediapipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set up the webcam and canvas
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to open camera")
    cap.release()
    exit()
frame_height, frame_width, _ = frame.shape
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Global current drawing color (default white)
current_color = (255, 255, 255)

# Mapping for drawing finger thickness (in pixels)
finger_thickness = {
    "index": 5,
    "middle": 10,
    "ring": 15,
    "pinky": 20
}

# Define a color palette: each box has a BGR color and a rectangle (x, y, w, h)
palette = [
    {"color": (255, 0, 0), "pos": (10, 10, 50, 50)},      # Blue
    {"color": (0, 255, 0), "pos": (70, 10, 50, 50)},      # Green
    {"color": (0, 0, 255), "pos": (130, 10, 50, 50)},     # Red
    {"color": (0, 255, 255), "pos": (190, 10, 50, 50)},   # Yellow
    {"color": (255, 255, 255), "pos": (250, 10, 50, 50)}  # White
]

def draw_palette(frame, current_color):
    """Draws the color boxes on the frame. Highlights the active color."""
    for box in palette:
        x, y, w_box, h_box = box["pos"]
        color = box["color"]
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, -1)
        # If this box's color matches the current color, draw a border
        if color == current_color:
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 0), 3)

def detect_pinch(hand_landmarks, frame_width, frame_height, threshold=40):
    """
    Checks if a pinch gesture is occurring.
    Uses thumb tip (landmark 4) and index finger tip (landmark 8).
    Returns (True, pinch_point) if distance is below threshold, else (False, None).
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    x2, y2 = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
    distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    if distance < threshold:
        pinch_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        return True, pinch_point
    return False, None

def finger_up_status(hand_landmarks):
    """
    Returns a dictionary with True/False for each finger (excluding thumb)
    being "up" (tip above PIP joint).
    """
    lm = hand_landmarks.landmark
    status = {
        "index": lm[8].y < lm[6].y,
        "middle": lm[12].y < lm[10].y,
        "ring": lm[16].y < lm[14].y,
        "pinky": lm[20].y < lm[18].y
    }
    return status

def get_drawing_finger(hand_landmarks):
    """
    Returns the finger name (index, middle, ring, or pinky) if exactly one is up.
    Otherwise, returns None.
    """
    status = finger_up_status(hand_landmarks)
    up_fingers = [finger for finger, up in status.items() if up]
    if len(up_fingers) == 1:
        return up_fingers[0]
    return None

def is_erase_mode(hand_landmarks):
    """
    Returns True if the hand is in an open-palm configuration,
    i.e. all four fingers (index, middle, ring, pinky) are up.
    """
    status = finger_up_status(hand_landmarks)
    return all(status.values())

def erase_area(canvas, hand_landmarks, frame_width, frame_height):
    """
    Erases (clears to black) the region of the canvas covered by your palm.
    Computes a bounding rectangle from landmarks 0, 5, 9, 13, and 17.
    """
    pts = []
    for idx in [0, 5, 9, 13, 17]:
        pt = hand_landmarks.landmark[idx]
        pts.append((int(pt.x * frame_width), int(pt.y * frame_height)))
    pts_np = np.array(pts)
    x, y, box_w, box_h = cv2.boundingRect(pts_np)
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)

prev_x, prev_y = None, None
prev_draw_finger = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a natural feel
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw the color palette on the frame
    draw_palette(frame, current_color)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # First, check for a pinch gesture (color selection)
            pinch, pinch_point = detect_pinch(hand_landmarks, frame_width, frame_height)
            if pinch:
                # Draw a small circle at the pinch point for visual feedback
                cv2.circle(frame, pinch_point, 10, (0, 0, 0), -1)
                # Check if the pinch point falls within any palette box
                for box in palette:
                    bx, by, bw, bh = box["pos"]
                    if bx <= pinch_point[0] <= bx + bw and by <= pinch_point[1] <= by + bh:
                        current_color = box["color"]
                        # Once a color is selected, break out of the loop
                        break
                # Skip drawing if a pinch is active
                prev_x, prev_y, prev_draw_finger = None, None, None
                continue

            # Check for erase gesture (open palm)
            if is_erase_mode(hand_landmarks):
                erase_area(canvas, hand_landmarks, frame_width, frame_height)
                prev_x, prev_y, prev_draw_finger = None, None, None
            else:
                # Check if exactly one finger is up to draw
                draw_finger = get_drawing_finger(hand_landmarks)
                if draw_finger is not None:
                    # Get the drawing properties (thickness based on which finger)
                    thickness = finger_thickness.get(draw_finger, 5)
                    # Use the tip of the corresponding finger for drawing
                    finger_tip_idx = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}[draw_finger]
                    tip = hand_landmarks.landmark[finger_tip_idx]
                    x, y = int(tip.x * frame_width), int(tip.y * frame_height)
                    if prev_x is not None and prev_y is not None and prev_draw_finger == draw_finger:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness=thickness)
                    prev_x, prev_y, prev_draw_finger = x, y, draw_finger
                else:
                    prev_x, prev_y, prev_draw_finger = None, None, None

    # Blend the canvas with the frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Gesture Paint", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
