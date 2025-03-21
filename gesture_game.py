import os
import cv2
import time
import random
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------------------------
# Initialize MediaPipe Hands.
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ---------------------------
# Initialize Webcam.
# ---------------------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to open camera.")
    cap.release()
    exit()
frame_height, frame_width, _ = frame.shape

# ---------------------------
# Setup Emoji Rendering via PIL.
# ---------------------------
# Update the font path as necessary on your system.
font_path = "C:/Windows/Fonts/seguiemj.ttf"  # Example for Windows
emoji_font = ImageFont.truetype(font_path, 64)  # Adjust size as needed
# Map moves to emoji:
emoji_map = {
    "Rock": "✊",
    "Paper": "✋",
    "Scissors": "✌️"
}

# ---------------------------
# Level Settings.
# ---------------------------
rounds_per_level = {1: 3, 2: 5, 3: 7}
max_level = 3
moves_list = ["Rock", "Paper", "Scissors"]

# ---------------------------
# Gesture Recognition Functions.
# ---------------------------
def get_finger_status(hand_landmarks, handedness):
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

def recognize_rps(hand_landmarks, handedness):
    status = get_finger_status(hand_landmarks, handedness)
    if not (status['thumb'] or status['index'] or status['middle'] or status['ring'] or status['pinky']):
        return "Rock"
    if status['thumb'] and status['index'] and status['middle'] and status['ring'] and status['pinky']:
        return "Paper"
    if (not status['thumb']) and status['index'] and status['middle'] and (not status['ring']) and (not status['pinky']):
        return "Scissors"
    return None

# ---------------------------
# Visual Helper Function.
# ---------------------------
def draw_background_box(frame, text, pos, box_color=(50,50,50), text_color=(255,255,255), scale=1, thickness=2):
    (x, y) = pos
    (w, h) = (frame_width - 100, 40)
    cv2.rectangle(frame, (50, y - 35), (50 + w, y + 10), box_color, -1)
    cv2.putText(frame, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)

# ---------------------------
# Game Outcome Functions.
# ---------------------------
def decide_winner(player, computer):
    if player == computer:
        return "Tie"
    if (player == "Rock" and computer == "Scissors") or \
       (player == "Paper" and computer == "Rock") or \
       (player == "Scissors" and computer == "Paper"):
        return "You win!"
    else:
        return "Computer wins!"

def show_celebration():
    start = time.time()
    while time.time() - start < 3:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Congratulations! You Won the Game!", (50, frame_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 3)
        cv2.imshow("Rock Paper Scissors", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def show_pre_intimation():
    start = time.time()
    while time.time() - start < 3:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "You're unbeatable! Level clinched!", (50, frame_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 3)
        cv2.imshow("Rock Paper Scissors", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def get_replay_decision():
    start_time = time.time()
    decision = None
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Replay? Fist=NO, Palm=YES", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                move = recognize_rps(hand_landmarks, hand_handedness.classification[0].label)
                if move == "Rock":
                    decision = "no"
                    break
                elif move == "Paper":
                    decision = "yes"
                    break
        cv2.imshow("Replay Decision", frame)
        if decision is not None:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            decision = "no"
            break
    cv2.destroyWindow("Replay Decision")
    return decision

# ---------------------------
# Main Game Loop per Level.
# ---------------------------
def play_level(level):
    required_rounds = rounds_per_level[level]
    round_count = 0
    score_player = 0
    score_computer = 0

    while round_count < required_rounds:
        # Check if player is unbeatable.
        rounds_remaining = required_rounds - round_count
        if score_player > score_computer + rounds_remaining:
            show_pre_intimation()
            break

        # --- Countdown Phase (3 seconds) ---
        countdown = 3
        start_time = time.time()
        while time.time() - start_time < countdown:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            remaining = int(countdown - (time.time() - start_time)) + 1
            draw_background_box(frame, f"Get ready! {remaining}", (80,80))
            cv2.putText(frame, f"Round: {round_count+1}/{required_rounds}", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            cv2.putText(frame, f"Score You: {score_player}  Computer: {score_computer}", (50, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            cv2.imshow("Rock Paper Scissors", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None, None

        # --- Capture Player's Move ---
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        player_move = "None"
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            move = recognize_rps(hand_landmarks, handedness)
            if move is not None:
                player_move = move

        # --- Generate Computer's Move Randomly ---
        computer_move = random.choice(moves_list)

        # --- Decide Round Winner ---
        round_result = decide_winner(player_move, computer_move)
        if round_result == "Tie":
            tie_start = time.time()
            while time.time() - tie_start < 2:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "Tie! Replay the round.", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("Rock Paper Scissors", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return None, None
            continue

        round_count += 1
        if round_result == "You win!":
            score_player += 1
        elif round_result == "Computer wins!":
            score_computer += 1

        # --- Display Round Result for 3 Seconds with Emoji ---
        res_start = time.time()
        while time.time() - res_start < 3:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # Convert frame to PIL image for emoji rendering.
            pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_im)
            # Draw player's move using emoji if available.
            if player_move in emoji_map:
                draw.text((50, 50), f"Your Move: {emoji_map[player_move]}", font=emoji_font, fill=(255,0,0))
            else:
                draw.text((50, 50), f"Your Move: {player_move}", font=emoji_font, fill=(255,0,0))
            # Draw computer's move as text (you can also add emoji here if desired).
            if computer_move in emoji_map:
                draw.text((50, 130), f"Computer: {emoji_map[computer_move]}", font=emoji_font, fill=(0,255,255))
            else:
                draw.text((50, 130), f"Computer: {computer_move}", font=emoji_font, fill=(0,255,255))
            # Draw round result.
            draw.text((50, 210), round_result, font=emoji_font, fill=(0,255,0))
            # Draw score.
            draw.text((50, frame_height - 50), f"Score You: {score_player}  Computer: {score_computer}", font=emoji_font, fill=(255,0,255))
            # Convert back to OpenCV image.
            output_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            cv2.imshow("Rock Paper Scissors", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None, None

    return score_player, score_computer

# ---------------------------
# Game Flow: Progressive Levels.
# ---------------------------
current_level = 1

while current_level <= max_level:
    print(f"Starting Level {current_level}...")
    result = play_level(current_level)
    if result is None:
        break
    score_player, score_computer = result
    print(f"Level {current_level} Result: You: {score_player}, Computer: {score_computer}")
    if score_player > score_computer:
        if current_level == max_level:
            show_celebration()
            print("Congratulations! You qualified all levels and won the game!")
            break
        else:
            print(f"You qualified Level {current_level}! Proceeding to Level {current_level+1}...")
            current_level += 1
    else:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"You lost Level {current_level}. Score - You: {score_player}  Computer: {score_computer}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.putText(frame, "Replay? Fist=NO, Palm=YES", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.imshow("Rock Paper Scissors", frame)
        cv2.waitKey(1500)
        decision = get_replay_decision()
        if decision == "yes":
            print("Replaying current level...")
            continue
        else:
            print("Game Over.")
            break

cap.release()
cv2.destroyAllWindows()
