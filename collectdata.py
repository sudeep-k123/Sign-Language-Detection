import os
import cv2
import numpy as np
import mediapipe as mp
from function import mediapipe_detection, draw_styled_landmarks, actions

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands

# Setup directories
directory = 'Image/'

# Ensure all directories exist
for letter in actions:
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

print("Instructions:")
print("1. Position your hand in the white rectangle")
print("2. Press a letter key (A-Z, except J and Q) to capture that sign")
print("3. Each action needs multiple samples (suggested: 30)")
print("4. Press 'q' to quit\n")

# Setup hands model for detection check
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Define ROI
        roi = frame[40:400, 0:300]

        # Process ROI for hand detection
        roi_rgb, results = mediapipe_detection(roi, hands)

        # Draw landmarks if hand detected
        if results.multi_hand_landmarks:
            draw_styled_landmarks(roi_rgb, results)
            cv2.putText(frame, "Hand Detected!", (310, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Hand Detected", (310, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw ROI rectangle
        cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

        # Display frames
        cv2.imshow("Data Collection", frame)
        cv2.imshow("Hand ROI", roi_rgb)

        # Check for key press
        key = cv2.waitKey(10) & 0xFF

        # Exit condition
        if key == ord('q'):
            break

        # Process letter keys
        if chr(key).upper() in actions:
            letter = chr(key).upper()
            # Count existing files
            letter_dir = os.path.join(directory, letter)
            if os.path.exists(letter_dir):
                count = len([f for f in os.listdir(letter_dir) if f.endswith('.png')])
            else:
                os.makedirs(letter_dir)
                count = 0

            save_path = os.path.join(letter_dir, f"{count}.png")

            # Check if hand is detected before saving
            if results.multi_hand_landmarks:
                cv2.imwrite(save_path, roi)
                print(f"Saved: {save_path} - Sample {count + 1} for {letter}")
            else:
                print(f"⚠️ No hand detected! Position your hand in the rectangle and try again.")

cap.release()
cv2.destroyAllWindows()