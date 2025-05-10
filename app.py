import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
try:
    with open("model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model.weights.h5")
    model.summary()  # Print model architecture
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# Define gesture classes
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                    'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])


# Helper function to detect hands
def mediapipe_detection(image, hands):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# Extract keypoints from hand landmarks - FIXED FUNCTION
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        # Get only the first hand
        hand = results.multi_hand_landmarks[0]
        # Extract 21 landmarks with x, y, z coordinates (total 63 values)
        keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark]).flatten()
        print(f"✅ Hand detected: {len(keypoints)} values extracted")
        return keypoints  # Should be 63 values (21 landmarks × 3 coordinates)
    else:
        # Return zero array if no hand detected
        print("⚠️ No hand detected")
        return np.zeros(21 * 3)  # 63 zeros


# Draw hand landmarks on frame
def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


# Initialize variables
sequence = []
sentence = []
threshold = 0.5  # Reduced threshold for better detection

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

print("🎥 Webcam started. Press 'q' to exit.")

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        # Define region for hand detection - adjust as needed for your setup
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

        # Process frame for hand landmarks
        image, results = mediapipe_detection(cropframe, hands)
        draw_landmarks(image, results)

        # Extract keypoints and add to sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames

        # Make prediction when we have enough frames
        # In app.py, add more debug information:
        if len(sequence) == 30:
            X_input = np.array(sequence).reshape((1, 30, 63))

            # Debug information
            print(f"Input shape: {X_input.shape}")

            # Make prediction with verbose=0 to reduce output noise
            res = model.predict(X_input, verbose=0)[0]

            # Show all predictions with values for debugging
            for i, (action, score) in enumerate(zip(actions, res)):
                print(f"{action}: {score:.4f}")

            predicted_action = actions[np.argmax(res)]
            probability = res[np.argmax(res)]

            print(f"👉 Prediction: {predicted_action}, Confidence: {probability:.4f}")

            # Lower threshold for testing
            if probability > 0.3:  # Try with lower threshold first
                if len(sentence) == 0 or predicted_action != sentence[-1]:
                    sentence.append(predicted_action)

            # Keep sentence manageable
            if len(sentence) > 5:
                sentence = sentence[-5:]

        # Display results
        # Make sure this part is present in your display loop:
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)

        if len(sequence) == 30:  # Remove probability check for debugging
            cv2.putText(frame, f"{predicted_action} ({probability * 100:.1f}%)", (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Gesture", (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display sentence at bottom
        if sentence:
            text = ' '.join(sentence)
            cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, text, (3, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show both the main frame and the cropped region
        cv2.imshow('Gesture Recognition', frame)
        cv2.imshow('Hand ROI', cropframe)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("🛑 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()