#import dependencies
import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    """Processes an image using MediaPipe Hands and returns the result."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False                   # Make image read-only for efficiency
    results = model.process(image)                  # Process the image
    image.flags.writeable = True                    # Make image writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

def draw_styled_landmarks(image, results):
    """Draws styled hand landmarks on an image using MediaPipe."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

def extract_keypoints(results):
    """Extracts hand landmarks as a flattened NumPy array."""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Use only the first detected hand
        keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        return keypoints
    else:
        return np.zeros(21 * 3)  # Return a zero-filled array if no hand is detected

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Define the gestures/actions for training
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

# Number of video sequences per action
no_sequences = 30

# Number of frames per sequence
sequence_length = 30