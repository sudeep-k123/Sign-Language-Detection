from function import *
from time import sleep
import os
import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm  # For progress bar

# Configuration
SHOW_IMAGES = True  # Set to True to see images during processing
NUM_WORKERS = 4  # Adjust based on your CPU cores
DATA_PATH = 'MP_Data'  # Updated to match function.py
ACTIONS = actions  # Use actions from function.py
NO_SEQUENCES = no_sequences  # Use from function.py
SEQUENCE_LENGTH = sequence_length  # Use from function.py

# Ensure data directories exist
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


# Function to process a single sequence
def process_sequence(action, sequence):
    print(f"Processing {action}, sequence {sequence}")
    try:
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as hands:
            # Load the image for this action/sequence
            image_path = os.path.join("Image", action, f"{sequence}.png")
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"⚠️ Could not load image {image_path}")
                return

            # Create a sequence of keypoints from this single image
            keypoints_list = []
            for frame_num in range(SEQUENCE_LENGTH):
                # Process the same image multiple times with slight variations
                # This simulates motion and creates multiple frames for training

                # Add small random noise to simulate different frames
                if frame_num > 0:
                    noise = np.random.normal(0, 0.005, frame.shape).astype(np.float32)
                    noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                else:
                    noisy_frame = frame

                # Perform hand detection
                image, results = mediapipe_detection(noisy_frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Display status
                if SHOW_IMAGES and frame_num == 0:  # Only show the first frame to avoid too many windows
                    cv2.putText(image, f'Collecting: {action}, Seq {sequence}',
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Processing', image)
                    cv2.waitKey(100)  # Brief delay

                # Extract keypoints
                keypoints = extract_keypoints(results)

                if keypoints is not None and len(keypoints) == 63:
                    keypoints_list.append(keypoints)
                else:
                    print(f"⚠️ Invalid keypoints for {action}, sequence {sequence}, frame {frame_num}")
                    # Use zeros if no valid keypoints
                    keypoints_list.append(np.zeros(63))

            # Save keypoints
            npy_folder = os.path.join(DATA_PATH, action, str(sequence))
            for frame_num, keypoints in enumerate(keypoints_list):
                npy_path = os.path.join(npy_folder, f"{frame_num}.npy")
                np.save(npy_path, keypoints)

            print(f"✅ Saved {SEQUENCE_LENGTH} frames for {action}, sequence {sequence}")

    except Exception as e:
        print(f"❌ Error processing {action}, sequence {sequence}: {e}")


# Process sequences - can use parallel or sequential processing
USE_PARALLEL = False  # Set to True for parallel processing

if USE_PARALLEL:
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_sequence, action, sequence)
                   for action in ACTIONS for sequence in range(NO_SEQUENCES)]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Sequences"):
            pass
else:
    # Sequential processing (easier for debugging)
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            process_sequence(action, sequence)

# Clean up OpenCV windows
cv2.destroyAllWindows()
print("Data processing complete!")