from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration - use values from function.py for consistency
DATA_PATH = 'MP_Data'  # Match with function.py and data.py
ACTIONS = actions
NO_SEQUENCES = no_sequences
SEQUENCE_LENGTH = sequence_length
TEST_SIZE = 0.2  # Increased test size for better validation
EPOCHS = 100
BATCH_SIZE = 16
LOG_DIR = os.path.join('Logs')

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Create label mapping
label_map = {label: num for num, label in enumerate(ACTIONS)}
print(f"Label Map: {label_map}")

# Load sequences and labels
sequences, labels = [], []
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")

            if not os.path.exists(npy_path):
                print(f"Warning: Missing file {npy_path}")
                window.append(np.zeros(63))  # Fill with zeros if file is missing
            else:
                try:
                    res = np.load(npy_path)
                    window.append(res)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    window.append(np.zeros(63))

        sequences.append(window)
        labels.append(label_map[action])

print(f"Loaded {len(sequences)} sequences with {len(labels)} labels")

# Convert data to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels).astype('float32')

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X min: {X.min()}, X max: {X.max()}, X mean: {X.mean()}")

# Ensure dataset is large enough for splitting
if len(X) < 4:  # Need at least a few samples
    raise ValueError("Not enough data for train/test split. Increase `NO_SEQUENCES`.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Set up callbacks for training
tb_callback = TensorBoard(log_dir=LOG_DIR)
checkpoint_callback = ModelCheckpoint('best_model.h5',
                                     save_best_only=True,
                                     monitor='val_categorical_accuracy',
                                     mode='max')
early_stopping = EarlyStopping(monitor='val_loss',
                              patience=15,
                              restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.2,
                             patience=5,
                             min_lr=0.00001)

# Build improved LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

# Compile model
model.compile(optimizer='Adam',
             loss='categorical_crossentropy',
             metrics=['categorical_accuracy'])

# Print model summary
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, checkpoint_callback, early_stopping, reduce_lr]
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show(block=False)
plt.pause(3)
plt.close()

# Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5')

print("Model saved successfully!")

# Optional: Make predictions on test data to verify
predictions = model.predict(X_test)
print(f"Prediction shape: {predictions.shape}")

# Show a few test predictions
for i in range(min(5, len(X_test))):
    true_label = ACTIONS[np.argmax(y_test[i])]
    pred_label = ACTIONS[np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100
    print(f"Sample {i}: True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2f}%")
