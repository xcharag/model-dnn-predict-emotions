import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('eeg_emotion_model.h5')

# Get the number of features from the model
n_features = model.input_shape[1]

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Label mapping
label_mapping = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

def get_live_eeg_features():
    """
    Placeholder function to get live EEG features.
    Replace this with actual code to extract features from your EEG device.
    Should return a numpy array of shape (n_features,) where n_features matches the training data.
    For example, compute FFT features from live EEG signal.
    """
    # For example, if reading from a file or device
    # Here, I'll simulate with random data for demonstration
    # In real use, extract FFT or other features from live EEG signal
    return np.random.rand(n_features)

def predict_emotion(features):
    # Preprocess
    features_scaled = scaler.transform(features.reshape(1, -1))
    # Predict
    prediction = model.predict(features_scaled)
    predicted_label = np.argmax(prediction)
    emotion = label_mapping[predicted_label]
    confidence = prediction[0][predicted_label]
    return emotion, confidence

# Main loop for live prediction
print("Starting live emotion prediction...")
while True:
    try:
        # Get live features
        features = get_live_eeg_features()
        # Predict
        emotion, confidence = predict_emotion(features)
        print(f"Predicted Emotion: {emotion} with confidence {confidence:.3f}")
        # Add delay or condition to control prediction rate
        import time
        time.sleep(1)  # Predict every second, adjust as needed
    except KeyboardInterrupt:
        print("Stopping prediction.")
        break
    except Exception as e:
        print(f"Error: {e}")
        break
