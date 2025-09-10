import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import os
import random
from scipy import signal

# Check if required files exist
if not os.path.exists('eeg_emotion_model.h5'):
    print("Error: Model file 'eeg_emotion_model.h5' not found!")
    print("Please run train.py first to create the model.")
    exit(1)

if not os.path.exists('scaler.pkl'):
    print("Error: Scaler file 'scaler.pkl' not found!")
    print("Please run train.py first to create the scaler.")
    exit(1)

# Load the trained model and scaler
model = tf.keras.models.load_model('eeg_emotion_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Label mapping (reverse for predictions)
label_mapping_reverse = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

print("Model and scaler loaded successfully.")
print("Waiting for EEG simulator data...")

# Function to get live EEG features (read from simulator CSV)
def get_live_eeg_features():
    try:
        # Read the full feature vector from simulator (2548 features)
        df = pd.read_csv('live_eeg.csv')

        # The simulator now generates all 2548 features directly
        # Extract feature values (skip any non-numeric columns)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if feature_cols:
            # Use the first row of features
            features = df[feature_cols].iloc[0].values
            return features
        else:
            # Fallback: try to use all numeric columns
            numeric_data = df.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                features = numeric_data.iloc[0].values
                # Ensure we have 2548 features
                if len(features) < 2548:
                    features = np.pad(features, (0, 2548 - len(features)), 'constant')
                elif len(features) > 2548:
                    features = features[:2548]
                return features

        print("No suitable feature columns found in CSV. Using random data.")
        return np.random.rand(2548)

    except FileNotFoundError:
        print("Simulator file not found. Using random data.")
        return np.random.rand(2548)
    except Exception as e:
        print(f"Error reading simulator data: {e}. Using random data.")
        return np.random.rand(2548)

# Function to retrain the model with new data
def retrain_model():
    try:
        print("Starting model retraining...")

        # Load original data
        import kagglehub
        path = kagglehub.dataset_download("birdy654/eeg-brainwave-dataset-feeling-emotions")
        csv_path = os.path.join(path, 'emotions.csv')
        original_data = pd.read_csv(csv_path)
        original_data['label'] = original_data['label'].map({'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2})

        print(f"Original data shape: {original_data.shape}")

        # Load new data if exists
        new_data_path = 'new_data.csv'
        if os.path.exists(new_data_path):
            new_data = pd.read_csv(new_data_path)
            print(f"New data shape: {new_data.shape}")

            # Handle different column formats
            if 'label' in new_data.columns:
                # New data has label column
                new_features = new_data.drop('label', axis=1)
                new_labels = new_data['label']
            else:
                # New data doesn't have label, use all columns as features
                new_features = new_data
                new_labels = pd.Series([0] * len(new_data))  # Default to NEGATIVE

            print(f"New features shape: {new_features.shape}")

            # Ensure feature count matches
            if new_features.shape[1] != original_data.shape[1] - 1:  # -1 for label column
                print(f"Feature count mismatch: original has {original_data.shape[1]-1}, new has {new_features.shape[1]}")
                # Pad or truncate to match
                target_features = original_data.shape[1] - 1
                if new_features.shape[1] < target_features:
                    padding = np.zeros((len(new_features), target_features - new_features.shape[1]))
                    new_features = np.concatenate([new_features.values, padding], axis=1)
                else:
                    new_features = new_features.values[:, :target_features]

                new_features = pd.DataFrame(new_features, columns=original_data.columns[:-1])

            # Create combined data properly
            new_data_with_label = new_features.copy()
            new_data_with_label['label'] = new_labels.values

            combined_data = pd.concat([original_data, new_data_with_label], ignore_index=True)
        else:
            combined_data = original_data

        # Preprocess
        X = combined_data.drop('label', axis=1)
        y = combined_data['label']
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

        # Check for any NaN or infinite values
        if X.isnull().any().any() or np.isinf(X.values).any():
            print("Warning: Found NaN or infinite values in features. Filling with zeros.")
            X = X.fillna(0).replace([np.inf, -np.inf], 0)

        X_scaled = scaler.fit_transform(X)  # Refit scaler on combined data
        print(f"Scaled features shape: {X_scaled.shape}")

        # Retrain (quick epochs for incremental update)
        model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)

        # Save updated model and scaler
        model.save('eeg_emotion_model.h5')
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Model retrained and saved.")

    except Exception as e:
        print(f"Error during retraining: {e}")
        print("Continuing with original model...")

# Main prediction loop
prediction_count = 0
retrain_interval = 10  # Retrain every 10 predictions

while True:
    # Get live features
    features = get_live_eeg_features()
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    prediction = model.predict(features_scaled, verbose=0)
    predicted_label = np.argmax(prediction)
    predicted_emotion = label_mapping_reverse[predicted_label]
    print(f"Predicted Emotion: {predicted_emotion}")
    
    # Get user feedback
    correct_label = input("Enter the correct emotion (NEGATIVE/NEUTRAL/POSITIVE) or 'skip' to continue: ").strip().upper()
    if correct_label in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
        # Save new data point with correct format
        label_num = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}[correct_label]

        # Create DataFrame with feature columns and label
        feature_dict = {f'feature_{i}': [features[i]] for i in range(len(features))}
        feature_dict['label'] = [label_num]

        new_row = pd.DataFrame(feature_dict)
        new_row.to_csv('new_data.csv', mode='a', header=not os.path.exists('new_data.csv'), index=False)
        prediction_count += 1

        print(f"Added new training sample with label: {correct_label}")

        # Retrain if interval reached
        if prediction_count % retrain_interval == 0:
            retrain_model()
    
    # Continue loop
    if input("Continue predicting? (y/n): ").strip().lower() != 'y':
        break
