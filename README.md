# EEG Emotion Classification with Deep Learning

## Overview

This repository contains Python code for performing emotion classification using EEG (Electroencephalogram) data. Emotion classification from EEG signals is an important application in neuroscience and human-computer interaction. The code leverages deep learning techniques to analyze EEG data and predict emotional states.

## Key Features

1. **Data Loading and Preprocessing:**
   - Loads EEG data from a CSV file containing features extracted from EEG signals.
   - Converts emotion labels (e.g., 'NEGATIVE', 'NEUTRAL', 'POSITIVE') to numerical values (0, 1, 2) for classification.

2. **Data Visualization:**
   - Utilizes various visualization techniques to gain insights into the EEG data:
     - Pie chart to display the distribution of emotions in the dataset.
     - Time-series plot to visualize EEG signals over time.
     - Power Spectral Density (PSD) plot for spectral analysis.
     - Correlation heatmap to understand feature correlations.
     - t-SNE (t-Distributed Stochastic Neighbor Embedding) visualization for dimensionality reduction.

3. **Feature Significance Analysis:**
   - Conducts statistical tests (t-tests) to identify significant and non-significant features for each emotion category.
   - Presents results through bar charts, making it easy to understand which features contribute most to emotion prediction.

4. **Advanced Preprocessing:**
   - Normalizes feature data using z-score normalization.
   - Splits the dataset into training and testing sets.

5. **Deep Learning Model:**
   - Builds a deep neural network model for emotion classification.
   - The model architecture consists of multiple dense layers with ReLU activation functions and dropout layers.
   - Compiles the model using the Adam optimizer and sparse categorical cross-entropy loss.
   - Trains the model on the training data with validation, tracking accuracy during training.

6. **Model Evaluation:**
   - Evaluates the trained model on the testing dataset, reporting the accuracy of emotion classification.
   - Generates a confusion matrix and a classification report for a comprehensive performance assessment.

7. **Random Sample Visualization:**
   - Selects a random EEG sample from the testing dataset and predicts its emotion label.
   - Plots EEG signals from the selected sample for visualization.

## Getting Started

### Prerequisites
- Python 3.7+
- Required packages: tensorflow, scikit-learn, pandas, numpy, matplotlib, seaborn, scipy, plotly

Install the packages using:
```
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn scipy plotly
```

### Data
Download the EEG emotions dataset from Kaggle: [EEG Brainwave Dataset: Feeling Emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)

Place the `emotions.csv` file in the project directory.

### Training the Model
Run the training script:
```
python train.py
```
This will train the model and save it as `eeg_emotion_model.h5` along with the scaler as `scaler.pkl`.

### Live Prediction
To perform live emotion prediction:
1. Modify the `get_live_eeg_features()` function in `predict.py` to extract features from your EEG device.
2. Run the prediction script:
```
python predict.py
```
This will start a loop that continuously predicts emotions from live EEG data.

For detailed usage instructions and explanations, please refer to the code comments.

## Customization

Feel free to customize and extend this code for your specific EEG emotion classification projects.

