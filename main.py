import os
# Suppress TensorFlow's information and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.preprocess import load_data
from utils.train import train_model
from utils.test import evaluate_model

# Specify paths to the dataset and the generated CSV file
data_path = r"C:\Users\gyane\Desktop\road-sign-detection-ai\dataset\GTSRB\Training"
labels_csv = r"C:\Users\gyane\Desktop\road-sign-detection-ai\dataset\GT-final_train.csv"

# Load the data
X, y = load_data(data_path, labels_csv)

# Normalize the images
X = X / 255.0  # Normalize pixel values to [0, 1]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# Train the model
model = train_model(X_train, y_train, X_test, y_test)

# Evaluate the model
evaluate_model(model, X_test, y_test)
