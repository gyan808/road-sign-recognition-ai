from utils.preprocess import load_data
from utils.train import train_model
from utils.test import evaluate_model
import os

data_path = os.path.join("dataset", "GTSRB", "Final_Training", "Images")
X, y = load_data(data_path)
print("Data loaded:", X.shape, y.shape)

# Normalize pixel values
X = X / 255.0

# Train the model
model = train_model(X, y)

# Optional: Save model
model.save("model/road_sign_model.h5")

# Evaluate (you can use the same data or split a test set separately)
evaluate_model(model, X, y)
