import os
import cv2
import numpy as np
import pandas as pd

def load_data(data_dir, labels_csv, img_size=(32, 32)):
    """
    Loads and preprocesses the data from the given directory.

    Args:
    - data_dir (str): Directory path where images are stored.
    - labels_csv (str): Path to the CSV file containing image labels.
    - img_size (tuple): The target image size (height, width).

    Returns:
    - images (numpy array): Preprocessed images as numpy array.
    - labels (numpy array): Labels corresponding to each image.
    """
    images = []
    labels = []

    # Read the CSV file for labels and image paths
    df = pd.read_csv(labels_csv)

    for index, row in df.iterrows():
        img_path = os.path.join(data_dir, row['Filename'])
        label = row['ClassId']

        if os.path.exists(img_path):  # Ensure the image exists
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)  # Resize images to the required size
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)
