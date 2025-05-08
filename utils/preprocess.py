import os
import cv2
import numpy as np

def load_data(data_dir, img_size=(32, 32)):
    images = []
    labels = []

    for class_id in range(43):
        class_path = os.path.join(data_dir, f"{class_id:05d}")
        for img_file in os.listdir(class_path):
            if img_file.endswith(".ppm"):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(class_id)

    return np.array(images), np.array(labels)
