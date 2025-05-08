import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_model(X_train, y_train, X_val, y_val, img_size=(32, 32), num_classes=43):
    """
    Builds and trains the CNN model.

    Args:
    - X_train (numpy array): Training data.
    - y_train (numpy array): Training labels.
    - X_val (numpy array): Validation data.
    - y_val (numpy array): Validation labels.
    - img_size (tuple): Image size (height, width).
    - num_classes (int): Number of unique classes.

    Returns:
    - model (tensorflow.keras.Model): Trained CNN model.
    """
    model = Sequential()

    # Add layers to the model
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax for classification

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

    return model
