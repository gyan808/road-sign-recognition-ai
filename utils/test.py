import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data.

    Args:
    - model (tensorflow.keras.Model): The trained CNN model.
    - X_test (numpy array): Test data.
    - y_test (numpy array): Test labels.

    Returns:
    - None
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
