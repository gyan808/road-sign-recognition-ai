from tensorflow.keras.utils import to_categorical

def evaluate_model(model, X_test, y_test):
    y_test_cat = to_categorical(y_test, num_classes=43)
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Test accuracy: {acc*100:.2f}%")
