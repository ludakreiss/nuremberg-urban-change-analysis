import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return results, y_pred


def false_change_rate(y_true, y_pred):
    false_positive = ((y_pred == 1) & (y_true == 0)).sum()
    predicted_change = (y_pred == 1).sum()

    if predicted_change == 0:
        return 0.0

    return false_positive / predicted_change


def add_feature_noise(X, noise_std=0.01):
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise
