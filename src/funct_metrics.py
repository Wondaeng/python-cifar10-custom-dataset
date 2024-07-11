import numpy as np


def calculate_accuracy(preds, labels):
    y_true = np.array(labels)
    y_pred = np.array(preds)

    correct_predictions = y_true == y_pred
    acc = np.mean(correct_predictions)
    
    return acc