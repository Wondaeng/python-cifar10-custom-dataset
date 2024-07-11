import numpy as np
import os
import time

def calculate_accuracy(preds, labels):
    # Ensure inputs are numpy arrays for element-wise comparison
    # Redundancy in main_train.make_prediction function
    y_true = np.array(labels)
    y_pred = np.array(preds)

    # Compare true and predicted values to get a boolean array
    correct_predictions = y_true == y_pred

    # Count the number of correct predictions
    # In the context of arithmetic operations, True == 1, False == 0
    correct_count = np.sum(correct_predictions)

    # Calculate accuracy
    acc = correct_count / len(y_true)

    return acc


def calculate_recall(preds, labels):
    # Ensure inputs are numpy arrays for element-wise comparison
    # Redundancy in main_train.make_prediction function
    y_true = np.array(labels)
    y_pred = np.array(preds)

    # Recall = TP / TP + FN (i.e., sensitivity)
    # Rate of the number of positive among positive data captured by the model
    # (data's point of view)
    tp = np.sum((preds == 1) & (preds == labels))  # preds == labels --> true
    fn = np.sum((preds == 0) & (preds != labels))  # preds != labels --> false
    recall = tp / (tp + fn)
    return recall


def calculate_precision(preds, labels):
    # Ensure inputs are numpy arrays for element-wise comparison
    # Redundancy in main_train.make_prediction function
    y_true = np.array(labels)
    y_pred = np.array(preds)

    # Precision = TP / TP + FP (i.e., positive predictive value (PPV))
    # How precise the model: the rate of true positive among ones predicted as positive
    # (predictions' point of view)
    tp = np.sum((preds == 1) & (preds == labels))
    fp = np.sum((preds == 1) & (preds != labels))
    precision = tp / (tp + fp)
    return precision


def calculate_f1_score(preds, labels):
    # Harmonic mean of precision and recall
    precision = calculate_precision(preds, labels)
    recall = calculate_recall(preds, labels)
    f1_score = 2*(precision*recall)/(precision + recall)
    return f1_score


def get_wrong_predictions(preds, labels, fnames):
    fp = (preds == 1) & (preds != labels)
    fn = (preds == 0) & (preds != labels)
    fp_fnames = fnames[fp].tolist()
    fn_fnames = fnames[fn].tolist()
    return fp_fnames, fn_fnames
