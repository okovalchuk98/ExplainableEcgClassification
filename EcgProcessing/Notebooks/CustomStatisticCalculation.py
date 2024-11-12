import numpy as np
import math
import torch

def calculate_confusion_matrix_by_indexes(predicted_indexes, target_indexes, elements_number, tolerance=2):
    matched_predicted = np.zeros_like(predicted_indexes, dtype=bool)
    for i, pred_idx in enumerate(predicted_indexes):
        for true_idx in target_indexes:
            if abs(pred_idx - true_idx) <= tolerance:
                matched_predicted[i] = True
                break

    tp = int(np.sum(matched_predicted))
    if tp > len(target_indexes):
        #print("Predicted positive more then targest")
        tp = len(target_indexes)

    fp = len(predicted_indexes) - tp
    fn = len(target_indexes) - tp
    tn = elements_number - (tp + fp + fn)
    return tp, tn, fp, fn

def calculate_statistic_by_indexes(predicted_indexes, target_indexes, elements_number, tolerance=2):
    tp, tn, fp, fn = calculate_confusion_matrix_by_indexes(predicted_indexes, target_indexes, elements_number, tolerance)
    accuracy, precision, recall, f1 = calculate_statistic(tp, tn, fp, fn)
    return accuracy, precision, recall, f1, (tp, tn, fp, fn)

def calculate_batch_statistic(predicted_r_peacks, target_r_peacks, elements_number, tolerance=2):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    confusion_matrix = (0,0,0,0)

    for (predicted, target) in zip(predicted_r_peacks, target_r_peacks):
        predicted_indexes = np.where(predicted > 0)[0]
        target_indexes = np.where(target > 0)[0]

        accuracy, precision, recall, f1, matrix = calculate_statistic_by_indexes(predicted_indexes, target_indexes, elements_number, tolerance)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        confusion_matrix = tuple(matrix[i] + confusion_matrix[i] for i in range(len(confusion_matrix)))


    return accuracies, precisions, recalls, f1s, confusion_matrix

def calculate_batch_tensor_statistic(predicted_r_peacks, target_r_peacks, elements_number, tolerance=2):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    confusion_matrix = (0,0,0,0)

    for (predicted, target) in zip(predicted_r_peacks, target_r_peacks):
        predicted_indexes = torch.nonzero(predicted > 0).flatten()  # Find non-zero indices
        target_indexes = torch.nonzero(target > 0).flatten() 

        accuracy, precision, recall, f1, matrix = calculate_statistic_by_indexes(predicted_indexes, target_indexes, elements_number, tolerance)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        confusion_matrix = tuple(matrix[i] + confusion_matrix[i] for i in range(len(confusion_matrix)))


    return accuracies, precisions, recalls, f1s, confusion_matrix

def calculate_statistic(tp, tn, fp, fn):
    recall = 0.0
    precision = 0.0
    f1 = 0.0

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp != 0:
        precision = tp / (tp + fp)

    if tp + fn != 0:
        recall = tp / (tp + fn)

    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    if math.isnan(accuracy):
        accuracy = 0.0
    if math.isnan(precision):
        precision = 0.0
    if math.isnan(recall):
        recall = 0.0
    if math.isnan(f1):
        f1 = 0.0
    return accuracy, precision, recall, f1
