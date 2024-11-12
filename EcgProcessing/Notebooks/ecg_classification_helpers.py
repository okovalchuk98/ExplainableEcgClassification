import os
import numpy as np
import datetime

from SubModeles.EcgDataset import ECGDataSetItem
from SubModeles.CnnRPeakDetector import CnnRPeakDetector

def create_pathology_classification_dataset_by_ranges(ecgArrayData, rpeak_detection_model_path, left_offset, right_offset):
    detector = CnnRPeakDetector(2, rpeak_detection_model_path)

    beats = []
    range_selection_func = lambda peak: (peak - left_offset, peak + right_offset)

    index = 0
    for ecg_item in ecgArrayData:
        index += 1
        print(f"Processing item {index}/{len(ecgArrayData)}", end="\r")

        r_peacks = detector.detect_r_peak_indexes(ecg_item)
        beats = beats + split_db_ecg_beats_with_range(ecg_item, r_peacks, range_selection_func)

    return beats

def split_db_ecg_beats_with_range(ecg_item:ECGDataSetItem, detected_r_peacks, range_selection):
    all_supported_classes = { 'N': 0, 'V': 1, '/': 2, 'R': 3, 'L': 4, 'A': 5, 'F': 6, 'f': 7, 'NA': 8 }

    beats = []
    skiped_peak = 0
    for r_peack_index in detected_r_peacks:
        annotation_index = __get_peack_annotation_index(r_peack_index, ecg_item.r_peak_origin)
        if annotation_index == -1:
            skiped_peak += 1
            continue

        class_label = ecg_item.annotation_labels[annotation_index]
        if class_label not in all_supported_classes:
            class_label = "NA"

        left, right = range_selection(r_peack_index)
        if np.all([left > 0, right < len(ecg_item.signal_part)]):
            peak_time_ms = r_peack_index * (1000 / 400)
            original_time = datetime.datetime.strptime(ecg_item.original_signal_time, '%H:%M:%S.%f') + datetime.timedelta(milliseconds=peak_time_ms)
            original_time = original_time.strftime('%H:%M:%S.%f')

            beats.append({"ecg_image":  np.float32(ecg_item.signal_part[left:right]), "signal_name": ecg_item.signal_name, "peak_time": original_time, "class": class_label, "class_number": all_supported_classes[class_label], "rpeak_index": r_peack_index})

    return beats

def __get_peack_annotation_index(detected_peack_index, annotation_indexes):
    resul_index = -1
    for index in range(20):
        search_prev_index_result = np.where(annotation_indexes == detected_peack_index - index)
        if len(search_prev_index_result[0]) > 0:
            resul_index = search_prev_index_result[0][0]
            break
        
        search_next_index_result = np.where(annotation_indexes == detected_peack_index + index)
        if len(search_next_index_result[0]) > 0:
            resul_index = search_next_index_result[0][0]
            break

    return resul_index

def generate_knowlage_integration(signal, knowledge_window_size):
    knowledge_part = np.zeros(len(signal), dtype=np.float32)
    last_window_index = 0
    while last_window_index < len(signal) - knowledge_window_size * 0.7:
        start_window_index = last_window_index
        last_window_index = start_window_index + knowledge_window_size

        signal_part = signal[start_window_index:last_window_index]
        max_index = np.argmax(signal_part)

        one_side_pick_length = 20
        pick_signal_start_index, pick_index = (0, max_index) if max_index - one_side_pick_length < 0 else (max_index - one_side_pick_length, one_side_pick_length)
        pick_signal_part = signal_part[pick_signal_start_index:max_index + one_side_pick_length]

        left_signal = pick_signal_part[:pick_index]
        right_signal =pick_signal_part[pick_index:]
        if len(left_signal) == 0 or len(right_signal) == 0:
            continue

        min_left = np.argmin(left_signal)
        min_right = np.argmin(right_signal)
        if abs(signal_part[max_index] - left_signal[min_left]) < 0.05 and abs(signal_part[max_index] - right_signal[min_right]) < 0.05:
            continue

        max_signal_index = max_index + start_window_index
        __set_knowledge(knowledge_part, max_signal_index)
        last_window_index = max_signal_index + 100

    return knowledge_part

def __set_knowledge(array, index):
    left_index = index - 20
    if left_index < 0:
        left_index = 0

    right_index = index + 20
    if right_index >= len(array):
        right_index = len(array) - 1

    array[left_index:right_index] = [1] * (right_index - left_index)
