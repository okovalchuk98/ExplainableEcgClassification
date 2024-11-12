from torch.utils.data import Dataset

# Datasets
class ECGDataSetItem:
    def __init__(self, signal_part, targets, knowledge, r_peak_origin, annotation_labels, fs, signal_name, original_signal_time):
        self.signal_part = signal_part
        self.targets = targets
        self.knowledge = knowledge
        self.r_peak_origin = r_peak_origin
        self.annotation_labels = annotation_labels
        self.fs = fs
        self.signal_name = signal_name
        self.original_signal_time = original_signal_time

class ECGDataset(Dataset):
    def __init__(self, data_items):
        self.data_items = data_items

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]

        origin_signal = data_item.signal_part
        knowledge = data_item.knowledge
        target = data_item.targets
        origin_r_peak = data_item.r_peak_origin
        annotation = data_item.annotation_labels

        return origin_signal, target, knowledge, origin_r_peak, annotation
