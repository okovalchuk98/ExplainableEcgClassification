import torch
import torch.nn as nn
import numpy as np
from .EcgDataset import ECGDataSetItem
from .AutoencoderModels import ECGAutoencoder

class CnnRPeakDetector:

    def __init__(self, input_chanels, model_config_file_path, threshold = 0.1) -> None:
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.cnn_model = ECGAutoencoder(input_chanels)
        self.cnn_model.to(self.device)
        self.cnn_model.load_state_dict(torch.load(model_config_file_path))
        self.cnn_model.eval()

        self.threshold = threshold
        pass

    def detect_r_peak_indexes(self, ecg_input:ECGDataSetItem):
        predictiton = self.detect_r_peak_prediction(ecg_input)
        return self.__cnn_output_to_peak_indexes(predictiton.detach().cpu().numpy())

    def detect_r_peak_prediction(self, ecg_input:ECGDataSetItem):
        signal_data = ecg_input.signal_part
        knowledge_data = ecg_input.knowledge

        signal_data = signal_data.reshape((1,1,len(signal_data)))
        knowledge_data = signal_data.reshape((1,1,len(knowledge_data)))
        
        input_marged = np.concatenate((signal_data, knowledge_data), axis=1).astype(np.float32)
        input_marged = torch.from_numpy(input_marged).float()
        input_marged = input_marged.to(self.device)

        model_prediction = self.cnn_model(input_marged)
        model_prediction = model_prediction.reshape(model_prediction.shape[2])
        return model_prediction

    def detect_r_peak_indexes_for_array(self, signal_array, knowledge):
        prediction = self.__detect_r_peak_indexes_prediction(signal_array, knowledge)
        return self.__cnn_output_to_peak_indexes(prediction.detach().cpu().numpy())

    def __detect_r_peak_indexes_prediction(self, signal_data, knowledge_data):
        signal_data = signal_data.reshape((1,1,len(signal_data)))
        knowledge_data = signal_data.reshape((1,1,len(knowledge_data)))
        
        input_marged = np.concatenate((signal_data, knowledge_data), axis=1).astype(np.float32)
        input_marged = torch.from_numpy(input_marged).float()
        input_marged = input_marged.to(self.device)

        model_prediction = self.cnn_model(input_marged)
        model_prediction = model_prediction.reshape(model_prediction.shape[2])
        return model_prediction

    def detect_r_peak_indexes_for_tensor(self, signal_tensor, knowledge_tensor):
        model_prediction = self.detect_r_peack_prediction_for_tensor(signal_tensor, knowledge_tensor)
        return self.__cnn_batch_output_to_peak_indexes(model_prediction.detach().cpu().numpy())

    def detect_r_peack_prediction_for_tensor(self, signal_tensor, knowledge_tensor):
        input_marged = torch.cat((signal_tensor, knowledge_tensor), dim=1).float()
        input_marged = input_marged.to(self.device)

        model_prediction = self.cnn_model(input_marged)
        model_prediction = model_prediction.reshape(model_prediction.shape[0],model_prediction.shape[2])
        return model_prediction

    def __cnn_output_to_peak_indexes(self, predictions):
        post_processed_output = self.__post_processing_cnn_output(predictions)
        predicted_indexes = self.__probability_to_peak_indexes(post_processed_output)
        return predicted_indexes

    def __cnn_batch_output_to_peak_indexes(self, predictions):
        all_post_processed = []
        for prediction in predictions:
            post_processed_output = self.__post_processing_cnn_output(prediction)
            predicted_indexes = self.__probability_to_peak_indexes(post_processed_output)

            all_post_processed.append(predicted_indexes)
        return np.array(all_post_processed)

    def __cnn_batch_output_post_processing(self, predictions):
        all_post_processed = []
        for prediction in predictions:
            post_processed_output = self.__post_processing_cnn_output(prediction)
            all_post_processed.append(post_processed_output)
        return np.array(all_post_processed)

    def __post_processing_cnn_output(self, prediction):
        window_size = 70

        prediction = [item if item > self.threshold else 0 for item in prediction]

        prediction_size = len(prediction)
        post_processed = np.zeros(prediction_size, dtype=np.float32)
        index = 0
        while index < prediction_size:
            if prediction[index] > 0:
                prediction_range = prediction[index:window_size+index]
                max_value = np.argmax(prediction_range)
                post_processed[max_value + index] = 1

                index = index + window_size - 1
            index+=1

        return post_processed
    
    def post_processing_cnn_output(self, predictions):
        all_post_processed = []
        for prediction in predictions:
            post_processed = self.__post_processing_cnn_output(prediction)
            all_post_processed.append(post_processed)
        return np.array(all_post_processed)

    def __probability_to_peak_indexes(self, prediction):
        predicted_r_peacks_indexes = np.where(prediction > 0)[0]
        return predicted_r_peacks_indexes
