import torch
import torch.nn as nn
from typing import List

class EcgClassificationModel(nn.Module):
    def __init__(self, input_chanels = 1, input_length = 700, num_classes=8):
        super().__init__()
        
        self._pool_kernel_size = 1
        self._pool_stride = 2

        self._dropout = 0.68

        layers: List[nn.Module] = []

        layers.append(nn.Conv1d(input_chanels, 64, kernel_size=5, stride=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm1d(64, eps=1e-05))

        layers.append(nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm1d(64, eps=1e-05))
        layers.append(nn.MaxPool1d(kernel_size=self._pool_kernel_size, stride=self._pool_stride))

        layers.append(nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm1d(128, eps=1e-05))

        layers.append(nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm1d(128, eps=1e-05))

        layers.append(nn.Conv1d(128, 256, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm1d(256, eps=1e-05))
        layers.append(nn.MaxPool1d(kernel_size=self._pool_kernel_size, stride=self._pool_stride))

        self.features = nn.Sequential(*layers)

        input_tensor = torch.randn(1, input_chanels, input_length)

        output_tensor = self.features(input_tensor)
        print(output_tensor.size())

        _, self.last_dim, self.last_size = output_tensor.size()

        self.last_outputSize = self.last_size * self.last_dim
        linear_output = 2048 if self.last_outputSize > 2*2048 else 300

        print(f"Classifier input:{self.last_outputSize}")
        self.classifier = nn.Sequential(
            nn.Linear(self.last_outputSize, linear_output),
            nn.BatchNorm1d(linear_output, eps=1e-5),
            nn.Dropout(self._dropout),

            nn.Linear(linear_output, 150),
            nn.Linear(150, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.last_outputSize)
        x = self.classifier(x)
        return x
