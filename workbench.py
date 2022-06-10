import torch
import torchaudio
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import os
import librosa
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt
import ptwt

## Dataset

class VAMLDataSet(Dataset):
    def __init__(self, device, subset, input_dir, target_dir, annotations):

        self.input_dir = input_dir
        self.target_dir = target_dir
        df = pd.read_csv(annotations)
        self.subset = subset
        self.annotations = df[(df['device'] == device) & (df['subset'] == subset) ]

        self.sample_rate = 44100
        self.num_samples = int(self.sample_rate * 1.5)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        input_path = self._get_input_path(index)
        target_path = self._get_target_path(index)
        params = torch.tensor(self.annotations.iloc[index, 4:9])
        input, sr1 = torchaudio.load(input_path)
        target, sr2 = torchaudio.load(target_path)

        if sr1 != sr2:
            raise ValueError(f"Input and target files at index {index} have different sample rates")
        if self.sample_rate != None and self.sample_rate != sr1:
            raise ValueError(f"Files at index {index} have a different sample rate from the rest of the data")

        return input, target, params

    def _get_input_path(self, index):
        batch = self.annotations.iloc[index, 3]
        file = "VAML_" + self.subset + "_" + str(batch) + ".wav"
        return os.path.join(self.input_dir, file)

    def _get_target_path(self, index):
        file = self.annotations.iloc[index, 0]
        return os.path.join(self.target_dir, file)

    def _align(self, tensor_1, tensor_2):
        tensor_1 = tensor_1[:, :self.num_samples]
        tensor_2 = tensor_2[:, :self.num_samples]
        return tensor_1, tensor_2

# Load Data
input_dir = "./Data/Input_Files"
target_dir = "./Data/Target_Files"
annotations = "./Data/VAML_Annotation.csv"

vox_train_dataset = VAMLDataSet("Vox", "Training", input_dir, target_dir, annotations)

wavelet = pywt.Wavelet('db4')
data, _, _ = vox_train_dataset.__getitem__(0)
transform_data = ptwt.wavedec(data, wavelet, mode='zero', level=2)

print(data.shape[1])
sum = 0
m = nn.ReLU()
for i in range(len(transform_data)):
    data_i = transform_data[i]
    data_i = m(data_i)
    transform_data2 = ptwt.wavedec(data_i, wavelet, mode='zero', level=2)
    for j in range(len(transform_data2)):
        sum += transform_data2[j].shape[1]
print(sum)