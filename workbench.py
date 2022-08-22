import torch
import torchaudio
import torchsummary
import torch.nn as nn
import torchaudio.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from VA_RNN import VA_RNN, CustomLoss, VAMLDataSet
from VA_CNN import VA_CNN

import os
import librosa
import librosa.display
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt
import ptwt

# Load Data
input_dir = "./Data/Input_Files"
target_dir = "./Data/Target_Files"
annotations = "./Data/VAML_Annotation.csv"

config = 0

vox_test_dataset = VAMLDataSet("Vox", "Training", input_dir, target_dir, annotations, config)

wavelet = pywt.Wavelet('db4')

n = 3
inv = 0                                                         # 1 to invert result
fac = pow(-1, inv)
RNN_file = "./Data/Output_Files/Vox_RNN_" + str(config) + "_" + str(n+1) + ".wav"
CNN_file = "./Data/Output_Files/Vox_CNN_" + str(config) + "_" + str(n+1) + ".wav"
data, target, params = vox_test_dataset.__getitem__(n)
par_idx = (params.shape==data.shape)
print(params.shape)
df = pd.read_csv(annotations)
df = df[(df['device'] == "Phase") & (df['subset'] == "Testing")]
t = np.linspace(0 + 1.5*n, 1.5 * (n+1), int(1.5*44100))
b = df.iloc[n, 5:9].to_numpy()
p = torch.tensor((b[2]*np.abs( np.sin( b[0]/2*t + b[1]/2 ) ) + b[3])/b[2])
#plt.plot(t, p)
#plt.show()

def spec(tensor):
    stft = librosa.stft(tensor.numpy(), n_fft=512, hop_length=256)
    spectrogram = np.abs(stft.reshape(stft.shape[1], stft.shape[2]))
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return log_spectrogram

RNN_H_SZ = 64
CNN_H_SZ = 8
CNN_LAYS = 9


RNN_model = VA_RNN(sr=44100, n_params=5, hidden_size=RNN_H_SZ, device="Comp", config=None)
CNN_model = VA_CNN(sr=44100, nparams=5, layers=CNN_LAYS, channels=CNN_H_SZ, device="Comp", config=None)
RNN_model.load_state_dict(torch.load(f"VA_RNN_Vox_{RNN_H_SZ}.pth", map_location=torch.device('cpu')))
CNN_model.load_state_dict(torch.load(f"VA_CNN_Vox_{CNN_LAYS}_{CNN_H_SZ}.pth", map_location=torch.device('cpu')))

#demo_path = "./Comp_Demo_Dry.wav"
#demo, _ = torchaudio.load(demo_path)
#print(demo.shape)
#RNN_demo = RNN_model.process_samples(demo)
#CNN_demo = CNN_model.process_samples(demo)
#print(RNN_demo.shape)
#print(CNN_demo.shape)
#torchaudio.save("Comp_Demo_RNN.wav", RNN_demo, 44100, bits_per_sample=16)
#torchaudio.save("Comp_Demo_CNN.wav", CNN_demo, 44100, bits_per_sample=16)

print(sum(pr.numel() for pr in RNN_model.parameters()))
print(sum(pc.numel() for pc in CNN_model.parameters()))

#RNN_parameters = []
#for param in RNN_model.state_dict():
    #RNN_parameters.append(param)
    #print(param, "\t", RNN_model.state_dict()[param].size())
#print(RNN_model.state_dict()[RNN_parameters[1]])

#CNN_parameters = []
#for param in CNN_model.state_dict():
    #CNN_parameters.append(param)
    #print(param, "\t", CNN_model.state_dict()[param].size())
#print(CNN_model.state_dict()[CNN_parameters[1]])

transform_data = ptwt.wavedec(data, wavelet, mode='zero', level=1)

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