import torch
import torchaudio
import torchsummary
import torch.nn as nn
import torchaudio.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from VA_CNN import CustomLoss, VAMLDataSet

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

vox_test_dataset = VAMLDataSet("Vox", "Testing", input_dir, target_dir, annotations, config)

wavelet = pywt.Wavelet('db4')

n = 10
inv = 0                                                         # 1 to invert result
fac = pow(-1, inv)
RNN_file = "./Data/Output_Files/Vox_RNN_" + str(config) + "_" + str(n+1) + ".wav"
CNN_file = "./Data/Output_Files/Vox_CNN_" + str(config) + "_" + str(n+1) + ".wav"
data, target, params = vox_test_dataset.__getitem__(n)
RNN_res, _ = torchaudio.load(RNN_file)
CNN_res, _ = torchaudio.load(CNN_file)
gain = params[0].numpy()
loss_fn = CustomLoss()
rnn_loss = loss_fn(fac * RNN_res, target).numpy()
cnn_loss = loss_fn(fac * CNN_res, target).numpy()

def spec(tensor):
    stft = librosa.stft(tensor.numpy(), n_fft=512, hop_length=256)
    spectrogram = np.abs(stft.reshape(stft.shape[1], stft.shape[2]))
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return log_spectrogram

tar_spec = spec(target)
RNN_res_spec = spec(RNN_res)
CNN_res_spec = spec(CNN_res)

t = np.linspace(0, 1.5, 66150)
fig1, axs1 = plt.subplots(3, 1, constrained_layout=True, sharex=True, sharey=True)
fig1.suptitle("Gain=" + str(gain), fontsize=16)
axs1[0].plot(t, torch.transpose(data, 0, 1))
axs1[0].set_title("Input")
axs1[0].set_xlabel("Time (s)")
axs1[1].plot(t, fac * torch.transpose(RNN_res, 0, 1), t, torch.transpose(target, 0, 1))
axs1[1].set_title("RNN Output" + " Loss=" + str(rnn_loss))
axs1[1].set_xlabel("Time (s)")
axs1[2].plot(t, fac * torch.transpose(CNN_res, 0, 1), t, torch.transpose(target, 0, 1))
axs1[2].set_title("CNN Output" + " Loss=" + str(cnn_loss))
axs1[2].set_xlabel("Time (s)")

fig2, axs2 = plt.subplots(3, 1, constrained_layout=True, sharex=True, sharey=True)
fig2.suptitle("Gain=" + str(gain), fontsize=16)
im1 = librosa.display.specshow(tar_spec, x_axis='time', y_axis='log', ax=axs2[0])
fig2.colorbar(im1, ax=axs2[0], format="%+2.f dB")
axs2[0].set(title='Target')
im2 = librosa.display.specshow(RNN_res_spec, x_axis='time', y_axis='log', ax=axs2[1])
fig2.colorbar(im2, ax=axs2[1], format="%+2.f dB")
axs2[1].set(title='RNN' + " Loss=" + str(rnn_loss))
im3 = librosa.display.specshow(tar_spec, x_axis='time', y_axis='log', ax=axs2[2])
fig2.colorbar(im3, ax=axs2[2], format="%+2.f dB")
axs2[2].set(title='CNN' + " Loss=" + str(cnn_loss))


plt.show()

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