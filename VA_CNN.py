############################################################################################
#
# Ry Currier
# 2022-05-18
# CNN Traing Script
#
############################################################################################

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

class VA_CNN(nn.Module):
    def __init__(self, channels=8, blocks=2, layers=9, dilation_growth=2, kernel_size=3):
        super(VA_CNN, self).__init__()
        self.channels = channels
        self.layers = layers
        self.dilation_growth = dilation_growth
        self.kernel_size = kernel_size
        self.blocks = nn.ModuleList()
        for b in range(blocks):
            self.blocks.append(VA_CNN_Block(1 if b == 0 else channels,
                                                                channels,
                                                                dilation_growth,
                                                                kernel_size,
                                                                layers))
        self.blocks.append(nn.Conv1d(channels*layers*blocks, 1, 1, 1, 0))


    def forward(self, x):
        x = x.permute(1, 2, 0)
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])
        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)
            z[:, n*self.channels*self.layers:(n+1)*self.channels*self.layers, :] = zn
        return self.blocks[-1](z).permute(2, 0, 1)

    def train(self, data_in, data_tar, val_in, val_tar, 
                num_epochs, loss_fn, optim, batch_size):
        for n in range(num_epochs):
            shuffle = torch.randperm(data_in.shape[1])
            ep_loss = 0
            for batch in range(math.ceil(shuffle.shape[0] / batch_size)):
                self.zero_grad()
                batch_in = data_in[:, shuffle[batch*batch_size:(batch+1)*batch_size], :]
                batch_tar = data_tar[:, shuffle[batch*batch_size:(batch+1)*batch_size], :]
                output = self(batch_in)
                loss = loss_fn(output, batch_tar)
                loss.backward()
                optim.step()
                ep_loss += loss
            print(f"Epoch {n+1} Loss: {loss}")
            output, val_loss = self.process_samples(val_in, val_tar, loss_fn)
            print(f"Validation Loss: {val_loss}")
    
    def process_samples(self, data_in, data_out, loss_fn):
        with torch.no_grad():
            output = self(data_in)
            loss = loss_fn(output, data_out)
        return output, loss

    def save_model(self, file_name, dir=''):
        pass

class VA_CNN_Block(nn.Module):
    def __init__(self, chan_input, chan_output, dilation_growth, kernel_size, layers):
        super(VA_CNN_Block, self).__init__()
        self.channels = chan_output
        dilations = [dilation_growth ** lay for lay in range(layers)]
        self.layers = nn.ModuleList()

        for dil in dilations:
            self.layers.append(VA_CNN_Layer(chan_input, chan_output, dil, kernel_size))
            chan_input = chan_output

    def forward(self, x):
        z = torch.empty([x.shape[0], len(self.layers)*self.channels, x.shape[2]])
        for n, layer in enumerate(self.layers):
            x, zn = layer(x)
            z[:, n*self.channels:(n+1)*self.channels, :] = zn
        return x, z 

class VA_CNN_Layer(nn.Module):
    def __init__(self, chan_input, chan_output, dilation, kernel_size):
        super(VA_CNN_Layer, self).__init__()
        self.channels = chan_output
        self.conv= nn.Conv1d(in_channels=chan_input,
                            out_channels=chan_output * 2,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=0,
                            dilation=dilation)
        self.mix = nn.Conv1d(in_channels=chan_output,
                            out_channels=chan_output,
                            kernel_size=1,
                            stride=1,
                            padding=0)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        z = torch.tanh(y[:, 0:self.channels, :]) \
            * torch.sigmoid(y[:, self.channels:, :])
        z = torch.cat((torch.zeros(residual.shape[0],
                                    self.channels,
                                    residual.shape[2] - z.shape[2]), z), dim=2)
        x = self.mix(z) + residual
        return x, z

# Error to Signal Ratio Loss
class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean(torch.pow(target - output, 2))
        energy = torch.mean(torch.pow(target, 2)) + 1e-5
        loss.div_(energy)
        return loss

class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean(torch.pow(torch.mean(target, 0) - torch.mean(output, 0), 2))
        energy = torch.mean(torch.pow(target, 2)) + 1e-5
        loss.div_(energy)
        return loss

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.ESRLOSS = ESRLoss()
        self.DCLOSS = DCLoss()

    def forward(self, output, target):
        esrloss = self.ESRLOSS(output, target)
        dcloss = self.DCLOSS(output, target)
        return 0.75 * esrloss + 0.25 * dcloss

class DataLoader():
    def __init__(self):
        self.sample_rate = None
        self.in_data = dict.fromkeys(["train", "val", "test"])
        self.out_data = dict.fromkeys(["train", "val", "test"])
        self.samples_dict = dict.fromkeys(["train", "val", "test"])

    def load(self, in_train, in_val, in_test, out_train, out_val, out_test):
        train_in, self.sample_rate = torchaudio.load(in_train)
        train_out, self.sample_rate = torchaudio.load(out_train)
        self.in_data["train"], self.out_data["train"], self.samples_dict["train"] = self._align(train_in, train_out)
        val_in, self.sample_rate = torchaudio.load(in_val)
        val_out, self.sample_rate = torchaudio.load(out_val)
        self.in_data["val"], self.out_data["val"], self.samples_dict["val"]  = self._align(val_in, val_out)
        test_in, self.sample_rate = torchaudio.load(in_test)
        test_out, self.sample_rate = torchaudio.load(out_test)
        self.in_data["test"], self.out_data["test"], self.samples_dict["test"]  = self._align(test_in, test_out)
        return self.in_data, self.out_data, self.samples_dict

    def _align(self, tensor_1, tensor_2):
        if tensor_1.shape[1] > tensor_2.shape[1]:
            tensor_1 = tensor_1[:, :tensor_2.shape[1]]
            num_samples = tensor_2.shape[1]
        elif tensor_1.shape[1] < tensor_2.shape[1]:
            tensor_2 = tensor_2[:, :tensor_1.shape[1]]
            num_samples = tensor_1.shape[1]
        return tensor_1, tensor_2, num_samples

# Load Data
loader = DataLoader()

in_train_path = "./training_data.wav"
out_train_path = "./VOX_training_data.wav"
in_val_path = "./validation_data.wav"
out_val_path = "./VOX_validation_data.wav"
in_test_path = "./testing_data.wav"
out_test_path = "./VOX_testing_data.wav"

in_dict, out_dict, num_samples_dict = loader.load(in_train_path, in_val_path, 
                                                    in_test_path, out_train_path,
                                                    out_val_path, out_test_path)
sample_rate = loader.sample_rate

frame_len = math.floor(sample_rate / 2)
num_samples_train = num_samples_dict["train"]
num_samples_val = num_samples_dict["val"]
num_samples_test = num_samples_dict["test"]
num_frames_train = math.floor(num_samples_train / frame_len)
num_frames_val = math.floor(num_samples_val / frame_len)
num_frames_test = math.floor(num_samples_test / frame_len)

train_input = torch.empty((frame_len, num_frames_train, 1))
train_target = torch.empty((frame_len, num_frames_train, 1))
val_input = torch.empty((frame_len, num_frames_val, 1))
val_target = torch.empty((frame_len, num_frames_val, 1))
test_input = torch.empty((frame_len, num_frames_test, 1))
test_target = torch.empty((frame_len, num_frames_test, 1))

for i in range(num_frames_train):
    train_input[:, i, :] = in_dict["train"].reshape((num_samples_train, 1))[i*frame_len:(i+1)*frame_len, :]
    train_target[:, i, :] = out_dict["train"].reshape((num_samples_train, 1))[i*frame_len:(i+1)*frame_len, :]

for i in range(num_frames_val):
    val_input[:, i, :] = in_dict["val"].reshape((num_samples_val, 1))[i*frame_len:(i+1)*frame_len, :]
    val_target[:, i, :] = out_dict["val"].reshape((num_samples_val, 1))[i*frame_len:(i+1)*frame_len, :]

for i in range(num_frames_test):
    test_input[:, i, :] = in_dict["test"].reshape((num_samples_test, 1))[i*frame_len:(i+1)*frame_len, :]
    test_target[:, i, :] = out_dict["test"].reshape((num_samples_test, 1))[i*frame_len:(i+1)*frame_len, :]

print(train_input.shape)
print(train_target.shape)
print(val_input.shape)
print(val_target.shape)
print(test_input.shape)
print(test_target.shape)

# Training

start_time = time.time()

NUM_EPOCHS = 2
model = VA_CNN() 
loss_fn = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

model.train(train_input, train_target, val_input, val_target, 
             NUM_EPOCHS, loss_fn, optimizer, 40)

output, loss = model.process_samples(test_input, test_target, loss_fn)
output = output.reshape((1, output.shape[0]*output.shape[1]))
print(output.shape)
print(f"Test Loss: {loss}")
#torchaudio.save("./VOX_CNN_output.wav", output, sample_rate, bits_per_sample=16)
print(torchaudio.info("./VOX_CNN_output.wav"))

finish_time = time.time()
