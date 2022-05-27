############################################################################################
#
# Ry Currier
# 2022-05-18
# RNN Traing Script
#
############################################################################################

import torch
import torchaudio 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class VA_RNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32):
        super(VA_RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rec= nn.LSTM(input_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def forward(self, x):
        x, self.hidden = self.rec(x, self.hidden)
        return self.lin(x)

    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()
    
    def reset_hidden(self):
        self.hidden = None

    def train(self, 
                data_in, 
                data_tar, 
                val_in,
                val_tar,
                num_epochs, 
                loss_fn, 
                optim, 
                batch_size,
                init_len=200):
       
        for n in range(num_epochs):
            
            shuffle = torch.randperm(data_in.shape[1])
            ep_loss = 0
            for b in range(math.ceil(shuffle.shape[0] / batch_size)):
                batch_in = data_in[:, shuffle[b*batch_size:(b+1)*batch_size], :]
                batch_tar = data_tar[:, shuffle[b*batch_size:(b+1)*batch_size], :]
                
                self(batch_in[0:init_len, :, :])
                self.zero_grad()

                start = init_len
                batch_loss = 0
                for k in range(math.ceil((batch_in.shape[0] - init_len) / 1000)):
                    output = self(batch_in[start:start + 1000, :, :])
                    loss = loss_fn(output, batch_tar[start:start + 1000, :, :])
                    loss.backward()
                    optim.step()

                    self.detach_hidden()
                    self.zero_grad()

                    start += 1000
                    batch_loss += loss

                ep_loss += batch_loss / (k+1)
                self.reset_hidden()
            print (f"Epoch {n+1} Loss: {ep_loss / (b+1)}")
            output, val_loss = self.process_samples(val_in, val_tar, loss_fn, 100000)
            print(f"Validation Loss: {val_loss}")

    def process_samples(self, in_data, out_data, loss_fn, length):
        with torch.no_grad():
            output = torch.empty_like(out_data)
            for l in range(int(output.size()[0] / length)):
                output[l*length:(l+1)*length] = self(in_data[l*length:(l+1)*length])
                self.detach_hidden
            if not (output.size()[0] / length).is_integer():
                output[(l+1)*length:-1] = self(in_data[(l+1)*length:-1])
            self.reset_hidden
            loss = loss_fn(output, out_data)
        return output, loss
                    
    def save_model(self, file_name, dir=''):
        pass

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

num_samples = num_samples_dict["train"]
frame_len = math.floor(sample_rate / 2)
num_frames = math.floor(num_samples / frame_len)

train_input = torch.empty((frame_len, num_frames, 1))
train_target = torch.empty((frame_len, num_frames, 1))
val_input = torch.empty((num_samples_dict["val"], 1, 1))
val_target = torch.empty((num_samples_dict["val"], 1, 1))
test_input = torch.empty((num_samples_dict["test"], 1, 1))
test_target = torch.empty((num_samples_dict["test"], 1, 1))

for i in range(num_frames):
    train_input[:, i, :] = in_dict["train"].reshape((num_samples, 1))[i*frame_len:(i+1)*frame_len, :]
    train_target[:, i, :] = out_dict["train"].reshape((num_samples, 1))[i*frame_len:(i+1)*frame_len, :]

val_input[:, 0, :] = in_dict["val"].reshape((num_samples_dict["val"], 1))[:, :]
val_target[:, 0, :] = out_dict["val"].reshape((num_samples_dict["val"], 1))[:, :]
test_input[:, 0, :] = in_dict["test"].reshape((num_samples_dict["test"], 1))[:, :]
test_target[:, 0, :] = out_dict["test"].reshape((num_samples_dict["test"], 1))[:, :]

NUM_EPOCHS = 2
BATCH_SIZE = 50

num_batches_train = math.floor(train_input.shape[1] / BATCH_SIZE)
r_batches_train = train_input.shape[1] - num_batches_train * BATCH_SIZE
train_input = F.pad(train_input, (0, 0, 0, BATCH_SIZE - r_batches_train))
train_target = F.pad(train_target, (0, 0, 0, BATCH_SIZE - r_batches_train))

num_batches_val = math.floor(val_input.shape[1] / BATCH_SIZE)
r_batches_val = val_input.shape[1] - num_batches_val * BATCH_SIZE
val_input = F.pad(val_input, (0, 0, 0, BATCH_SIZE - r_batches_val))
val_target = F.pad(val_target, (0, 0, 0, BATCH_SIZE - r_batches_val))

num_batches_test = math.floor(test_input.shape[1] / BATCH_SIZE)
r_batches_test = test_input.shape[1] - num_batches_test * BATCH_SIZE
test_input = F.pad(test_input, (0, 0, 0, BATCH_SIZE - r_batches_test))
test_target = F.pad(test_target, (0, 0, 0, BATCH_SIZE - r_batches_test))

print(train_input.shape)
print(train_target.shape)
print(val_input.shape)
print(val_target.shape)
print(test_input.shape)
print(test_target.shape)

model = VA_RNN() 
loss_fn = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.08)

model.train(train_input, train_target, val_input, val_target,
             NUM_EPOCHS, loss_fn, optimizer, BATCH_SIZE)

output, loss = model.process_samples(test_input, test_target, loss_fn, 100000)
output = output.reshape((1, 50 * num_samples_dict["test"]))
print(output.shape)
print(f"Test Loss: {loss}")
torchaudio.save("./VOX_RNN_output.wav", output, sample_rate, bits_per_sample=16)
print(torchaudio.info("./VOX_RNN_output.wav"))