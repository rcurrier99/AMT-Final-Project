############################################################################################
#
# Ry Currier
# 2022-05-18
# RNN for Virtual Analog Modeling Traing Script
#
############################################################################################

import torch
import torchaudio 
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class VA_RNN(pl.LightningModule):
    def __init__(self, input_size=2, output_size=1, hidden_size=32):
        super(VA_RNN, self).__init__()
        self.best_val_loss = 1000
        self.input_size = input_size
        self.output_size = output_size
        self.best_val_loss = 1000000
        self.rec= nn.LSTM(input_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)
        self.loss_fn = CustomLoss()

    def forward(self, x, p):
        x = x.permute(2, 0, 1)                      # --> (seq, batch, channel)
        p = p.reshape(p.shape[1], p.shape[0], 1)    # ...
        s = x.shape[0]                              # num samples
        c = p.shape[0]                              # num parameters
        r = int(s/c)                                # num repeats
        p = p.repeat(r, 1, 1)                       # match time series shape
        x = torch.cat((x, p), dim=-1)               # append on channel dim
        out, _ = self.rec(x)
        out = torch.tanh(self.lin(out))
        return out.permute(1, 2, 0)                 # --> (batch, channel, seq)

    @torch.jit.unused
    def training_step(self, batch, batch_idx):
        input, target, params = batch
        pred = self(input, params)
        loss = self.loss_fn(pred, target)
        self.log('train_loss', loss, on_step=True, 
                    on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target, params = batch
        pred = self(input, params)

        loss = self.loss_fn(pred, target)
        if loss <= self.best_val_loss:
            self.save_model("VA_RNN.pth")
            self.best_val_loss = loss
        self.log("val_loss", loss)

        outputs = {
            "input" : input.cpu().numpy(),
            "target" : target.cpu().numpy(),
            "pred" : pred.cpu().numpy(),
            "params" : params.cpu().numpy()}
        return outputs

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.08)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            patience=10, verbose=True)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : lr_scheduler,
            'monitor' : 'val_loss'
        }

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
                    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

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
vox_train_dataloader = DataLoader(vox_train_dataset, shuffle = True, batch_size=32)

vox_val_dataset = VAMLDataSet("Vox", "Validation", input_dir, target_dir, annotations)
vox_val_dataloader = DataLoader(vox_val_dataset, shuffle = True, batch_size=8)

train_sample_rate = vox_train_dataset.sample_rate
val_sample_rate = vox_val_dataset.sample_rate
if train_sample_rate != val_sample_rate:
    ValueError("training and validation data have different sample rates")
sample_rate = train_sample_rate

# Train Model
vox_trainer = pl.Trainer(max_epochs=60, log_every_n_steps=1)
vox_model = VA_RNN()
#torchsummary.summary(vox_model)
vox_trainer.fit(vox_model, vox_train_dataloader, vox_val_dataloader)

# Test Model
vox_test_dataset = VAMLDataSet("Vox", "Testing", input_dir, target_dir, annotations)
vox_test_dataloader = DataLoader(vox_test_dataset, shuffle = True, batch_size=1)
loss_fn = CustomLoss()

for batch_idx, batch in enumerate(vox_test_dataloader):
    input, target, params = batch
    print(input.shape)
    print(target.shape)
    print(params.shape)
    with torch.no_grad():
        output = vox_model(input, params)
        print(output.shape)

    audio = output.reshape((1, output.shape[2]))
    torchaudio.save(f"./Data/Output_Files/VOX_RNN_{batch_idx+1}.wav", 
                        audio, sample_rate, bits_per_sample=16)
    loss = loss_fn(output, target)
    print(f"Batch: {batch_idx + 1} Test Loss: {loss}")

############################################################################################
# TO DO:
#
# Get that ringing to disappear/ improve convergence
# Implement some frequency domain loss fncns, that might do the trick