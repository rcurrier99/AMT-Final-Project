############################################################################################
#
# Ry Currier
# 2022-05-18
# CNN for Virtual Analog Modeling Traing Script
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
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class VA_CNN(pl.LightningModule):
    ''' 
    CNN Class for Virtual Analog Modeling

    Built from WaveNet blocks, themselves composed of individual layers
    employed dilated convolutions
    '''
    def __init__(self, nparams, sr, device, config, channels=8, blocks=2, 
                layers=9, dilation_growth=2, kernel_size=3):
        super(VA_CNN, self).__init__()
        self.sample_rate = sr
        self.best_val_loss = 1000
        self.loss_fn = CustomLoss()
        self.nparams = nparams
        self.unit = device
        self.config = config
        self.channels = channels
        self.layers = layers
        self.dilation_growth = dilation_growth
        self.kernel_size = kernel_size
        self.blocks = nn.ModuleList()
        for b in range(blocks):
            self.blocks.append(VA_CNN_Block(
                1 if b == 0 else channels,
                channels,
                dilation_growth,
                kernel_size,
                layers,
                self.unit))
        self.blocks.append(nn.Conv1d(channels*layers*blocks, 1, 1, 1, 0))
        
        # Parameter Embedding
        self.embed = nn.Sequential(
            nn.Linear(self.nparams, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU())

    def forward(self, x, p):
        if self.unit != "Phase":
            cond = self.embed(p.float()).type_as(p)
        else:
            cond = p.type_as(p)
        #x = x.permute(1, 2, 0)
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]]).type_as(x)
        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x, cond)
            z[:, n*self.channels*self.layers:(n+1)*self.channels*self.layers, :] = zn
        return self.blocks[-1](z) #.permute(2, 0, 1)

    @torch.jit.unused 
    def training_step(self, batch, batch_idx):
        input, target, params = batch
        pred = self(input, params)

        pred, target = self.pre_emph(pred, target)
        loss = self.loss_fn(pred, target)
        self.log('train_loss', loss, on_step=True, 
                    on_epoch=True, prog_bar=True, 
                    logger=True, sync_dist=True)
        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target, params = batch
        pred = self(input, params)

        pred, target = self.pre_emph(pred, target)
        loss = self.loss_fn(pred, target)
        if loss <= self.best_val_loss:
            self.save_model(f"VA_CNN_{self.unit}_{self.layers}_{self.channels}_{loss}.pth")     # save best val loss network params
            self.best_val_loss = loss
        self.log("val_loss", loss, sync_dist=True)

        outputs = {
            "input" : input.cpu().numpy(),
            "target" : target.cpu().numpy(),
            "pred" : pred.cpu().numpy(),
            "params" : params.cpu().numpy()}
        return outputs

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        input, target, params = batch
        output = self(input, params)

        audio = output.reshape((1, output.shape[2])).cpu()
        file = f"./Data/Output_Files/{self.unit}_CNN_{self.config}_{batch_idx+1}.wav"           # save test output files
        torchaudio.save(file, audio, self.sample_rate, bits_per_sample=16)

        output, target = self.pre_emph(output, target)
        loss = self.loss_fn(output, target)
        self.log("test_loss", loss, sync_dist=True)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            patience=10, verbose=True)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : lr_scheduler,
            'monitor' : 'val_loss'
        }
    
    def process_samples(self, data_in):
        x = data_in.reshape([1, 1, data_in.shape[1]])
        p = torch.tensor([2, 5, 0, 0, 0]).float()
        p = p.reshape([1, p.shape[0]])
        #p = p.expand(-1, -1, x.shape[2])
        out = self(x, p)
        return out.view(1, out.shape[2])

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def pre_emph(self, pred, target):
        filter = nn.Conv1d(1, 1, 2, padding='same', bias=False)
        filter.weight.data = torch.tensor([-0.85, 1], requires_grad=False).reshape(1,1,2).type_as(pred)
        pred = filter(pred).type_as(pred)
        target = filter(target).type_as(target)
        return pred, target

class VA_CNN_Block(nn.Module):
    '''
    WaveNet Block Class

    Built from layers employing dilated convolutions
    '''
    def __init__(self, chan_input, chan_output, dilation_growth, kernel_size, layers, device):
        super(VA_CNN_Block, self).__init__()
        self.channels = chan_output
        dilations = [dilation_growth ** lay for lay in range(layers)]
        self.layers = nn.ModuleList()

        for dil in dilations:
            self.layers.append(VA_CNN_Layer(chan_input, chan_output, dil, kernel_size, device))
            chan_input = chan_output

    def forward(self, x, p):
        z = torch.empty([x.shape[0], len(self.layers)*self.channels, x.shape[2]])
        for n, layer in enumerate(self.layers):
            x, zn = layer(x, p)
            z[:, n*self.channels:(n+1)*self.channels, :] = zn
        return x, z 

class VA_CNN_Layer(nn.Module):
    '''
    WaveNet Layer Class Using Dilated Convolutions
    '''
    def __init__(self, chan_input, chan_output, dilation, kernel_size, device):
        super(VA_CNN_Layer, self).__init__()
        self.channels = chan_output
        self.unit = device
        self.conv= nn.Conv1d(
            in_channels=chan_input,
            out_channels=chan_output * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation)
        self.mix = nn.Conv1d(
            in_channels=chan_output,
            out_channels=chan_output,
            kernel_size=1,
            stride=1,
            padding=0)
        self.lin = nn.Linear(32, chan_output * 2)
        self.parconv = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=chan_output*2, 
            kernel_size=1,
            padding=dilation,
            dilation=dilation)
        #self.film = FiLM(32, chan_output, self.unit, dilation)

    def forward(self, x, p):
        residual = x
        y = self.conv(x)
        #z = torch.tanh(y[:, 0:self.channels, :]) * torch.sigmoid(y[:, self.channels:, :])
        #z = self.film(z, p)
        if self.unit != "Phase":
           p = self.lin(p.float())
           p = p.reshape((p.shape[0], p.shape[1], 1))
           p = p.expand(-1, -1, y.shape[2])
           z = torch.tanh(y[:, 0:self.channels, :] + p[:, 0:self.channels, :]) \
                * torch.sigmoid(y[:, self.channels:, :] + p[:, self.channels:, :])
           #z = self.film(z, p)
        else:
            p = self.parconv(p)
            z = torch.tanh(y[:, 0:self.channels, :] + p[:, 0:self.channels, :]) \
                * torch.sigmoid(y[:, self.channels:, :] + p[:, self.channels:, :])
        z = torch.cat(
            (torch.zeros(
                    residual.shape[0],
                    self.channels,
                    residual.shape[2] - z.shape[2]).type_as(z), z), dim=2)
        x = self.mix(z) + residual
        return x, z

class FiLM(nn.Module):
    ''' Class for CNN FiLM Conditioning '''
    def __init__(self, n_params, n_channels, unit, dil):
        super(FiLM, self).__init__()
        self.unit = unit
        if unit != "Phase":
            self.lay = nn.Linear(n_params, 2*n_channels)
        else:
            self.lay = nn.ConvTranspose1d(1, 2*n_channels, 1, padding=dil, dilation=dil)

    def forward(self, x, p):
        p = self.lay(p.float())
        if self.unit != "Phase":
            g, b = torch.chunk(p, 2, dim=-1)
            g = g.reshape((g.shape[0], g.shape[1], 1))
            b = b.reshape((b.shape[0], b.shape[1], 1))
            g = g.expand(-1, -1, x.shape[2])
            b = b.expand(-1, -1, x.shape[2])
        else:
            g, b = torch.chunk(p, 2, dim=1)

        s1 = x.shape
        s2 = g.shape
        s3 = b.shape

        return g*x + b


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

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output, target):
        return (target - output).sum(-1) / target.shape[-1]

class SCLoss(nn.Module):
    def __init__(self):
        super(SCLoss, self).__init__()

    def forward(self, output, target):
        outputhat = torch.abs(torch.stft(output))
        targethat = torch.abs(torch.stft(target))
        num = torch.norm(outputhat - targethat, p="fro")
        denom = torch.norm(outputhat, p="fro")
        return num / denom

class SMLoss(nn.Module):
    def __init__(self):
        super(SMLoss, self).__init__()

    def forward(self, output, target):
        outputhat = torch.log(torch.abs(torch.stft(output)))
        targethat = torch.log(torch.abs(torch.stft(target)))
        N = target.shape[-1]
        return torch.norm(outputhat - targethat, p=1) / N

class STFTLoss(nn.Module):
    def __init__(self):
        super(STFTLoss, self).__init__()
        self.SCLOSS = SCLoss()
        self.SMLOSS = SMLoss()

    def forward(self, output, target):
        scloss = self.SCLOSS(output, target)
        smloss = self.SMLOSS(output, target)
        return scloss + smloss

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.ESRLOSS = ESRLoss()
        self.DCLOSS = DCLoss()
        self.STFTLOSS = STFTLoss()

    def forward(self, output, target):
        esrloss = self.ESRLOSS(output, target)
        dcloss = self.DCLOSS(output, target)
        return 0.75 * esrloss + 0.25 * dcloss

class VAMLDataSet(Dataset):
    ''' 
    Data Set Class
    To be used w/ dataloader
    '''
    def __init__(self, device, subset, input_dir, target_dir, annotations, config=None):
        self.device = device
        self.input_dir = input_dir
        self.target_dir = target_dir
        df = pd.read_csv(annotations)
        self.subset = subset
        if config == None:
            self.annotations = df[(df['device'] == device) & (df['subset'] == subset)]
        else:
            self.annotations = df[(df['device'] == device) & (df['subset'] == subset) & (df['config'] == config)]

        self.sample_rate = 44100
        self.num_samples = int(self.sample_rate * 1.5)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        input_path = self._get_input_path(index)
        target_path = self._get_target_path(index)
        if self.device != "Phase":
            params = torch.tensor(self.annotations.iloc[index, 4:9])
        else: 
            t = np.linspace(0 + 1.5 * index, 1.5 * (index + 1), self.num_samples)
            b = self.annotations.iloc[index, 5:9].to_numpy()
            params = torch.tensor(b[2]*np.abs( np.sin( b[0]/2*t + b[1]/2 ) ) + b[3]).float()
            params = params.reshape(1, params.shape[0])
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

if __name__ == '__main__':

    WKRS = 0                            # Keep at 0 or else danger
    DEV = False                         # True for debugging on CPU, False for training on GPU
    GPUS = 0 if DEV else 1
    EPOCHS = 250                        # Max training epochs
    DEVICE = "Vox"                      # Device to be modeled ("Vox", "Comp", or "Phase")
    CONFIG = None                       # Parameter configuration(s) to be trained on (None for all configs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load Data
    input_dir = "./Data/Input_Files"
    target_dir = "./Data/Target_Files"
    annotations = "./Data/VAML_Annotation.csv"

    pl.seed_everything(42)

    vox_train_dataset = VAMLDataSet(DEVICE, "Training", 
                                        input_dir, target_dir, annotations, config=CONFIG)
    vox_train_dataloader = DataLoader(vox_train_dataset, 
                                        num_workers=WKRS,
                                        shuffle = True, 
                                        batch_size=32)

    vox_val_dataset = VAMLDataSet(DEVICE, "Validation", 
                                        input_dir, target_dir, annotations, config=CONFIG)
    vox_val_dataloader = DataLoader(vox_val_dataset, 
                                        num_workers = WKRS, 
                                        shuffle = False, 
                                        batch_size=8)

    train_sample_rate = vox_train_dataset.sample_rate
    val_sample_rate = vox_val_dataset.sample_rate
    if train_sample_rate != val_sample_rate:
        ValueError("training and validation data have different sample rates")
    sample_rate = train_sample_rate

    # Train Model
    trainer = pl.Trainer(gpus=GPUS, max_epochs=EPOCHS, 
                            log_every_n_steps=1, fast_dev_run=DEV)
    model = VA_CNN(sr=sample_rate, layers=9, channels=8, nparams=5, device=DEVICE, config=CONFIG)
    #torchsummary.summary(vox_model)
    trainer.fit(model, vox_train_dataloader, vox_val_dataloader)

    # Test Model
    vox_test_dataset = VAMLDataSet(DEVICE, "Testing", 
                                        input_dir, target_dir, annotations, config=CONFIG)
    vox_test_dataloader = DataLoader(vox_test_dataset, 
                                        num_workers=WKRS, 
                                        shuffle = False, 
                                        batch_size=1)
    if not DEV:
        trainer.test(dataloaders=vox_test_dataloader)
