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

class VA_RNN(pl.LightningModule):
    ''' 
    RNN Class for Virtual Analog Modeling
    '''
    def __init__(self, sr, n_params, device, config, 
                    input_size=1, output_size=1, hidden_size=32):
        super(VA_RNN, self).__init__()
        self.sample_rate = sr
        self.input_size = input_size + n_params
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.unit = device
        self.config = config

        self.best_val_loss = 1000000

        self.rec= nn.LSTM(self.input_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)
        self.embed = nn.ConvTranspose1d(1, 8, 1)
        self.lim = nn.Hardtanh()
        self.loss_fn = CustomLoss()
        #self.enc = nn.Conv1d(1, 8, 3, padding='same')
        #self.dec = nn.ConvTranspose1d(8, 1, 3, padding=1)
        #self.cond = nn.ModuleList([
            #FiLM(n_params, 32),
            #nn.ConvTranspose1d(32, 16, 3, padding=1),
            #FiLM(n_params, 16),
            #nn.ConvTranspose1d(16, 8, 3, padding=1),
            #FiLM(n_params, 8),
            #nn.ConvTranspose1d(8, 4, 3, padding=1),
            #FiLM(n_params, 4),
            #nn.ConvTranspose1d(4, 2, 3, padding=1),
            #FiLM(n_params, 2),
            #nn.ConvTranspose1d(2, 1, 3, padding=1)
        #])

    def forward(self, x, p):
        #x = self.enc(x)

        x = x.permute(2, 0, 1)                      # --> (seq, batch, channel)
        p = p.permute(2, 0, 1)                      # ...
        res = x.type_as(x)                          # save residual

        x = torch.cat((x, p), dim=-1)               # append on channel dim
        out, _ = self.rec(x)
        #out = out.permute(1, 2, 0)
        #for i, lay in enumerate(self.cond):
            #s1 = out.shape
            #s2 = p.shape
            #out = lay(out, p) if (i % 2 == 0) else lay(out)
        #return self.lim(out + res.permute(1, 2, 0))
        out = self.lim(self.lin(out) + res)         # dont explode
        return out.permute(1, 2, 0)                 # --> (batch, channel, seq)

        
        #return self.dec(out)

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
            self.save_model(f"VA_RNN_{self.unit}_{self.hidden_size}_{loss}.pth")        # save best val loss network params
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
        file = f"./Data/Output_Files/{self.unit}_RNN_{self.config}_{batch_idx+1}.wav"   # save test output files
        torchaudio.save(file, audio, self.sample_rate, bits_per_sample=16)

        output, target = self.pre_emph(output, target)
        loss = self.loss_fn(output, target)
        self.log("test_loss", loss, sync_dist=True)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0005)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=10,
            verbose=True)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : lr_scheduler,
            'monitor' : 'val_loss'
        }

    def process_samples(self, in_data):
        x = in_data.reshape([1, 1, in_data.shape[1]])
        res = x
        p = torch.tensor([2, 5, 0, 0, 0]).float()
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

class FiLM(nn.Module):
    ''' RNN FiLM Conditioning Class '''
    def __init__(self, n_params, n_channels):
        super(FiLM, self).__init__()
        self.channels = n_channels
        self.lay = nn.ConvTranspose1d(n_params, 2*n_channels, 1)

    def forward(self, x, p):
        p = self.lay(p.float())
        s4 = p.shape
        g, b = torch.chunk(p, 2, dim=1)
        s2 = g.shape
        s3 = b.shape


        s1 = x.shape

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
        outputhat = torch.abs(
            torch.stft(output.view(-1, output.size(-1)), 1024))
        targethat = torch.abs(
            torch.stft(target.view(-1, target.size(-1)), 1024))
        num = torch.norm(outputhat - targethat, p="fro")
        denom = torch.norm(outputhat, p="fro") + 1e-5
        num.div_(denom)
        return num

class SMLoss(nn.Module):
    def __init__(self):
        super(SMLoss, self).__init__()

    def forward(self, output, target):
        outputhat = torch.log(
            torch.abs(
                torch.stft(
                    output.view(-1, output.size(-1)), 1024)) + 1e-5)
        targethat = torch.log(
            torch.abs(
                torch.stft(
                    target.view(-1, target.size(-1)), 1024)) + 1e-5)
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
        return 0.75*scloss + 0.25*smloss

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.ESRLOSS = ESRLoss()
        self.DCLOSS = DCLoss()
        self.STFTLOSS = STFTLoss()

    def forward(self, output, target):
        esrloss = self.ESRLOSS(output, target)
        dcloss = self.DCLOSS(output, target)
        stftloss = self.STFTLOSS(output, target)
        return 0.75 * esrloss + 0.25 * dcloss # + stftloss)/2

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
        self.tbptt_len = 1225
        self.num_samples = int(self.sample_rate * 1.5)
        self.num_frames = int(np.floor(self.num_samples / self.tbptt_len))

    def __len__(self):
        if self.subset != "Training":
            return len(self.annotations)
        else:
            return self.num_frames * len(self.annotations)
    
    def __getitem__(self, index):
        idx = int(np.floor(index / self.num_frames))
        rem = index - self.num_frames * idx

        index = index if self.subset != "Training" else idx

        input_path = self._get_input_path(index)
        target_path = self._get_target_path(index)

        input, sr1 = torchaudio.load(input_path)
        target, sr2 = torchaudio.load(target_path)

        if sr1 != sr2:
            raise ValueError(f"Input and target files at index {index} have different sample rates")
        if self.sample_rate != None and self.sample_rate != sr1:
            raise ValueError(f"Files at index {index} have a different sample rate from the rest of the data")
        
        if self.device != "Phase":
            params = torch.tensor(self.annotations.iloc[index, 4:9]).float()
            params = params.reshape(params.shape[0], 1)
            params = params.expand(-1, input.shape[1])
        else: 
            t = np.linspace(0 + 1.5 * index, 1.5 * (index + 1), self.num_samples)
            b = self.annotations.iloc[index, 5:9].to_numpy()
            params = torch.tensor((b[2]*np.abs( np.sin( b[0]/2*t + b[1]/2 ) ) + b[3])/b[2]).float()
            params = params.reshape(1, params.shape[0])

        if self.subset == "Training":
            input = input[:, rem * self.tbptt_len:(rem + 1) * self.tbptt_len]
            target = target[:, rem * self.tbptt_len:(rem + 1) * self.tbptt_len]
            params = params[:, rem * self.tbptt_len:(rem + 1) * self.tbptt_len]

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
    DEV = True                          # True for debugging on CPU, False for training on GPU
    GPUS = 0 if DEV else 1
    EPOCHS = 500                        # Max training epochs
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
                                        num_workers=WKRS,
                                        shuffle = False, 
                                        batch_size=8)

    train_sample_rate = vox_train_dataset.sample_rate
    val_sample_rate = vox_val_dataset.sample_rate
    if train_sample_rate != val_sample_rate:
        ValueError("training and validation data have different sample rates")
    sample_rate = train_sample_rate

    # Train Model
    trainer = pl.Trainer(
        gpus=GPUS, 
        max_epochs=EPOCHS, 
        log_every_n_steps=1,
        fast_dev_run=DEV)
    model = VA_RNN(sr=sample_rate, hidden_size=32, n_params=5, device=DEVICE, config=CONFIG)
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
