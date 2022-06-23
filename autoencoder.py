
############################################################################################
#
# Ry Currier
# 2022-06-06
# Transformer for Virtual Analog Modeling
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
import librosa
import pickle
import numpy as np
import pandas as pd
import pywt
import ptwt

class VA_Transformer(pl.LightningModule):
    def __init__(
            self,
            sr,
            lr,
            n_params,
            channels=1,
            num_heads=3,
            num_encoder_layers=3,
            num_decoder_layers=3,
            ff_expansion=2,
            output_size=1
    ):
        super(VA_Transformer, self).__init__()
        self.lr = lr
        self.sample_rate = sr
        self.loss_fn = CustomLoss()
        self.best_val_loss = 1000000
        self.transformer = nn.Transformer(
            channels, 
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            ff_expansion,
            dropout=0
        )
        self.lin = nn.Linear(pow(ff_expansion, num_encoder_layers), output_size)

    def forward(self, x, t, p):
        x = x.permute(1, 0, 2)                      # --> (channel, batch, seq)
        t = t.permute(1, 0, 2)                      # ...
        p = p.reshape(1, p.shape[0], p.shape[1])    # ...
        p = p.expand(x.shape[0], -1, -1)            # expand p along time dim
        x = torch.cat((x, p), dim=-1)               # append on seq dim
        t = torch.cat((t, p), dim=-1)               # ...
        out = self.transformer(x, t)
        #out = torch.tanh(self.lin(out))
        return out.permute(1, 0, 2)[:, :, 0]        # --> (batch, channel, seq)

    @torch.jit.unused
    def training_step(self, batch, batch_idx):
        input, target, params = batch
        pred = self(input, target, params)
        loss = self.loss_fn(pred, target)
        self.log('train_loss', loss, on_step=True, 
                    on_epoch=True, prog_bar=True, 
                    logger=True, sync_dist=True)
        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target, params = batch
        pred = self(input, target, params)

        loss = self.loss_fn(pred, target)
        if loss <= self.best_val_loss:
            self.save_model("VA_RNN.pth")
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
        output = self(input, target, params)

        audio = output.reshape((1, output.shape[2])).cpu()
        torchaudio.save(f"./Data/Output_Files/VOX_VAE_{batch_idx+1}.wav", 
                            audio, self.sample_rate, bits_per_sample=16)
        loss = self.loss_fn(output, target)
        self.log("test_loss", loss, sync_dist=True)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            patience=10, verbose=True)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : lr_scheduler,
            'monitor' : 'val_loss'
        }

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

if __name__ == '__main__':

    WKRS = 0
    DEV = True
    GPUS = 0
    LEARNING_RATE  = 0.0005
    EPOCHS = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load Data
    input_dir = "./Data/Input_Files"
    target_dir = "./Data/Target_Files"
    annotations = "./Data/VAML_Annotation.csv"

    vox_train_dataset = VAMLDataSet("Vox", "Training", input_dir, target_dir, annotations)
    vox_train_dataloader = DataLoader(vox_train_dataset,
                                        num_workers=WKRS,
                                        shuffle = True, 
                                        batch_size=32)

    vox_val_dataset = VAMLDataSet("Vox", "Validation", input_dir, target_dir, annotations)
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
    vox_trainer = pl.Trainer(gpus=GPUS, max_epochs=EPOCHS, 
                                log_every_n_steps=1, fast_dev_run=DEV)
    vox_model = VA_Transformer(sr=sample_rate, lr=LEARNING_RATE, n_params=5)
    #torchsummary.summary(vox_model)
    vox_trainer.fit(vox_model, vox_train_dataloader, vox_val_dataloader)

    # Test Model
    vox_test_dataset = VAMLDataSet("Vox", "Testing", input_dir, target_dir, annotations)
    vox_test_dataloader = DataLoader(vox_test_dataset, 
                                        num_workers=WKRS,
                                        shuffle = False, 
                                        batch_size=1)
    if not DEV:
        vox_trainer.test(dataloaders=vox_test_dataloader)
