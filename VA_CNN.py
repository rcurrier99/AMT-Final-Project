
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import random
import numpy as np
import matplotlib.pyplot as plt

class VA_CNN(nn.Module()):
    def __init__(self, chan_input, chan_output, dilation, kernel_size):
        super(VA_CNN, self).__init__()
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
            * torch.sigmoid(y[:, self.channels, :])
        z = torch.cat((torch.zeros(residual.shape[0],
                                    self.channels,
                                    residual.shape[2] - z.shape[2]), z), dim=2)
        x = self.mix(z) + residual
        return x, z

    def save_model(self, file_name, dir=''):
        pass

def train(data_input, data_output):
    model = VA_CNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08, weight_decay=1e-4)

    with torch.no_grad():
        output = model(data_input)
        loss = criterion(output, data_output)

    return output
