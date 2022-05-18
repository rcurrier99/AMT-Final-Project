
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import random
import numpy as np
import matplotlib.pyplot as plt

class VA_RNN(nn.Module()):
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

    def save_model(self, file_name, dir=''):
        pass

def train(data_input, data_output):
    model = VA_RNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.08)

    with torch.no_grad():
        output = model(data_input)
        loss = criterion(output, data_output)

    return output
