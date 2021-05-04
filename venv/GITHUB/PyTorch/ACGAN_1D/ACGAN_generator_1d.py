#reference : https://towardsdatascience.com/understanding-acgans-with-code-pytorch-2de35e05d3e4

import torch
from torch import nn


class generator(nn.Module):

    # generator model
    def __init__(self, in_channels):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(in_channels, 14)

        self.t1 = nn.Sequential(
            nn.Linear(14, 14),
            nn.Tanh()
        )
        self.t2 = nn.Sequential(
            nn.Linear(14, 10),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 7)
        x = self.fc1(x)
        x = x.view(-1, 14)
        x = self.t1(x)
        x = self.t2(x)
        return x  # output of generator