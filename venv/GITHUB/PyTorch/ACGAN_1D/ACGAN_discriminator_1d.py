#reference : https://towardsdatascience.com/understanding-acgans-with-code-pytorch-2de35e05d3e4

import torch
from torch import nn


class discriminator(nn.Module):

    def __init__(self, classes=7):
        # we have 10 classes in the CIFAR dataset with 6000 images per class.
        super(discriminator, self).__init__()
        self.c1 = nn.Sequential(
            nn.Linear(10 ,14),
            nn.Tanh()
        )
        self.c2 = nn.Sequential(
            nn.Linear(14 ,14),
            nn.Tanh()
        )

        self.fc_source = nn.Linear(14, 1)
        self.fc_class = nn.Linear(14, classes)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = x.view(-1, 14)
        rf = self.sig(
            self.fc_source(x))  # checks source of the data---i.e.--data generated(fake) or from training set(real)
        c = self.soft(self.fc_class(x))
        return rf, c