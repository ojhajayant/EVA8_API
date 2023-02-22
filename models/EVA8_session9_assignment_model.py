#!/usr/bin/env python
"""
EVA8_session9_assignment_model.py: CNN class definition for EVA8 assignment 9
"""
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./')
dropout_value = 0.05


# Define the ULTIMUS block
class ULTIMUS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ULTIMUS, self).__init__()
        self.K = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )
        self.Q = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )
        self.V = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )
        self.OUT = nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=in_dim),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

    def forward(self, x):
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        am = F.softmax(torch.matmul(q, k.t()) / (8 ** 0.5),
                       dim=-1)
        z = torch.matmul(am, v)
        out = self.OUT(z)
        out = out + x  # Add skip connection
        return out


# Define the EVA8_session9_assignment_model architecture
class EVA8_session9_assignment_model(nn.Module):
    def __init__(self):
        super(EVA8_session9_assignment_model, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x3, output:32x32x16, RF:3x3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x16, output:32x32x32, RF:5x5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x32, output:32x32x48, RF:7x7

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=32)
        )  # input:32x32x48, output:1x1x48

        self.ultimus1 = ULTIMUS(48, 8)
        self.ultimus2 = ULTIMUS(48, 8)
        self.ultimus3 = ULTIMUS(48, 8)
        self.ultimus4 = ULTIMUS(48, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_features=48, out_features=10),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        x = self.fc(x)
        return x
