#!/usr/bin/env python
"""
EVA8_session4_assignment_model.py: CNN class definitions for EVA8 assignment4
"""
from __future__ import print_function
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./')
dropout_value = 0.029  # value worked quite good for the MNIST model used
# during session 4 & 5.


class EVA8_session4_assignment_model(nn.Module):
    """
    the model definition used in session 4 & 5 to be trained on MNIST dataset.
    """

    def __init__(self, normalization='batch'):
        super(EVA8_session4_assignment_model, self).__init__()
        self.normalization = normalization

        # Input Block
        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        input_block_conv_layer_0 = [nn.Conv2d(in_channels=1, out_channels=10,
                                              kernel_size=(3, 3),
                                              padding=0, bias=False)]
        if self.normalization == 'batch':
            input_block_conv_layer_0.append(nn.BatchNorm2d(10))
        elif self.normalization == 'layer':
            input_block_conv_layer_0.append(nn.LayerNorm([10, 26, 26]))
        elif self.normalization == 'group':
            input_block_conv_layer_0.append(nn.GroupNorm(5, 10))

        input_block_conv_layer_0.append(nn.Dropout(dropout_value))
        input_block_conv_layer_0.append(nn.ReLU())
        self.convblock1 = nn.Sequential(*input_block_conv_layer_0)  # input_size
        # = 28 output_size = 26 receptive_field = 3

        # CONVOLUTION BLOCK 1
        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        block_1_conv_layer_0 = [nn.Conv2d(in_channels=10, out_channels=10,
                                          kernel_size=(3, 3),
                                          padding=0, bias=False)]
        if self.normalization == 'batch':
            block_1_conv_layer_0.append(nn.BatchNorm2d(10))
        elif self.normalization == 'layer':
            block_1_conv_layer_0.append(nn.LayerNorm([10, 24, 24]))
        elif self.normalization == 'group':
            block_1_conv_layer_0.append(nn.GroupNorm(5, 10))

        block_1_conv_layer_0.append(nn.Dropout(dropout_value))
        block_1_conv_layer_0.append(nn.ReLU())
        self.convblock2 = nn.Sequential(*block_1_conv_layer_0)  # input_size =
        # 26 output_size = 24 receptive_field = 5

        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        block_1_conv_layer_1 = [nn.Conv2d(in_channels=10, out_channels=15,
                                          kernel_size=(3, 3),
                                          padding=0, bias=False)]
        if self.normalization == 'batch':
            block_1_conv_layer_1.append(nn.BatchNorm2d(15))
        elif self.normalization == 'layer':
            block_1_conv_layer_1.append(nn.LayerNorm([15, 22, 22]))
        elif self.normalization == 'group':
            block_1_conv_layer_1.append(nn.GroupNorm(5, 15))

        block_1_conv_layer_1.append(nn.Dropout(dropout_value))
        block_1_conv_layer_1.append(nn.ReLU())
        self.convblock3 = nn.Sequential(*block_1_conv_layer_1)  # input_size =
        # 24 output_size = 22 receptive_field = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # input_size = 22 output_size = 11
        # receptive_field = 8

        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        transition_layer_conv_layer_0 = [nn.Conv2d(in_channels=15,
                                                   out_channels=10,
                                                   kernel_size=(1, 1),
                                                   padding=0, bias=False)]
        if self.normalization == 'batch':
            transition_layer_conv_layer_0.append(nn.BatchNorm2d(10))
        elif self.normalization == 'layer':
            transition_layer_conv_layer_0.append(nn.LayerNorm([10, 11, 11]))
        elif self.normalization == 'group':
            transition_layer_conv_layer_0.append(nn.GroupNorm(5, 10))

        transition_layer_conv_layer_0.append(nn.Dropout(dropout_value))
        transition_layer_conv_layer_0.append(nn.ReLU())
        self.convblock4 = nn.Sequential(*transition_layer_conv_layer_0)  #
        # input_size = 11 output_size = 11 receptive_field = 8

        # CONVOLUTION BLOCK 2
        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        block_2_conv_layer_0 = [nn.Conv2d(in_channels=10, out_channels=10,
                                          kernel_size=(3, 3),
                                          padding=0, bias=False)]
        if self.normalization == 'batch':
            block_2_conv_layer_0.append(nn.BatchNorm2d(10))
        elif self.normalization == 'layer':
            block_2_conv_layer_0.append(nn.LayerNorm([10, 9, 9]))
        elif self.normalization == 'group':
            block_2_conv_layer_0.append(nn.GroupNorm(5, 10))

        block_2_conv_layer_0.append(nn.Dropout(dropout_value))
        block_2_conv_layer_0.append(nn.ReLU())
        self.convblock5 = nn.Sequential(*block_2_conv_layer_0)  # input_size =
        # 11 output_size = 9 receptive_field = 12

        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        block_2_conv_layer_1 = [nn.Conv2d(in_channels=10, out_channels=10,
                                          kernel_size=(3, 3),
                                          padding=0, bias=False)]
        if self.normalization == 'batch':
            block_2_conv_layer_1.append(nn.BatchNorm2d(10))
        elif self.normalization == 'layer':
            block_2_conv_layer_1.append(nn.LayerNorm([10, 7, 7]))
        elif self.normalization == 'group':
            block_2_conv_layer_1.append(nn.GroupNorm(5, 10))

        block_2_conv_layer_1.append(nn.Dropout(dropout_value))
        block_2_conv_layer_1.append(nn.ReLU())
        self.convblock6 = nn.Sequential(*block_2_conv_layer_1)  #
        # input_size = 9 output_size = 7 receptive_field = 16

        # For nn.Sequential construct to work, it needs a "list" of layers,
        # hence start out with the conv2d layer, as the first element of such
        # "list" and keep appending Norm/Dropout/Relu extra layers to it.
        block_2_conv_layer_2 = [nn.Conv2d(in_channels=10, out_channels=32,
                                          kernel_size=(3, 3),
                                          padding=0, bias=False)]
        if self.normalization == 'batch':
            block_2_conv_layer_2.append(nn.BatchNorm2d(32))
        elif self.normalization == 'layer':
            block_2_conv_layer_2.append(nn.LayerNorm([32, 5, 5]))
        elif self.normalization == 'group':
            block_2_conv_layer_2.append(nn.GroupNorm(4, 32))
        block_2_conv_layer_2.append(nn.Dropout(dropout_value))
        block_2_conv_layer_2.append(nn.ReLU())
        self.convblock7 = nn.Sequential(*block_2_conv_layer_2)  # input_size =
        # 7 output_size = 5 receptive_field = 20

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # input_size = 5 output_size = 1 receptive_field = 28

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1),
                      padding=0, bias=False),
            # No BatchNorm/DropOut/ReLU
        )  # input_size = 1 output_size = 1 receptive_field = 28

    def forward(self, x):
        # INPUT BLOCK LAYER
        x = self.convblock1(x)

        # CONVOLUTION BLOCK 1
        x = self.convblock2(x)
        x = self.convblock3(x)

        # TRANSITION BLOCK 1
        x = self.pool1(x)
        x = self.convblock4(x)

        # CONVOLUTION BLOCK 2
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)

        # OUTPUT BLOCK
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
