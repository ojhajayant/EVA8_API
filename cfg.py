#!/usr/bin/env python
"""
cfg.py: This contains the hooks, for providing different default or
user-supplied parameters. And also the global variables used across different
packages.
"""
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.append('./')


# IPYNB_ENV = True  # By default ipynb notebook env
# # The AGG backend(for matplotlib) is for writing to "file", not for rendering
# # in a "window".
# if not IPYNB_ENV:
#     plt.switch_backend('agg')
def get_args():
    parser = argparse.ArgumentParser(
        description='Training and Validation on CIFAR10 Dataset')
    parser.add_argument('--cmd', default='train',
                        choices=['train', 'test', 'lr_find'])
    parser.add_argument('--IPYNB_ENV', default=True, type=bool,
                        help='Is this ipynb environment?')
    parser.add_argument('--use_albumentations', default=True, type=bool,
                        help='Use Albumentations based img-aug methods?')
    parser.add_argument('--SEED', '-S', default=1, type=int, help='Random Seed')
    parser.add_argument('--dataset', '-D', default='CIFAR10', type=str,
                        help='Dataset--CIFAR10, MNIST, or...')
    parser.add_argument('--img_size', '-I', default=(32, 32), type=tuple,
                        help='Image Size')
    parser.add_argument('--batch_size', '-b', default=512, type=int,
                        help='batch size')
    parser.add_argument('--epochs', '-e', default=24, type=int,
                        help='training epochs')
    # Below (lr=0.01) was the default for the custom model architecture used for S7
    # parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    # Below (lr=0.0006) was the default for the ResNet18 used for S8 
    parser.add_argument('--criterion', default=nn.CrossEntropyLoss(),
                        type=nn.modules.loss._Loss,
                        help='The loss function to be used during training')
    parser.add_argument('--init_lr', default=1e-4, type=float,
                        help='lr lower range value used for the LR-range-test')
    parser.add_argument('--end_lr', default=0.1, type=float,
                        help='lr upper range value used for the LR-range-test')
    parser.add_argument('--max_lr_epochs', '-M', default=5, type=int,
                        help='at what epoch Max LR should reach?')
    parser.add_argument('--lr_range_test_epochs', '-E', default=150, type=int,
                        help='epoch value used for the LR-range-test')
    parser.add_argument('--best_lr', default=0.03, type=float,
                        help='best_lr obtained from the LR-range-test')
    parser.add_argument('--cycle_momentum', default=True, type=bool,
                        help='Make cyclic changes to momentum value during OCP?')
    parser.add_argument('--div_factor', '-f', default=10, type=int,
                        help='OCP div factor')
    parser.add_argument('--optimizer', default=optim.Adam, type=type(optim.Adam),
                        help='The optimizer to be used during training')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                        help='use gpu or not')
    parser.add_argument('--dropout', '-d', default=0.08, type=float,
                        help='dropout percentage for all layers')
    parser.add_argument('--l1_weight', default=0.000025, type=float,
                        help='L1-penalty value')
    parser.add_argument('--l2_weight_decay', default=0.0002125, type=float,
                        help='L2-penalty/weight_decay value')
    parser.add_argument('--L1', default=True, type=bool,
                        help='L1-penalty to be used or not?')
    parser.add_argument('--L2', default=False, type=bool,
                        help='L2-penalty/weight_decay to be used or not?')
    parser.add_argument('--data', '-s', default='./data/',
                        help='path to save train/test data')
    parser.add_argument('--best_model_path', default='./saved_models/',
                        help='best model saved path')
    parser.add_argument('--prefix', '-p', default='data', type=str,
                        help='folder prefix')
    parser.add_argument('--best_model', '-m',
                        default=' ',
                        type=str, help='name of best-accuracy model saved')
    args = parser.parse_args()
    return args


# Following are the set of global variable being used across.
current_best_acc = 0
last_best_acc = 0
train_losses = []
test_losses = []
train_acc = []
test_acc = []
train_acc_calc = []
lr_range_test_acc = []
lr_range_test_lr = []
momentum_values = []
learning_rate_values = []
