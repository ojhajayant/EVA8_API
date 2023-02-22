#!/usr/bin/env python
"""
train.py: This contains the model-training code.
"""
from __future__ import print_function

import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy

import cfg
from utils import misc

import cfg

sys.path.append('./')
from cfg import get_args

args = get_args()
file_path = args.data


def train(model, device, train_loader, optimizer, epoch, criterion,
          scheduler=None, L1=False):
    """
    main training code
    """
    global args
    model.train()
    pbar = tqdm(train_loader, position=0, leave=True)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to
        # do back-propagation because PyTorch accumulates the gradients on
        # subsequent backward passes. Because of this, when you start your
        # training loop, ideally you should zero out the gradients so that
        # you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        if L1:
            to_reg = []
            for param in model.parameters():
                to_reg.append(param.view(-1))
            l1 = args.l1_weight * misc.l1_penalty(torch.cat(to_reg))
        else:
            l1 = 0
        # Calculate loss
        # L1 regularization adds an L1 penalty equal to the
        # absolute value of the magnitude of coefficients
        # torch.nn.CrossEntropyLoss:criterion combines
        # nn.LogSoftmax() and nn.NLLLoss() in one single class.
        loss = criterion(y_pred, target) + l1
        cfg.train_losses.append(loss)
        # Backpropagation
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Capture the momentum and learning rate values
        if args.optimizer != optim.Adam:
            if optimizer.param_groups[0]['momentum'] is not None:
                cfg.momentum_values.append(optimizer.param_groups[0]['momentum'])
        cfg.learning_rate_values.append(optimizer.param_groups[0]['lr'])
        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max
        # log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
    cfg.train_acc.append(100 * correct / processed)
    return (100 * correct / processed)


def train_calc_acc(model, device, train_iter, criterion, lr,
                   scheduler=None, L1=False):
    """
    main training code
    """
    global args
    test_model = copy.deepcopy(model)
    optimizer = optim.SGD(test_model.parameters(), lr=lr, momentum=0.9)
    test_model.train()
    correct = 0
    processed = 0
    data, target = next(train_iter)
    # get samples
    data, target = data.to(device), target.to(device)
    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to
    # do back-propagation because PyTorch accumulates the gradients on
    # subsequent backward passes. Because of this, when you start your
    # training loop, ideally you should zero out the gradients so that
    # you do the parameter update correctly.

    # Predict
    y_pred = test_model(data)
    if L1:
        to_reg = []
        for param in test_model.parameters():
            to_reg.append(param.view(-1))
        l1 = args.l1_weight * misc.l1_penalty(torch.cat(to_reg))
    else:
        l1 = 0
    # Calculate loss
    # L1 regularization adds an L1 penalty equal to the
    # absolute value of the magnitude of coefficients
    # torch.nn.CrossEntropyLoss:criterion combines
    # nn.LogSoftmax() and nn.NLLLoss() in one single class.
    loss = criterion(y_pred, target) + l1
    # Backpropagation
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max
    # log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    cfg.train_acc_calc.append(100 * correct / processed)
    return 100 * correct / processed

