#!/usr/bin/env python
"""
main.py: This is the main script to be run to either train or make inference.
example usage is as below:
python main.py train --SEED 2 --batch_size 64  --epochs 10 --lr 0.01 \
                     --dropout 0.05 --l1_weight 0.00002  \
                     --l2_weight_decay 0.000125 \
                     --L1 True --L2 False  \
                     --data data  \
                     --best_model_path saved_models \
                     --prefix data
python main.py test  --batch_size 64  --data data  \
                     --best_model_path saved_models \
                     --best_model 'CIFAR10_model_epoch-39_L1-1_L2-0_val_acc-81.83.h5' \
                     --prefix data
"""
from __future__ import print_function

import os
import sys
import warnings

import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary

import cfg
from models import resnet18, custom_resnet, EVA8_session9_assignment_model, vit
from utils import preprocess
from utils import preprocess_albumentations
from utils import test
from utils import train
from utils import misc
from utils import lr_finder

import sys

sys.path.append('./')


def main_session_7_resnet():
    global args
    SEED = args.SEED
    cuda = args.cuda
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # Print the default and/or user supplied arg values, if any.
    print("\n\tHere are the different args values for this run:")
    for arg in vars(args):
        print("\t" + arg, ":", getattr(args, arg))

    # Create the required folder paths.
    best_model_path = args.best_model_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    data_path = args.data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Get the image size tuple
    img_size = args.img_size

    # Calculate mean & std for Normalization
    mean, std, class_names = preprocess.get_dataset_mean_std()
    mean_tuple = (mean[0], mean[1], mean[2])
    std_tuple = (std[0], std[1], std[2])

    # Dataloader Arguments & Test/Train Dataloaders
    if args.use_albumentations:
        print(
            "Using albumentation lib for image-augmentation & other transforms")
        train_dataset, test_dataset, train_loader, test_loader, \
        lr_find_loader = \
            preprocess_albumentations.preprocess_data_albumentations(mean_tuple,
                                                                     std_tuple,
                                                                     img_size)
    else:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess.preprocess_data(mean_tuple, std_tuple)

    # Data Statistics: It is important to know your data very well. Let's
    # check some statistics around our data and how it actually looks like
    preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
    misc.plot_train_samples(train_loader, args.batch_size)

    # Using L1-regularization here (l1_weight = 0.000025, reusing some older
    # assignment values, but works OK here too)
    L1 = args.L1
    L2 = args.L2

    if L2:
        weight_decay = args.l2_weight_decay
    else:
        weight_decay = 0

    criterion = args.criterion  # nn.NLLLoss()

    optimizer = args.optimizer

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # Get the model loaded with summary
    model = resnet18.ResNet18().to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))

    if args.cmd == 'lr_find':
        # Use LR-Range-Test to find maximum best LR as applicable to OCP
        # (best_lr)
        init_lr = args.init_lr
        init_weight_decay = weight_decay  # Based on L2 True/False
        end_lr = args.end_lr
        lr_range_test_epochs = args.lr_range_test_epochs

        best_lr = lr_finder.find_network_lr(model,
                                            criterion,
                                            device,
                                            lr_find_loader,
                                            init_lr,
                                            end_lr,
                                            num_epochs=lr_range_test_epochs)
        print("best_lr is {}".format(best_lr))
    elif args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        best_lr = args.best_lr

        # Setup optimizer & scheduler parameters for OCP
        CYCLE_MOMENTUM = args.cycle_momentum  # If True, momentum value cycles
        # from base_momentum of 0.85 to max_momentum of 0.95 during OCP cycle
        MOMENTUM = 0.9
        WEIGHT_DECAY = weight_decay
        DIV_FACTOR = args.div_factor  # default 10
        # final_div_factor = div_factor for NO annihilation
        FINAL_DIV_FACTOR = DIV_FACTOR
        EPOCHS = args.epochs
        MAX_LR_EPOCH = EPOCHS // 2
        NUM_OF_BATCHES = len(train_loader)
        PCT_START = MAX_LR_EPOCH / EPOCHS
        # Based on above found maximum LR, initialize LRMAX and LRMIN
        LRMAX = best_lr
        LRMIN = LRMAX / DIV_FACTOR

        # Initialize optimizer and scheduler parameters
        optim_params = {"lr": LRMIN,
                        "momentum": MOMENTUM,
                        "weight_decay": WEIGHT_DECAY}
        scheduler_params = {"max_lr": LRMAX,
                            "steps_per_epoch": NUM_OF_BATCHES,
                            "epochs": EPOCHS,
                            "pct_start": PCT_START,
                            "anneal_strategy": "linear",
                            "div_factor": DIV_FACTOR,
                            "final_div_factor": FINAL_DIV_FACTOR,
                            "cycle_momentum": CYCLE_MOMENTUM}
        optimizer = optim.SGD(model.parameters(), **optim_params)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        **scheduler_params)
        last_best = ''
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            print('LR:', optimizer.param_groups[0]['lr'])
            train.train(model,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                        criterion,
                        scheduler=scheduler,
                        L1=L1)
            model_name = test.test(model, device, test_loader, optimizer, epoch,
                                   criterion)
            last_best = last_best if (model_name == '') else model_name
        misc.plot_momentum_lr()
        misc.plot_acc()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        model_name = args.best_model
        print("Loaded the best model: {} from last training session".format(
            model_name))
        model = misc.load_model(resnet18.ResNet18(), device,
                                model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print(
            "The confusion-matrix and classification-report for this model are:")
        y_pred = misc.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        misc.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1),
                                 y_pred, test_dataset,
                                 title_str='Predicted Vs Actual')
        x_test = torch.from_numpy(x_test)

        misc.show_gradcam_mislabelled(model, device, x_test,
                                      y_test.reshape(-1, 1), y_pred,
                                      test_dataset, mean_tuple,
                                      std_tuple, layer='layer3')


def main_session_8_custom_net():
    global args
    SEED = args.SEED
    cuda = args.cuda
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # Print the default and/or user supplied arg values, if any.
    print("\n\tHere are the different args values for this run:")
    for arg in vars(args):
        print("\t" + arg, ":", getattr(args, arg))

    # Create the required folder paths.
    best_model_path = args.best_model_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    data_path = args.data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Get the image size tuple
    img_size = args.img_size

    # Calculate mean & std for Normalization
    mean, std, class_names = preprocess.get_dataset_mean_std()
    mean_tuple = (mean[0], mean[1], mean[2])
    std_tuple = (std[0], std[1], std[2])

    # Dataloader Arguments & Test/Train Dataloaders
    if args.use_albumentations:
        print(
            "Using albumentation lib for image-augmentation & other transforms")
        train_dataset, test_dataset, train_loader, test_loader, \
        lr_find_loader = \
            preprocess_albumentations.preprocess_data_albumentations(mean_tuple,
                                                                     std_tuple,
                                                                     img_size)
    else:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess.preprocess_data(mean_tuple, std_tuple)

    # Data Statistics: It is important to know your data very well. Let's
    # check some statistics around our data and how it actually looks like
    if args.cmd != 'test':
        preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
        misc.plot_train_samples(train_loader, args.batch_size)

    # Using L1-regularization here (l1_weight = 0.000025, reusing some older
    # assignment values, but works OK here too)
    L1 = args.L1
    L2 = args.L2

    if L2:
        weight_decay = args.l2_weight_decay
    else:
        weight_decay = 0

    criterion = args.criterion  # nn.CrossEntropyLoss()

    optimizer = args.optimizer

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # Get the model loaded with summary(10 classes)
    model = custom_resnet.custom_resnet(10). \
        to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))

    if args.cmd == 'lr_find':
        # Use LR-Range-Test to find maximum best LR as applicable to OCP
        # (best_lr)
        init_lr = args.init_lr
        end_lr = args.end_lr
        lr_range_test_epochs = args.lr_range_test_epochs

        best_lr = lr_finder.find_network_lr(model,
                                            criterion,
                                            device,
                                            lr_find_loader,
                                            init_lr,
                                            end_lr,
                                            num_epochs=lr_range_test_epochs)
        print("best_lr is {}".format(best_lr))
    elif args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        best_lr = args.best_lr

        # Setup optimizer & scheduler parameters for OCP
        CYCLE_MOMENTUM = args.cycle_momentum  # If True, momentum value cycles
        # from base_momentum of 0.85 to max_momentum of 0.95 during OCP cycle
        MOMENTUM = 0.9
        WEIGHT_DECAY = weight_decay
        DIV_FACTOR = args.div_factor  # default 10
        # final_div_factor = div_factor for NO annihilation
        FINAL_DIV_FACTOR = DIV_FACTOR
        EPOCHS = args.epochs  # 24 here
        MAX_LR_EPOCHS = args.max_lr_epochs  # 5 here
        NUM_OF_BATCHES = len(train_loader)
        PCT_START = MAX_LR_EPOCHS / EPOCHS
        # Based on above found maximum LR, initialize LRMAX and LRMIN
        LRMAX = best_lr
        LRMIN = LRMAX / DIV_FACTOR

        # Initialize optimizer and scheduler parameters
        optim_params = {"lr": LRMIN,
                        "momentum": MOMENTUM,
                        "weight_decay": WEIGHT_DECAY}
        scheduler_params = {"max_lr": LRMAX,
                            "steps_per_epoch": NUM_OF_BATCHES,
                            "epochs": EPOCHS,
                            "pct_start": PCT_START,
                            "anneal_strategy": "linear",
                            "div_factor": DIV_FACTOR,
                            "final_div_factor": FINAL_DIV_FACTOR,
                            "cycle_momentum": CYCLE_MOMENTUM}
        optimizer = optim.SGD(model.parameters(), **optim_params)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        **scheduler_params)
        last_best = ''
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            print('LR:', optimizer.param_groups[0]['lr'])
            train.train(model,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                        criterion,
                        scheduler=scheduler,
                        L1=L1)
            model_name = test.test(model, device, test_loader, optimizer, epoch,
                                   criterion)
            last_best = last_best if (model_name == '') else model_name
        misc.plot_momentum_lr()
        misc.plot_acc()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        model_name = args.best_model
        print("Loaded the best model: {} from last training session".format(
            model_name))
        model = misc.load_model(custom_resnet.custom_resnet(10), device,
                                model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print(
            "The confusion-matrix and classification-report for this model are:")
        y_pred = misc.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        misc.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1),
                                 y_pred, test_dataset,
                                 title_str='Predicted Vs Actual')
        x_test = torch.from_numpy(x_test)

        misc.show_gradcam_mislabelled(model, device, x_test,
                                      y_test.reshape(-1, 1), y_pred,
                                      test_dataset, mean_tuple,
                                      std_tuple, layer='layer3')


def main_session_9_ultimus():
    global args
    SEED = args.SEED
    cuda = args.cuda
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # Print the default and/or user supplied arg values, if any.
    print("\n\tHere are the different args values for this run:")
    for arg in vars(args):
        print("\t" + arg, ":", getattr(args, arg))

    # Create the required folder paths.
    best_model_path = args.best_model_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    data_path = args.data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Get the image size tuple
    img_size = args.img_size

    # Calculate mean & std for Normalization
    mean, std, class_names = preprocess.get_dataset_mean_std()
    mean_tuple = (mean[0], mean[1], mean[2])
    std_tuple = (std[0], std[1], std[2])

    # Dataloader Arguments & Test/Train Dataloaders
    if args.use_albumentations:
        print(
            "Using albumentation lib for image-augmentation & other transforms")
        train_dataset, test_dataset, train_loader, test_loader, \
        lr_find_loader = \
            preprocess_albumentations.preprocess_data_albumentations(mean_tuple,
                                                                     std_tuple,
                                                                     img_size)
    else:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess.preprocess_data(mean_tuple, std_tuple)

    # Data Statistics: It is important to know your data very well. Let's
    # check some statistics around our data and how it actually looks like
    if args.cmd != 'test':
        preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
        misc.plot_train_samples(train_loader, args.batch_size)

    # Using L1-regularization here (l1_weight = 0.000025, reusing some older
    # assignment values, but works OK here too)
    L1 = args.L1
    L2 = args.L2

    if L2:
        weight_decay = args.l2_weight_decay
    else:
        weight_decay = 0

    criterion = args.criterion  # nn.NLLLoss()

    optimizer = optim.Adam

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # Get the model loaded with summary(10 classes)
    model = EVA8_session9_assignment_model.EVA8_session9_assignment_model(). \
        to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))

    if args.cmd == 'lr_find':
        # Use LR-Range-Test to find maximum best LR as applicable to OCP
        # (best_lr)
        init_lr = args.init_lr
        end_lr = args.end_lr
        lr_range_test_epochs = args.lr_range_test_epochs

        best_lr = lr_finder.find_network_lr(model,
                                            criterion,
                                            device,
                                            lr_find_loader,
                                            init_lr,
                                            end_lr,
                                            num_epochs=lr_range_test_epochs)
        print("best_lr is {}".format(best_lr))
    elif args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        best_lr = args.best_lr

        WEIGHT_DECAY = weight_decay
        DIV_FACTOR = args.div_factor  # default 10
        # final_div_factor = div_factor for NO annihilation
        FINAL_DIV_FACTOR = DIV_FACTOR
        EPOCHS = args.epochs  # 24 here
        MAX_LR_EPOCHS = args.max_lr_epochs  # 5 here
        NUM_OF_BATCHES = len(train_loader)
        PCT_START = MAX_LR_EPOCHS / EPOCHS
        # Based on above found maximum LR, initialize LRMAX and LRMIN
        LRMAX = best_lr
        LRMIN = LRMAX / DIV_FACTOR

        # Initialize optimizer and scheduler parameters
        optim_params = {"lr": LRMIN,
                        "weight_decay": WEIGHT_DECAY}
        scheduler_params = {"max_lr": LRMAX,
                            "steps_per_epoch": NUM_OF_BATCHES,
                            "epochs": EPOCHS,
                            "pct_start": PCT_START,
                            "anneal_strategy": "linear",
                            "div_factor": DIV_FACTOR,
                            "final_div_factor": FINAL_DIV_FACTOR}
        optimizer = optim.Adam(model.parameters(), **optim_params)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        **scheduler_params)
        last_best = ''
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            print('LR:', optimizer.param_groups[0]['lr'])
            train.train(model,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                        criterion,
                        scheduler=scheduler,
                        L1=L1)
            model_name = test.test(model, device, test_loader, optimizer, epoch,
                                   criterion)
            last_best = last_best if (model_name == '') else model_name
        # misc.plot_momentum_lr()
        misc.plot_acc_loss()
        misc.plot_acc()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        model_name = args.best_model
        print("Loaded the best model: {} from last training session".format(
            model_name))
        model = misc.load_model(
            EVA8_session9_assignment_model.EVA8_session9_assignment_model(),
            device,
            model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print(
            "The confusion-matrix and classification-report for this model are:")
        y_pred = misc.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        misc.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1),
                                 y_pred, test_dataset,
                                 title_str='Predicted Vs Actual')
        # x_test = torch.from_numpy(x_test)
        #
        # misc.show_gradcam_mislabelled(model, device, x_test,
        #                               y_test.reshape(-1, 1), y_pred,
        #                               test_dataset, mean_tuple,
        #                               std_tuple, layer='layer3')


def main_session_10_vit():
    global args
    SEED = args.SEED
    cuda = args.cuda
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # Print the default and/or user supplied arg values, if any.
    print("\n\tHere are the different args values for this run:")
    for arg in vars(args):
        print("\t" + arg, ":", getattr(args, arg))

    # Create the required folder paths.
    best_model_path = args.best_model_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    data_path = args.data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Get the image size tuple
    img_size = args.img_size

    # Calculate mean & std for Normalization
    mean, std, class_names = preprocess.get_dataset_mean_std()
    mean_tuple = (mean[0], mean[1], mean[2])
    std_tuple = (std[0], std[1], std[2])

    # Dataloader Arguments & Test/Train Dataloaders
    if args.use_albumentations:
        print(
            "Using albumentation lib for image-augmentation & other transforms")
        train_dataset, test_dataset, train_loader, test_loader, \
        lr_find_loader = \
            preprocess_albumentations.preprocess_data_albumentations(mean_tuple,
                                                                     std_tuple,
                                                                     img_size)
    else:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess.preprocess_data(mean_tuple, std_tuple)

    # Data Statistics: It is important to know your data very well. Let's
    # check some statistics around our data and how it actually looks like
    if args.cmd != 'test':
        preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
        misc.plot_train_samples(train_loader, args.batch_size)

    # Using L1-regularization here (l1_weight = 0.000025, reusing some older
    # assignment values, but works OK here too)
    L1 = args.L1
    L2 = args.L2

    if L2:
        weight_decay = args.l2_weight_decay
    else:
        weight_decay = 0

    criterion = args.criterion  # nn.NLLLoss()

    optimizer = optim.AdamW

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # Get the model loaded with summary(10 classes)
    model = vit.ViT(image_size=32, patch_size=2,
                    num_classes=10, dim=64,
                    depth=2,
                    heads=4, mlp_dim=64,
                    pool='cls',
                    channels=32,
                    dim_head=16, dropout=0.,
                    emb_dropout=0.).to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))

    if args.cmd == 'lr_find':
        # Use LR-Range-Test to find maximum best LR as applicable to OCP
        # (best_lr)
        init_lr = args.init_lr
        end_lr = args.end_lr
        lr_range_test_epochs = args.lr_range_test_epochs

        best_lr = lr_finder.find_network_lr(model,
                                            criterion,
                                            device,
                                            lr_find_loader,
                                            init_lr,
                                            end_lr,
                                            num_epochs=lr_range_test_epochs)
        print("best_lr is {}".format(best_lr))
    elif args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        best_lr = args.best_lr

        WEIGHT_DECAY = weight_decay
        DIV_FACTOR = args.div_factor  # default 10
        # final_div_factor = div_factor for NO annihilation
        FINAL_DIV_FACTOR = DIV_FACTOR
        EPOCHS = args.epochs  # 24 here
        MAX_LR_EPOCHS = args.max_lr_epochs  # 5 here
        NUM_OF_BATCHES = len(train_loader)
        PCT_START = MAX_LR_EPOCHS / EPOCHS
        # Based on above found maximum LR, initialize LRMAX and LRMIN
        LRMAX = best_lr
        LRMIN = LRMAX / DIV_FACTOR

        # Initialize optimizer and scheduler parameters
        optim_params = {"lr": LRMIN,
                        "weight_decay": WEIGHT_DECAY}
        scheduler_params = {"max_lr": LRMAX,
                            "steps_per_epoch": NUM_OF_BATCHES,
                            "epochs": EPOCHS,
                            "pct_start": PCT_START,
                            "anneal_strategy": "linear",
                            "div_factor": DIV_FACTOR,
                            "final_div_factor": FINAL_DIV_FACTOR}
        optimizer = optim.AdamW(model.parameters(), **optim_params)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        **scheduler_params)
        last_best = ''
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            print('LR:', optimizer.param_groups[0]['lr'])
            train.train(model,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                        criterion,
                        scheduler=scheduler,
                        L1=L1)
            model_name = test.test(model, device, test_loader, optimizer, epoch,
                                   criterion)
            last_best = last_best if (model_name == '') else model_name
        # misc.plot_momentum_lr()
        misc.plot_acc_loss()
        misc.plot_acc()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        model_name = args.best_model
        print("Loaded the best model: {} from last training session".format(
            model_name))
        model = misc.load_model(
            vit.ViT(image_size=32, patch_size=2,
                    num_classes=10, dim=64,
                    depth=2,
                    heads=4, mlp_dim=64,
                    pool='cls',
                    channels=32,
                    dim_head=16, dropout=0.,
                    emb_dropout=0.),
            device,
            model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print(
            "The confusion-matrix and classification-report for this model are:")
        y_pred = misc.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        misc.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1),
                                 y_pred, test_dataset,
                                 title_str='Predicted Vs Actual')
        x_test = torch.from_numpy(x_test)
        save_dir = os.path.join(os.getcwd(), args.best_model_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        misc.show_gradcam_mislabelled(model, device, x_test,
                                      y_test.reshape(-1, 1), y_pred,
                                      test_dataset, mean_tuple,
                                      std_tuple, layer=model.mlp_head[2] # the 1x1 convolutional layer
)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    args = cfg.get_args()
    # args.epochs = 2
    # args.max_lr_epochs = 1
    main_session_10_vit()
    # main_session_9_ultimus()
#     main_session_8_custom_net()
    # main_session_7_resnet()
