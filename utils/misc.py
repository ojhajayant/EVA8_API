#!/usr/bin/env python
"""
misc.py: This contains miscellaneous-utility code used across.
"""
from __future__ import print_function

import json
import os
import shutil
import sys
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
from torch.autograd import Variable
from torchvision import transforms
from pytorch_grad_cam import GradCAM

import cfg

sys.path.append('./')
from cfg import get_args

args = get_args()
file_path = args.data


# IPYNB_ENV = True  # By default ipynb notebook env


def plot_train_samples(train_loader, batch_size):
    """
    Plot dataset class samples
    """
    global args
    num_classes = len(np.unique(train_loader.dataset.targets))
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'plot_class_samples'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    if not args.IPYNB_ENV:
        print(
            "Saving plot {} class samples to {}".format(num_classes, filepath))
    else:
        print("Here are a few samples BEFORE TRANSFORMS APPLIED:")
    fig = plt.figure(figsize=(8, 3))
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(np.array(train_loader.dataset.targets)[:] == i)[0]
        features_idx = train_loader.dataset.data[idx, ::]
        img_num = np.random.randint(features_idx.shape[0])
        im = features_idx[img_num]
        ax.set_title(train_loader.dataset.classes[i])
        plt.imshow(im)
        plt.savefig('plot2.png')
        from IPython.display import Image
        Image(filename='plot2.png')
        display(plt.gcf())
        if not args.IPYNB_ENV:
            plt.savefig(filepath)
    if args.IPYNB_ENV:
        plt.savefig('plot3.png')
        from IPython.display import Image
        Image(filename='plot3.png')
        display(plt.gcf())
        plt.show()
    print("Here are a few samples AFTER TRANSFORMS APPLIED:")
    batch = next(iter(train_loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=batch_size // 8)
    plt.figure(figsize=(25, 25))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig('plot4.png')
    from IPython.display import Image
    Image(filename='plot4.png')
    display(plt.gcf())
    plt.show()


def l1_penalty(x):
    """
    L1 regularization adds an L1 penalty equal
    to the absolute value of the magnitude of coefficients
    """
    global args

    return torch.abs(x).sum()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the model to the path
    """
    global args
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def model_pred(model, device, y_test, test_dataset, batchsize=100):
    """
    Make inference on the test-data &
    print classification-report
    """
    global args
    start = 0
    stop = batchsize
    model.eval()
    dataldr_args = dict(shuffle=False, batch_size=batchsize, num_workers=4,
                        pin_memory=True) if args.cuda else dict(
        shuffle=False, batch_size=batchsize)
    test_ldr = torch.utils.data.DataLoader(test_dataset, **dataldr_args)
    y_pred = np.zeros((y_test.shape[0], 1))
    with torch.no_grad():
        for data, target in test_ldr:
            batch_nums = np.arange(start, stop)
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred[batch_nums] = output.argmax(dim=1,
                                               keepdim=True).cpu().numpy()
            start += batchsize
            stop += batchsize
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred,
                                target_names=test_dataset.classes))
    return y_pred


def display_mislabelled(model, device, x_test, y_test, y_pred, test_dataset,
                        title_str):
    """
    Plot 3 groups of 10 mislabelled data class-samples.
    """
    global args
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'plot_mislabelled'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    if not args.IPYNB_ENV:
        print("Saving plot for the mislabelled images to {}".format(filepath))
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle(title_str, fontsize=24)
    idx1 = np.where(y_test[:] != y_pred)[0]
    for j in range(3):
        for i in range(len(test_dataset.classes)):
            ax = fig.add_subplot(3, 10, j * 10 + i + 1, xticks=[], yticks=[])
            idx = np.where(y_test[:] == i)[0]
            intsct = np.intersect1d(idx1, idx)
            features_idx = x_test[intsct, ::]
            img_num = np.random.randint(features_idx.shape[0])
            im = features_idx[img_num]
            if args.dataset == 'CIFAR10':
                ax.set_title('Act:{} '.format(
                    test_dataset.classes[int(i)]) + ' Pred:{} '.format(
                    test_dataset.classes[int(y_pred[intsct[img_num]][0])]),
                             fontsize=8)
            elif args.dataset == 'MNIST':
                ax.set_title('Act:{} '.format(i) + ' Pred:{} '.format(
                    int(y_pred[intsct[img_num]][0])), fontsize=8)
            plt.imshow(im)
            plt.savefig('plot5.png')
            from IPython.display import Image
            Image(filename='plot5.png')
            display(plt.gcf())
            if not args.IPYNB_ENV:
                plt.savefig(filepath)
    if args.IPYNB_ENV:
        plt.savefig('plot6.png')
        from IPython.display import Image
        Image(filename='plot6.png')
        display(plt.gcf())
        plt.show()


def load_model(describe_model_nn, device, model_name):
    """
    load the best-accuracy model from the given name
    """
    global args
    save_dir = os.path.join(os.getcwd(), args.best_model_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    model = describe_model_nn  # describe_model_nn is for example: Net1()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model


def save_acc_loss(test_losses, test_acc, test_loss_file_name,
                  test_acc_file_name):
    """
    Save test-accuracies and test-losses during training.
    """
    global args
    import os
    import numpy as np
    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), file_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath_test_loss = os.path.join(save_dir, test_loss_file_name)
    filepath_test_acc = os.path.join(save_dir, test_acc_file_name)
    np.save(filepath_test_loss, test_losses)
    np.save(filepath_test_acc, test_acc)


def load_acc_loss(test_loss_file_name, test_acc_file_name):
    """
    Load the accuracy and loss data from files.
    """
    global args
    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), file_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath_test_loss = os.path.join(save_dir, test_loss_file_name)
    filepath_test_acc = os.path.join(save_dir, test_acc_file_name)
    loaded_test_losses = np.load(filepath_test_loss).tolist()
    loaded_test_acc = np.load(filepath_test_acc).tolist()
    return loaded_test_losses, loaded_test_acc


def plot_acc():
    """
    Plot both accuracy and loss plots.
    """
    _ = plt.plot(cfg.train_acc)
    _ = plt.plot(cfg.test_acc)
    _ = plt.title('model accuracy')
    _ = plt.ylabel('accuracy')
    _ = plt.xlabel('epoch')
    _ = plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plot7.png')
    from IPython.display import Image
    Image(filename='plot7.png')
    display(plt.gcf())
    _ = plt.show()


def plot_momentum_lr():
    """
    Plot both momentum and learning rate plots.
    """
    _ = plt.plot(cfg.momentum_values)
    _ = plt.plot(cfg.learning_rate_values)
    _ = plt.title('Momentum & LR')
    _ = plt.ylabel('Value')
    _ = plt.xlabel('Batch')
    _ = plt.legend(['Momentum', 'Learning Rate'], loc='upper left')
    plt.savefig('plot8.png')
    from IPython.display import Image
    Image(filename='plot8.png')
    display(plt.gcf())
    _ = plt.show()


def plot_acc_loss():
    """
    Plot both accuracy and loss plots.
    """
    global args
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'plot_acc_loss'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(cfg.train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(cfg.train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(cfg.test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(cfg.test_acc)
    axs[1, 1].set_title("Test Accuracy")
    if not args.IPYNB_ENV:
        fig.savefig(filepath)
    else:
        plt.savefig('plot9.png')
        from IPython.display import Image
        Image(filename='plot9.png')
        display(plt.gcf())
        fig.show()


def write(dic, path):
    global args
    with open(path, 'w+') as f:
        # write params to txt file
        f.write(json.dumps(dic))


def superimpose(original_img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(original_img)
    cam = cam / np.max(cam)
    cam = np.clip(cam, 0, 1)
    return np.uint8(255 * cam)


def show_gradcam_mislabelled(model, device, x_test, y_test, y_pred,
                             test_dataset, mean_tuple, std_tuple,
                             layer='layer3',
                             disp_nums=3, fig_size=(40, 10)):
    # Define the normalization and pre-processing steps
    normalize = transforms.Normalize(mean=mean_tuple,
                                     std=std_tuple)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    if device == 'cuda':
        x_test = x_test.cuda()
    import matplotlib.gridspec as gridspec
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'grad_cam'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    class_names = test_dataset.classes
    NUM_DISP = disp_nums
    idx1 = np.where(y_test[:] != y_pred)[0]
    for _ in range(NUM_DISP):
        fig = plt.figure(figsize=fig_size)
        outer = gridspec.GridSpec(1, len(class_names), wspace=0.2, hspace=0.2)
        for i in range(len(class_names)):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                     subplot_spec=outer[i],
                                                     wspace=0.1, hspace=0.1)
            idx = np.where(y_test[:] == i)[0]
            intsct = np.intersect1d(idx1, idx)
            features_idx = x_test[intsct, ::]
            img_num = np.random.randint(features_idx.shape[0])
            im_orig = features_idx[img_num].numpy()
            im_orig = preprocess(im_orig).unsqueeze(0)
            model.eval()
            # Instantiate GradCAM
            layer_names = [layer]
            layers = [model._modules[name] for name in layer_names]
            gradcam = GradCAM(model=model, target_layers=layers, use_cuda=True)
            cam = gradcam(im_orig)
            heatmap = cam[0]
            for j in range(2):
                ax = plt.Subplot(fig, inner[j])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if args.dataset == 'CIFAR10':
                    _ = ax.set_title('Act:{} '.format(
                        test_dataset.classes[int(i)]) + ' Pred:{} '.format(
                        test_dataset.classes[int(y_pred[intsct[img_num]][0])]),
                                     fontsize=8)
                elif args.dataset == 'MNIST':
                    _ = ax.set_title('Act:{} '.format(i) + ' Pred:{} '.format(
                        int(y_pred[intsct[img_num]][0])),
                                     fontsize=8)
                if j == 0:
                    im_orig = im_orig.permute(0, 2, 3, 1).squeeze().numpy()
                    _ = ax.imshow(np.clip(im_orig, 0, 1))
                    plt.savefig('plot10.png')
                    display(plt.gcf())
                    from IPython.display import Image
                    Image(filename='plot10.png')
                    display(plt.gcf())
                else:
                    super_imposed_img = superimpose(im_orig, heatmap)
                    _ = ax.imshow(super_imposed_img)
                    plt.savefig('plot11.png')
                    from IPython.display import Image
                    Image(filename='plot11.png')
                    display(plt.gcf())
                _ = fig.add_subplot(ax)
        if not args.IPYNB_ENV:
            fig.show()
            fig.savefig(filepath)
        else:
            fig.show()
    plt.savefig('plot12.png')
    from IPython.display import Image
    Image(filename='plot12.png')
    display(plt.gcf())
    plt.show()
