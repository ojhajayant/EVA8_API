###  Following is the directory struct being followed:

![alt text](https://github.com/ojhajayant/EVA8_API/blob/main/EVA8_API_DIR_STRUCT.png "Logo Title Text 1")

> Here are a little details on the above struct:

 *   Create:
models folder - this is where you'll add all of your future models. 

*   Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class

*   main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):

    > training and test loops

    > data split between test and train

    > epochs

    > batch size

    > which optimizer to run

    > do we run a scheduler?

*   utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:

    > image transforms,

    > gradcam,

    > misclassification code,

    > tensorboard related stuff

    > advanced training policies, etc etc


Here are the different args values for this different runs:

	> cmd : Either of "lr_find", "train", "test"

	> IPYNB_ENV : True

	> use_albumentations : True

	> SEED : 1

	> dataset : CIFAR10

	> img_size : (32, 32)

	> batch_size : 128
  
        > epochs : 20

	> criterion : NLLLoss()

	> init_lr : 0.0001 (for LR-Range test)

	> end_lr : 0.05 (for LR-Range test)

	> lr_range_test_epochs : 100 (epochs used for LR-Range test)

	> best_lr : 0.504999999999

	> cycle_momentum : True

	> optimizer : <class 'torch.optim.sgd.SGD'>

	> cuda : True

	> dropout : 0.08

	> l1_weight : 2.5e-05

	> l2_weight_decay : 0.0002125

	> L1 : True

	> L2 : False

	> data : ./data/

	> best_model_path : ./saved_models/

	> prefix : data

	> best_model :  CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-90.6.h5
    
    
 The example runs once this has been cloned at the user area:
 
 ```
 !git clone https://git@github.com/ojhajayant//EVA8_API.git
 ```
 
 1. "lr_find" cmd:
 
 ```
 !python /content/EVA8_API/main.py --cmd lr_find
 ```
 
 2. "train" cmd:
 
 ```
 !python /content/EVA8_API/main.py --cmd train --best_lr 0.050499
 ```
 
 3. "test" cmd:
 
 ```
 !python /content/EVA8_API/main.py --cmd test --best_model CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-90.6.h5
 ```
