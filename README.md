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
