# Gait Recognition

PyTorch re-implementation of some models in the paper [A comprehensive study on cross-view gait based human identification with deep CNNs](https://ieeexplore.ieee.org/abstract/document/7439821).


### Usage

0. Download our prepared [Casia-B data](https://drive.google.com/file/d/1YfmCKYoYJvxvOITdp4qxOi5ak0MTZJLD/view?usp=sharing).

0. Install the [fastai](https://github.com/fastai/fastai) library.

0. Clone this repo.

0. Extract the data and arrange the directories as follows.
    ```
    somedir
    |-- data
    |   `-- casiab-nm
    `-- gaitrecog
        |-- train_on_gei.py
        `-- ...
    ```

0. Go to somedir/gaitrecog, and either use the jupyter notebook train_on_gei.ipynb or run the following from a terminal.
    ```bash
    python train_on_gei.py --model lb --task tr --split tv
    python train_on_gei.py --model lb --task ts --split ts --trained casiab-nm_lb_sgd-0.01-0.9-0.0005_st-14-96_bs128_tv_239
    ```


### Note

Sometimes the training never converges due to bad initialization. In this case, stop the training and restart it.