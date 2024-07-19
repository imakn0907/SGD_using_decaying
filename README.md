# Iteration and stochastic first-order oracle complexities of stochastic gradient descent using constant and decaying learning rates
Code for reproducing experiments in our paper.

## Abstract
The performance of stochastic gradient descent (SGD), which is the simplest first-order optimizer for training deep neural networks, depends on not only the learning rate but also the batch size. They both affect the number of iterations and the stochastic first-order oracle (SFO) complexity needed for training. In particular, the previous numerical results indicated that, for SGD using a constant learning rate, the number of iterations needed for training decreases when the batch size increases, and the SFO complexity needed for training is minimized at a critical batch size and that it increases once the batch size exceeds that size. Here, we study the relationship between batch size and the iteration and SFO complexities needed for nonconvex optimization in deep learning with SGD using constant or decaying learning rates and show that SGD using the critical batch size minimizes the SFO complexity. We also provide numerical comparisons of SGD with the existing first-order optimizers and show the usefulness of SGD using a critical batch size. Moreover, we show that measured critical batch sizes are close to the sizes estimated from our theoretical results.

## Wandb Setup
Please change entity name `xxxxx` to your wandb entitiy.
```
wandb.init(config = args,
           project = wandb_project_name,
           name = wandb_exp_name,
           entity = "×××××")
```

## Usage
Training on CIFAR-10 dataset.
```
python3 main_CIFAR10.py
```
Training on CIFAR-100 dataset.
```
python3 main_CIFAR-100.py
```
## Cite
If you use this code or our results in your research, please cite:
```
@article{doi:10.1080/02331934.2024.2367635,
author = {Kento Imaizumi and Hideaki Iiduka},
title = {Iteration and stochastic first-order oracle complexities of stochastic gradient descent using constant and decaying learning rates},
journal = {Optimization},
volume = {0},
number = {0},
pages = {1--24},
year = {2024},
publisher = {Taylor \& Francis},
doi = {10.1080/02331934.2024.2367635},
URL = {https://doi.org/10.1080/02331934.2024.2367635},
eprint = {https://doi.org/10.1080/02331934.2024.2367635}
}
```
