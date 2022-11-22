# PLSP (AAAI2023) 
This is an official implementation of [Learning with Partial Labels from Semi-supervised Perspective], which is accepted by AAAI2023.

## Abstract
Partial Label (PL) learning refers to the task of learning from the partially labeled data, where each training instance is ambiguously equipped with a set of candidate labels but only one is valid. Advances in the recent deep PL learning literature have shown that the deep learning paradigms, e.g., self-training, contrastive learning, or class activate values, can achieve promising performance. Inspired by the impressive success of deep Semi-Supervised (SS) learning, we transform the PL learning problem into the SS learning problem, and propose a novel PL learning method, namely Partial Label learning with Semi-supervised Perspective (PLSP). Specifically, we first form the pseudo-labeled dataset by selecting a small number of reliable pseudo-labeled instances with high-confidence prediction scores and treating the remaining instances as pseudo-unlabeled ones. Then we design a SS learning objective, consisting of a supervised loss for pseudo-labeled instances and a semantic consistency regularization for pseudo-unlabeled instances. We further introduce a complementary regularization for those non-candidate labels to constrain the model predictions on them to be as small as possible. Empirical results demonstrate that PLSP significantly outperforms the existing PL baseline methods, especially on high ambiguity levels.

## Prerequisite
* The requirements are in **requirements.txt**. However, the settings are not limited to it (CUDA 11.0, Pytorch 1.7 for one RTX3090). 

## Usage
1. Train the model by running the following command directly.

For Fashion-MNIST:
```
python -u main.py --exp-dir experiment/results-fmnist --dataname fmnist --model lenet --epochs 250 --batch-size 256 --lr 0.1 --wd 1e-4 --threshold 0.75 --lambda_0 0.01 --num-labeled-instances 200 --start-ssl-epoch 10 --partial_rate 0.7
```

For CIFAR-10:
```
python -u main.py --exp-dir experiment/results-cifar10 --dataname cifar10 --model densenet --epochs 250 --batch-size 256 --lr 0.1 --wd 1e-4 --threshold 0.75 --lambda_0 0.01 --num-labeled-instances 200 --start-ssl-epoch 10 --partial_rate 0.1
```

For CIFAR-100:
```
python -u main.py --exp-dir experiment/results-cifar100 --dataname cifar100 --model resnet --epochs 250 --train-iterations 800 --batch-size 64 --lr 0.1 --wd 1e-4 --threshold 0.75 --lambda_0 0.01 --num-labeled-instances 200 --start-ssl-epoch 50 --partial_rate 0.2
```
