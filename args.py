import argparse

import numpy as np
import torch

# using GPU if available
# device = torch.device("cpu")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_args():
    parser = argparse.ArgumentParser(
        description=
        'Learning with Partial Labels from Semi-supervised Perspective (PLSP)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--pretrain-epoch', type=int, default=10)
    parser.add_argument('--train-iterations',
                        default=200,
                        type=int,
                        help='the number of iterations in one epoch')
    parser.add_argument('--batch-size', type=int, default=256)

    parser.add_argument('--checkpoint',
                        default='checkpoint',
                        type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument(
        '--exp-dir',
        default='experiment/PLLSSL',
        type=str,
        help='experiment directory for saving checkpoints and logs')
    parser.add_argument('--print-freq',
                        '-p',
                        default=50,
                        type=int,
                        help='print frequency (default: 10)')

    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--threshold-warmup',
                        dest='threshold_warmup',
                        action='store_true')

    parser.add_argument('--gamma_0',
                        default=1.0,
                        type=float,
                        help='hyper-patameter gamma_0 for weight of ss loss')
    parser.add_argument('--lambda_0',
                        default=0.05,
                        type=float,
                        help='hyper-patameter_\lambda for ISDA')

    parser.add_argument('--dataname',
                        default='cifar10',
                        type=str,
                        help='dataset (cifar10 [default] or cifar100)')
    parser.add_argument('--model',
                        default='densenet',
                        type=str,
                        help='deep networks to be trained')

    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--partial_rate',
                        default=0.1,
                        type=float,
                        help='ambiguity level (q)')

    # Select pseudo-labeled instances
    parser.add_argument(
        '--num-labeled-instances',
        type=int,
        default=200,
        help='The number of selected pseudo-labeled instances in each label')

    args = parser.parse_args()
    return args
