import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg, stats
from scipy.special import comb
from torchvision.transforms import (Compose, Lambda, Normalize, Pad,
                                    RandomCrop, RandomErasing,
                                    RandomHorizontalFlip, ToPILImage, ToTensor)

from utils.augment.autoaugment_extra import CIFAR10Policy, ImageNetPolicy
from utils.augment.cutout import Cutout
from utils.gen_index_dataset import (NumpyDataset, gen_index_dataset,
                                     labeled_dataset, unlabeled_dataset)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.485, 0.456, 0.406)  # (0.5071, 0.4867, 0.4408)
cifar100_std = (0.229, 0.224, 0.225)  # (0.2675, 0.2565, 0.2761)
fmnist_mean = (0.1307)
fmnist_std = (0.3081)

datanames_cv_no_augmentation = ['fmnist']
datanames_cv_augmentation = ['cifar10', 'cifar100']


class TransformFixMatch(object):
    def __init__(self, mean, std, augment='autoaugment'):
        self.weak = Compose([
            ToPILImage(),
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
        ])

        self.strong = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16)
        ])

        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformMNIST(object):
    def __init__(self, mean, std):
        self.weak = Compose([
            RandomHorizontalFlip(),
            RandomCrop(28, 4, padding_mode='reflect'),
            ToTensor(),
        ])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16)
        ])

        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def get_ssl_dataloader(args, train_dataset, indexs_l, indexs_u, pseudo_labels):

    train_transform_labeled, _ = get_transform(args)
    if args.dataname == 'cifar10':
        train_transform_unlabeled = TransformFixMatch(mean=cifar10_mean,
                                                      std=cifar10_std)
    elif args.dataname == 'cifar100':
        train_transform_unlabeled = TransformFixMatch(mean=cifar100_mean,
                                                      std=cifar100_std)
    elif args.dataname == 'fmnist':
        train_transform_unlabeled = TransformMNIST(mean=fmnist_mean,
                                                   std=fmnist_std)

    labeled_train_dataset = labeled_dataset(train_dataset,
                                            pseudo_labels,
                                            indexs_l,
                                            transform=train_transform_labeled)
    unlabeled_train_dataset = unlabeled_dataset(
        train_dataset, indexs_u, transform=train_transform_unlabeled)
    labeled_train_loader = torch.utils.data.DataLoader(
        dataset=labeled_train_dataset,
        batch_size=args.batch_size // 4,
        shuffle=True,
        drop_last=True,
        num_workers=0)
    unlabeled_train_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0)

    return labeled_train_loader, unlabeled_train_loader


def get_transform(args):
    if args.dataname in datanames_cv_no_augmentation:
        if args.dataname == 'fmnist':
            normalize = transforms.Normalize(mean=fmnist_mean, std=fmnist_std)

        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        print('Standard Augmentation for Labeled Instances!')
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            normalize,
        ])

    elif args.dataname in datanames_cv_augmentation:
        if args.dataname == 'cifar10':
            normalize = transforms.Normalize(mean=cifar10_mean,
                                             std=cifar10_std)
        elif args.dataname == 'cifar100':
            normalize = transforms.Normalize(mean=cifar100_mean,
                                             std=cifar100_std)

        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        print('Standard Augmentation for Labeled Instances!')
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            normalize,
        ])

    return train_transform, test_transform


def generate_uniform_cv_candidate_labels_fps(dataname,
                                             train_labels,
                                             partial_rate=0.7):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix = np.eye(K)
    transition_matrix[np.where(
        ~np.eye(transition_matrix.shape[0], dtype=bool))] = p_1
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K):  # for each class
            if jj == train_labels[j]:  # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def generate_uniform_cv_candidate_labels_uss(dataname, train_labels):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    cardinality = (2**K - 2).float()
    number = torch.tensor([comb(K, i + 1) for i in range(K - 1)]).float(
    )  # 1 to K-1 because cannot be empty or full label set, convert list to tensor
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K - 1)  # tensor of K-1
    for i in range(K - 1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

    random_n = torch.from_numpy(np.random.uniform(0, 1,
                                                  n)).float()  # tensor: n
    mask_n = torch.ones(n)  # n is the number of train_data
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    temp_num_partial_train_labels = 0  # save temp number of partial train_labels

    for j in range(n):  # for each instance
        for jj in range(K - 1):  # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj + 1  # decide the number of partial train_labels
                mask_n[j] = 0

        temp_num_fp_train_labels = temp_num_partial_train_labels - 1
        candidates = torch.from_numpy(np.random.permutation(
            K.item())).long()  # because K is tensor type
        candidates = candidates[candidates != train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]

        partialY[
            j, temp_fp_train_labels] = 1.0  # fulfill the partial label matrix
    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def prepare_cv_datasets(args):
    _, test_transform = get_transform(args)
    if args.dataname == 'fmnist':
        ordinary_train_dataset = dsets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=True,
                                                    download=True)
        test_dataset = dsets.FashionMNIST(root='./data/FashionMNIST',
                                          train=False,
                                          transform=test_transform)
    elif args.dataname == 'cifar10':
        ordinary_train_dataset = dsets.CIFAR10(root='./data/cifar10',
                                               train=True,
                                               download=True)
        test_dataset = dsets.CIFAR10(root='./data/cifar10',
                                     train=False,
                                     transform=test_transform)
    elif args.dataname == 'cifar100':
        ordinary_train_dataset = dsets.CIFAR100(root='./data/cifar100',
                                                train=True,
                                                download=True)
        test_dataset = dsets.CIFAR100(root='./data/cifar100',
                                      train=False,
                                      transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0)
    if args.dataname == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    return (train_loader, test_loader, ordinary_train_dataset, test_dataset,
            num_classes)


def prepare_train_loaders_for_uniform_cv_candidate_labels(
        dataname, ordinary_train_dataset, batch_size, partial_rate):
    data = ordinary_train_dataset.data
    if dataname in datanames_cv_no_augmentation:
        data = np.array([
            Image.fromarray(data[i].numpy(), mode="L")
            for i in range(data.shape[0])
        ],
                        dtype=object)
    labels = torch.tensor(ordinary_train_dataset.targets)
    K = torch.max(labels) + 1
    if partial_rate > 0:
        partialY = generate_uniform_cv_candidate_labels_fps(
            dataname, labels, partial_rate)
    elif partial_rate == -1:
        partialY = generate_uniform_cv_candidate_labels_uss(dataname, labels)
    trueY = torch.zeros_like(partialY)
    trueY[torch.arange(trueY.size(0)), labels] = 1.0
    partial_matrix_dataset = gen_index_dataset(data, partialY.float(),
                                               trueY.float())
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=partial_matrix_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0)
    dim = int(data.reshape(-1).shape[0] / data.shape[0])
    return (partial_matrix_train_loader, partial_matrix_dataset, dim)
