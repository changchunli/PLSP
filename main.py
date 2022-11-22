import errno
import math
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import LambdaLR

import args
from args import process_args
from cifar_models import densenet, resnet18
from evaluation import validate
from utils.gen_index_dataset import gen_index_dataset
from utils.models import LeNet, linear_model, mlp_model
from utils.ST_loss import EstimatorCV, STLoss
from utils.utils import AverageMeter, mkdir_p, save_checkpoint
from utils.utils_data import (
    get_ssl_dataloader, get_transform, prepare_cv_datasets,
    prepare_train_loaders_for_uniform_cv_candidate_labels)

args = process_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


def seed_torch(seed=args.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


seed_torch()

record_path = 'ds_{ds}_pr_{pr}_arch_{arch}_op_{op}_lr_{lr}_wd_{wd}_gamma0_{gamma0}_lambda0_{lambda0}_threshold_{threshold}_pte_{pte}_nl_{nl}_sd_{seed}'.format(
    ds=args.dataname,
    pr=args.partial_rate,
    arch=args.model,
    op='sgd',
    lr=args.lr,
    wd=args.wd,
    gamma0=args.gamma_0,
    lambda0=args.lambda_0,
    threshold=args.threshold,
    pte=args.pretrain_epoch,
    nl=args.num_labeled_instances,
    seed=args.seed)
args.exp_dir = os.path.join(args.exp_dir, record_path)
if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)

record_file = args.exp_dir + '/training_process.txt'
micro_f1_file = args.exp_dir + '/micro_f1_epoch.txt'
macro_f1_file = args.exp_dir + '/macro_f1_epoch.txt'
check_point = os.path.join(args.exp_dir, args.checkpoint)


def main():
    best_prec1 = 0
    val_macro_f1_all = []
    val_micro_f1_all = []

    (_, test_loader, ordinary_train_dataset, _,
     class_num) = prepare_cv_datasets(args)
    (partial_matrix_train_loader, train_dataset,
     dim) = prepare_train_loaders_for_uniform_cv_candidate_labels(
         dataname=args.dataname,
         ordinary_train_dataset=ordinary_train_dataset,
         batch_size=args.batch_size,
         partial_rate=args.partial_rate)
    _, test_transform = get_transform(args)
    warmup_train_dataset = gen_index_dataset(train_dataset.images,
                                             train_dataset.given_label_matrix,
                                             train_dataset.true_labels,
                                             transform=test_transform)
    warmup_train_loader = torch.utils.data.DataLoader(
        dataset=warmup_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    model = create_model(dim, class_num)
    fc = Full_layer(int(model.feature_num), class_num)

    print('Number of final features: {}'.format(int(model.feature_num)))

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]) +
        sum([p.data.nelement() for p in fc.parameters()])))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    st_criterion = STLoss(int(model.feature_num), class_num,
                          len(train_dataset), args.threshold).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()

    model = torch.nn.DataParallel(model).cuda()
    fc = torch.nn.DataParallel(fc).cuda()

    grouped_parameters = [{
        'params': model.parameters(),
        'weight_decay': args.wd,
    }, {
        'params': fc.parameters(),
        'weight_decay': args.wd,
    }]
    optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     last_epoch=-1)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        fc.load_state_dict(checkpoint['fc'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        st_criterion = checkpoint['st_criterion']
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(args.pretrain_epoch):
        pretrain(warmup_train_loader, model, fc, optimizer, scheduler, epoch)

    for epoch in range(start_epoch, args.epochs):
        indexs_l, indexs_u, pseudo_labels = create_ssl_dataset(
            train_dataset, model, fc, class_num, epoch, args)
        labeled_train_loader, unlabeled_train_loader = get_ssl_dataloader(
            args, train_dataset, indexs_l, indexs_u, pseudo_labels)
        train_loader = {'l': labeled_train_loader, 'u': unlabeled_train_loader}
        train_ssl(train_loader, model, fc, st_criterion, optimizer, scheduler,
                  epoch)

        scheduler.step()

        # evaluate on validation set
        (val_macro_f1, val_micro_f1, val_batch_time,
         val_losses) = validate(test_loader, model, fc, ce_criterion, epoch)
        with open(record_file, 'a+') as fd:
            string = ('Test: [{0}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Macro F1 {macro_f1:.4f}\t'
                      'Micro F1 {micro_f1:.4f}\t'.format(
                          epoch,
                          batch_time=val_batch_time,
                          loss=val_losses,
                          macro_f1=val_macro_f1,
                          micro_f1=val_micro_f1))
            print(string)
            fd.write(string + '\n')

        val_macro_f1_all.append(val_macro_f1)
        val_micro_f1_all.append(val_micro_f1)

        # remember best prec@1 and save checkpoint
        is_best = val_micro_f1 > best_prec1
        best_prec1 = max(val_micro_f1, best_prec1)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'fc': fc.state_dict(),
                'best_acc': best_prec1,
                'optimizer': optimizer.state_dict(),
                'st_criterion': st_criterion,
                'val_macro_f1': val_macro_f1_all,
                'val_micro_f1': val_micro_f1_all,
            },
            is_best,
            checkpoint=check_point)
        print('Best accuracy: ', best_prec1)
        np.savetxt(micro_f1_file, np.array(val_micro_f1_all))
        np.savetxt(macro_f1_file, np.array(val_macro_f1_all))

    print('Best accuracy: ', best_prec1)
    print('Average accuracy',
          sum(val_micro_f1_all[len(val_micro_f1_all) - 10:]) / 10)
    np.savetxt(micro_f1_file, np.array(val_micro_f1_all))
    np.savetxt(macro_f1_file, np.array(val_macro_f1_all))


def pretrain(train_loader, model, fc, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    train_loader_iter = iter(train_loader)

    # switch to train mode
    model.train()
    fc.train()

    end = time.time()
    for i in range(args.train_iterations):
        # load partial labeled instances
        try:
            x, target, _, _ = train_loader_iter.next()
        except:
            train_loader_iter = iter(train_loader)
            x, target, _, _ = train_loader_iter.next()

        x, target = x.cuda(), target.cuda()
        output = fc(model(x))

        # disambiguation-free loss
        confidence = target / target.sum(dim=1, keepdim=True)
        loss_df = -(
            (confidence * F.log_softmax(output, dim=1)).sum(dim=1)).mean()
        loss_cl = -(
            (1.0 - target) *
            torch.log(1.0000001 - F.softmax(output, dim=1))).sum(-1).mean()
        loss = loss_df + loss_cl

        # measure accuracy and record loss
        losses.update(loss.data.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            with open(record_file, 'a+') as fd:
                string = (
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                    'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                        epoch,
                        i + 1,
                        args.train_iterations,
                        batch_time=batch_time,
                        loss=losses))

                print(string)
                fd.write(string + '\n')


def train_ssl(train_loader, model, fc, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    labeled_train_loader = train_loader['l']
    unlabeled_train_loader = train_loader['u']

    labeled_train_loader_iter = iter(labeled_train_loader)
    unlabeled_train_loader_iter = iter(unlabeled_train_loader)

    ratio = args.lambda_0 * (epoch / args.epochs)

    # switch to train mode
    model.train()
    fc.train()

    end = time.time()
    for i in range(args.train_iterations):
        # load pseudo-labeled instances
        try:
            x_l, partial_target_l, pseudo_target_l, _, idx_x_l = labeled_train_loader_iter.next(
            )
        except:
            labeled_train_loader_iter = iter(labeled_train_loader)
            x_l, partial_target_l, pseudo_target_l, _, idx_x_l = labeled_train_loader_iter.next(
            )

        # load pseudo-unlabeled instances
        try:
            (
                x_u_w, x_u_s
            ), partial_target_u, _, idx_x_u = unlabeled_train_loader_iter.next(
            )
        except:
            unlabeled_train_loader_iter = iter(unlabeled_train_loader)
            (
                x_u_w, x_u_s
            ), partial_target_u, _, idx_x_u = unlabeled_train_loader_iter.next(
            )

        x_l, partial_target_l, pseudo_target_l = x_l.cuda(
        ), partial_target_l.cuda(), pseudo_target_l.cuda()
        x_u_w, x_u_s, partial_target_u = x_u_w.cuda(), x_u_s.cuda(
        ), partial_target_u.cuda()

        x = {'l': x_l, 'u_w': x_u_w, 'u_s': x_u_s}
        idx_x = {'l': idx_x_l, 'u': idx_x_u}
        partial_target = {'l': partial_target_l, 'u': partial_target_u}

        labeled_loss, unlabeled_loss, cl_loss = criterion(
            model, fc, x, partial_target, pseudo_target_l, idx_x, ratio,
            args.threshold_warmup)

        lam = min((epoch / 50) * args.gamma_0, args.gamma_0)

        loss = lam * (labeled_loss + unlabeled_loss) + cl_loss

        # measure accuracy and record loss
        losses.update(loss.data.item(), x_l.size(0) + x_u_w.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            with open(record_file, 'a+') as fd:
                string = (
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                    'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                    'lam {lam:.4f} lr {lr:.4f}'.format(
                        epoch,
                        i + 1,
                        args.train_iterations,
                        batch_time=batch_time,
                        loss=losses,
                        lam=lam,
                        lr=scheduler.get_last_lr()[0]))

                print(string)
                fd.write(string + '\n')


def create_ssl_dataset(train_dataset, model, fc, K, epoch, args):
    _, test_transform = get_transform(args)
    train_dataset_ = gen_index_dataset(train_dataset.images,
                                       train_dataset.given_label_matrix,
                                       train_dataset.true_labels,
                                       transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset_,
                                               batch_size=2048,
                                               shuffle=False,
                                               num_workers=0)
    outputs_total = []
    labels_total = []
    true_labels_total = []
    indexs_total = []
    features_total = []

    model.eval()
    fc.eval()
    for i, (images, labels, true_labels, index) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            features = model(images)
            outputs = fc(features)
            outputs = outputs * torch.abs(outputs - 1.0)

        outputs_total.append(outputs)
        features_total.append(features)
        labels_total.append(labels)
        true_labels_total.append(true_labels)
        indexs_total.append(index)

    indexs_total_ = torch.cat(indexs_total)
    outputs_total = torch.cat(outputs_total, dim=0)
    features_total = torch.cat(features_total, dim=0)
    labels_total = torch.cat(labels_total, dim=0)
    true_labels_total = torch.cat(true_labels_total, dim=0)

    _, pseudo_labels1 = torch.max(outputs_total.data, 1)
    max_probs_, pseudo_labels2 = torch.max((outputs_total * labels_total).data,
                                           1)
    indexs_total = indexs_total_[(pseudo_labels2 == pseudo_labels1).cpu()]
    max_probs = max_probs_[(pseudo_labels2 == pseudo_labels1)]
    pseudo_labels = pseudo_labels2[(pseudo_labels2 == pseudo_labels1)]

    labeled_indexs = []
    unlabeled_indexs = []
    unlabeled_indexs.extend(
        indexs_total_[(pseudo_labels2 != pseudo_labels1).cpu()])
    for i in range(K):
        max_probs_i = max_probs[pseudo_labels == i]
        indexs_total_i = indexs_total[pseudo_labels == i]
        if max_probs_i.size(0) <= args.num_labeled_instances:
            labeled_indexs.extend(indexs_total_i)
        else:
            group_i = list(zip(max_probs_i, indexs_total_i))
            group_i.sort(key=lambda x: x[0], reverse=True)
            labeled_indexs.extend(
                [x[1] for x in group_i[:args.num_labeled_instances]])
            unlabeled_indexs.extend(
                [x[1] for x in group_i[args.num_labeled_instances:]])

    pseudo_labels_ = torch.zeros_like(outputs_total.cpu())
    pseudo_labels_[torch.arange(pseudo_labels1.size(0)), pseudo_labels1] = 1.0

    _, labeled_pseudo = torch.max(pseudo_labels_[labeled_indexs], 1)
    _, labeled_truth = torch.max(true_labels_total[labeled_indexs], 1)
    micro_f1 = f1_score(labeled_truth.numpy(),
                        labeled_pseudo.numpy(),
                        average='micro')
    print(micro_f1)

    return labeled_indexs, unlabeled_indexs, pseudo_labels_


class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''
    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


def create_model(dim, class_num):
    if args.model == 'mlp':
        model = mlp_model(input_dim=dim, hidden_dim=500, output_dim=class_num)
    elif args.model == 'linear':
        model = linear_model(input_dim=dim, output_dim=class_num)
    elif args.model == 'lenet':
        model = LeNet(output_dim=class_num
                      )  #  linear,mlp,lenet are for MNIST-type datasets.
    elif args.model == 'densenet':
        model = densenet(num_classes=class_num)
    elif args.model == 'resnet':
        model = resnet18(num_classes=class_num)

    return model


if __name__ == '__main__':
    main()
