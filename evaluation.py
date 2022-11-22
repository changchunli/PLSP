import time

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import (linear_kernel, pairwise_kernels,
                                      rbf_kernel)

from utils.utils import AverageMeter


def validate(val_loader, model, fc, criterion, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    predicted_total = np.array([])
    targets_total = np.array([])

    model.eval()
    fc.eval()

    end = time.time()
    for i, (inputs, labels) in enumerate(val_loader):
        if len(np.shape(labels)) == 2:
            targets_temp = np.array(np.where(labels == 1)[1])
            targets_total = np.hstack((targets_total, targets_temp))
            labels = torch.tensor(targets_temp)
        elif len(np.shape(labels)) == 1:
            targets_total = np.hstack((targets_total, labels))

        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = fc(model(inputs))

        # measure accuracy and record loss
        loss = criterion(outputs, labels)
        losses.update(loss.data.item(), inputs.size(0))

        _, predicted = torch.max(outputs.data, 1)
        predicted_total = np.hstack(
            (predicted_total, np.array(predicted.cpu())))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    macro_f1, micro_f1 = classification_metrics(targets_total, predicted_total)
    return macro_f1, micro_f1, batch_time, losses


def classification_metrics(labels_truth, labels_predicted):
    macro_f1 = f1_score(labels_truth, labels_predicted, average='macro')
    micro_f1 = f1_score(labels_truth, labels_predicted, average='micro')
    (precision, recall, fbeta_score,
     support) = precision_recall_fscore_support(labels_truth, labels_predicted)

    return macro_f1, micro_f1
