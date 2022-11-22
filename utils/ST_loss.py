import math
import sys
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = math.pi**2 / 8


def normal_cdf(value, loc=0.0, scale=1.0):
    return 0.5 * (1 + torch.erf((value - loc) / (scale * math.sqrt(2))))


def sigmoid_expectation(mu, sigma2, alpha=ALPHA):
    return normal_cdf(alpha * mu / torch.sqrt(1 + alpha * alpha * sigma2))


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter=0):
        return self.value

    def __call__(self, iter=0):
        return self.value


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num,
                                      feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        """features: Tensor(N, A)
        labels: Tensor(N, C)
        """
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        NxCxA_labels = labels.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_labels)

        Amount_CxA = NxCxA_labels.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_labels)

        var_temp = torch.bmm(var_temp.permute(1, 2, 0),
                             var_temp.permute(1, 0, 2)).div(
                                 Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = labels.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = labels.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV +
                                      self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm((self.Ave - ave_CxA).view(C, A, 1),
                      (self.Ave - ave_CxA).view(C, 1, A)))

        self.CoVariance = (
            self.CoVariance.mul(1 - weight_CV) +
            var_temp.mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) +
                    ave_CxA.mul(weight_AV)).detach()

        self.Amount += labels.sum(0)


class STLoss(nn.Module):
    def __init__(self, feature_num, class_num, instance_num, p_cutoff):
        super(STLoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.instance_num = instance_num

        self.cross_entropy = nn.CrossEntropyLoss()

        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function

        # for flexmatch
        self.selected_label = (torch.ones(
            (self.instance_num, ), dtype=torch.long) * -1).cuda()
        self.classwise_acc = torch.zeros((class_num, )).cuda()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        """for E_a[-log p(j|x; Theta, W)]"""
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxCxW_ij = weight_m.expand(N, C, C, A)

        NxCxW_kj = weight_m.expand(N, 1, C, A).permute(0, 2, 1, 3)

        CV_temp = cv_matrix[labels]

        NxCxW_ij_NxCxW_kj = (NxCxW_ij - NxCxW_kj).view(N, C * C, A)
        sigma2 = ratio * torch.bmm(
            torch.bmm(NxCxW_ij_NxCxW_kj, CV_temp).view(N * C * C, 1, A),
            NxCxW_ij_NxCxW_kj.view(N * C * C, A, 1)).view(N, C, C)

        aug_result = F.softmax(y.view(N, 1, C) + 0.5 * sigma2, dim=2)
        aug_result = aug_result.mul(torch.eye(C).cuda().expand(N, C,
                                                               C)).sum(2).view(
                                                                   N, C)
        aug_result = aug_result.clamp(1e-6, 1. - 1e-6)

        return aug_result

    def isda_aug1(self, fc, features, y, labels, cv_matrix, ratio):
        """for E_a[p(j|x; Theta, W)]"""
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        mu = y.view(N, C, 1).expand(N, C, C) - y.view(N, 1, C)

        weight_m = list(fc.parameters())[0]

        NxCxW_ij = weight_m.expand(N, C, C, A)

        NxCxW_kj = weight_m.expand(N, 1, C, A).permute(0, 2, 1, 3)

        CV_temp = cv_matrix[labels]

        NxCxW_kj_NxCxW_ij = (NxCxW_kj - NxCxW_ij).view(N, C * C, A)
        sigma2 = ratio * torch.bmm(
            torch.bmm(NxCxW_kj_NxCxW_ij, CV_temp).view(N * C * C, 1, A),
            NxCxW_kj_NxCxW_ij.view(N * C * C, A, 1)).view(N, C, C)

        sigmoid_exp = sigmoid_expectation(mu, sigma2)
        sigmoid_exp = sigmoid_exp.clamp(1e-6, 1. - 1e-6)
        aug_result = (sigmoid_exp.reciprocal().sum(2) - C).reciprocal()
        aug_result = aug_result.clamp(1e-6, 1. - 1e-6)

        return aug_result, sigmoid_exp

    def forward(self,
                model,
                fc,
                x,
                partial_target_x,
                pseudo_target_x_l,
                idx_x,
                ratio,
                threshold_warmup=True):
        """Parameters:
        model: feature extractor
        fc: the single-layer classifier
        x: dict{'l': Tensor(N_l, A), 'u_w': Tensor(N_u, A), 'u_s': Tensor(N_u, A)}
        partial_target_x: dict{'l': Tensor(N_l, C), 'u': Tensor(N_u, C)}
        pseudo_target_x_l: Tensor(N_l, C)
        ratio: float
        threshold_warmup: boolean
        """

        # for flexmatch
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.instance_num:
            if threshold_warmup:
                for i in range(self.class_num):
                    self.classwise_acc[i] = pseudo_counter[i] / max(
                        pseudo_counter.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.class_num):
                    self.classwise_acc[i] = pseudo_counter[i] / max(
                        wo_negative_one.values())

        # hyper-params for update
        p_cutoff = self.p_fn()

        x_l = x['l']
        x_u_w = x['u_w']
        x_u_s = x['u_s']
        x_ = torch.cat([x_l, x_u_w, x_u_s], dim=0)
        N_l = x_l.size(0)
        N_u = x_u_w.size(0)

        idx_x_l = idx_x['l']
        idx_x_u = idx_x['u']

        partial_target_x_l = partial_target_x['l']
        partial_target_x_u = partial_target_x['u']
        partial_target_x_ = torch.cat([partial_target_x_l, partial_target_x_u],
                                      dim=0)

        features = model(x_)
        y = fc(features)

        # Update CV with labeled instances
        self.estimator.update_CV((features[:N_l]).detach(), pseudo_target_x_l)
        CoVariance = self.estimator.CoVariance.detach()

        # obtain the pseudo labels of unlabeled instances with their class
        # activated values
        outputs_u = (y[N_l:N_l + N_u]).detach()
        _, pseudo_target_x_u_ = torch.max(
            (outputs_u * torch.abs(outputs_u - 1.0) * partial_target_x_u).data,
            1)

        features_u = features[N_l:]
        y_u = y[N_l:]
        _, pseudo_target_x_l_ = torch.max(pseudo_target_x_l, 1)

        # for labeled instances, as well as strong-augmented and weak-augmented unlabeled instances
        isda_aug_y = self.isda_aug(
            fc, features, y,
            torch.cat(
                [pseudo_target_x_l_, pseudo_target_x_u_, pseudo_target_x_u_],
                dim=0), CoVariance, ratio)

        with torch.no_grad():
            isda_aug_y1_u_ori, _ = self.isda_aug1(fc, features[N_l:N_l + N_u],
                                                  y[N_l:N_l + N_u],
                                                  pseudo_target_x_u_,
                                                  CoVariance, ratio)

            isda_aug_y1_u = isda_aug_y1_u_ori * partial_target_x_u
            isda_aug_y1_u = isda_aug_y1_u / isda_aug_y1_u.sum(dim=1,
                                                              keepdim=True)
            isda_aug_y1_u = isda_aug_y1_u.detach()
            isda_aug_y1_u_ori = isda_aug_y1_u_ori.detach()

        max_probs, pseudo_target_x_u = torch.max(isda_aug_y1_u_ori, dim=-1)
        mask = max_probs.ge(
            p_cutoff *
            (self.classwise_acc[pseudo_target_x_u] /
             (2. - self.classwise_acc[pseudo_target_x_u]))).float()  # convex

        select = max_probs.ge(p_cutoff).long()
        if idx_x_u[select == 1].nelement() != 0:
            self.selected_label[idx_x_u[select == 1]] = (
                pseudo_target_x_u.long())[select == 1]

        pseudo_target_x_u1 = torch.zeros_like(partial_target_x_u).scatter_(
            1, pseudo_target_x_u.view(-1, 1), 1)
        mask1 = ((pseudo_target_x_u1 *
                  partial_target_x_u).sum(1)).ge(0.0).float()

        labeled_loss = -(pseudo_target_x_l *
                         torch.log(isda_aug_y[:N_l])).sum(-1).mean()
        unlabeled_loss = -(
            (isda_aug_y1_u * torch.log(isda_aug_y[N_l + N_u:])).sum(-1) *
            mask * mask1).mean()

        cl_loss = -(
            ((1.0 - partial_target_x_) *
             torch.log(1.0000001 - isda_aug_y[:N_l + N_u])).sum(-1)).mean()

        return labeled_loss, unlabeled_loss, cl_loss
