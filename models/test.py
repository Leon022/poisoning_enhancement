#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.7

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.poison_data import MyDataset_cifar, MyDataset_mnist, MyDataset_cifar_DBA, MyDataset_mnist_DBA
from utils.options import args_parser

args = args_parser()
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def test_img_poison(net_g, datatest, args, epoch):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    datatest = list(datatest)
    if args.dataset == 'cifar10' or args.dataset == 'gtsrb':
        if args.attack_methods == 'CBA' or args.attack_methods == 'MP':
            datatest_poison = MyDataset_cifar(datatest[:3000], args.target_label, portion=1.0,
                                             mode="test")
        elif args.attack_methods == 'DBA':
            datatest_poison = MyDataset_cifar_DBA(args, datatest[:3000], args.target_label,
                                                 portion=1.0, mode="test", idx=-1)
    elif args.dataset == 'mnist':
        if args.attack_methods == 'CBA' or args.attack_methods == 'MP':
            datatest_poison = MyDataset_mnist(datatest[:3000], args.target_label, portion=1.0,
                                             mode="test")
        elif args.attack_methods == 'DBA':
            datatest_poison = MyDataset_mnist_DBA(args, datatest[:3000], args.target_label,
                                                 portion=1.0, mode="test", idx=-1)
    data_loader = DataLoader(datatest_poison, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        # sum up batch loss
        # target = torch.argmax(target, dim=1)
        test_loss += F.cross_entropy(log_probs, target.long(), reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def test_distance(args, model, dataset):
    featureListnum = []
    dis_loss_n = 0
    for k in [6, 9]:
        if args.dataset == 'cifar10' or args.dataset == 'mnist':
            testloader = DataLoader(dataset[k][:200], batch_size=100, shuffle=True)
        elif args.dataset == 'gtsrb':
            testloader = DataLoader(dataset[k][:100], batch_size=100, shuffle=True)
        model.eval()
        model.to(args.device)

        # 进行测试, 生成测试数据
        for i, (datas, labels) in enumerate(testloader):
            batch_size = 100
            datas = datas.to(args.device)
            x = model(datas)
            x = x.cpu().detach().numpy().reshape(batch_size, -1)
            labels = labels.cpu().detach().numpy()
            if i == 0:
                featureList = x  # 保存encode之后的特征
                labelsList = labels  # 保存对应的label
            else:
                featureList = np.append(featureList, x, axis=0)
                labelsList = np.append(labelsList, labels, axis=0)

        featureList_A = np.mean(featureList, axis=0)
        # featureList_B = np.mean(featureList, axis=1)
        featureListnum.append(featureList_A)
        # dis_loss_n += np.var(featureList_B)

    dis_loss_j = np.linalg.norm(featureListnum[0] - featureListnum[1])

    return dis_loss_j