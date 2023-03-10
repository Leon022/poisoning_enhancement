#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--rho', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--aggregation_methods', type=str,  default="fedavg",
                        help='fedavg, RFA, foolsgold, RLR')
    parser.add_argument('--robustLR_threshold', type=int, default=0,
                        help="For RLR: break ties when votes sum to 0")

    # dataset and model arguments
    parser.add_argument('--dataset', type=str, default='gtsrb', help="name of dataset")
    parser.add_argument('--iid', type=str, default=True, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=43, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--model', type=str, default='alexnet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # attack arguments
    parser.add_argument('--attack_start', type=int, default=4, help="the epoch of attack starting")
    parser.add_argument('--local_ep_ba', type=int, default=10, help="the number of local backdoor epochs: E")
    parser.add_argument('--local_bs_ba', type=int, default=64, help="local backdoor batch size: B")
    parser.add_argument('--lr_ba', type=float, default=0.01, help="learning backdoor rate: lr")
    parser.add_argument('--attack_methods', type=str, default="MP",
                        help='CBA, DBA, MP')
    parser.add_argument('--attacker_list', nargs='+', type=int, default=[10,20,30,40,50,60,70,80,90,0])
    parser.add_argument('--base_label', type=int, default=-6, help="poison source label (-1 means all labels)")
    parser.add_argument('--target_label', type=int, default=9, help="poison target label")
    parser.add_argument('--back_prop', type=float, default=0.3, help="poison data proportion")


    # PoE arguments
    parser.add_argument('--use_en', type=str, default=True, help='whether launch PoE')
    parser.add_argument('--epoch_en', type=int, default=50, help="the number of PoE epochs")
    parser.add_argument('--epsilon', type=float, default=0.1, help="smoothing ratio")


    #  other  arguments
    parser.add_argument('--gpu', type=int, default=5, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--use_seed', type=str, default='False', help='whether use random seed')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args


