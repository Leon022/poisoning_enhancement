#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import math
import torch

from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.functional import log_softmax
import torch.nn.functional as F
from models.nets import LeNet, MLP, ResNet18, ResNet50
import time

logger = logging.getLogger("logger")
import os
import json
import numpy as np
import copy
from utils.options import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def init_weight_accumulator(target_model):
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(data)

    return weight_accumulator

def accumulate_weight(args, weight_accumulator, epochs_submit_update_dict, state_keys, num_samples_dict):
    """
     return Args:
         updates: dict of (num_samples, update), where num_samples is the
             number of training samples corresponding to the update, and update
             is a list of variable weights
     """
    if args.aggregation_methods == 'foolsgold':
        updates = dict()
        for i in range(0, len(state_keys)):
            local_model_gradients = epochs_submit_update_dict[state_keys[i]][0]  # agg 1 interval
            num_samples = num_samples_dict[state_keys[i]]
            updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
        return None, updates

    else:
        updates = dict()
        for i in range(0, len(state_keys)):
            local_model_update_list = epochs_submit_update_dict[state_keys[i]]
            update = dict()
            num_samples = num_samples_dict[state_keys[i]]

            for name, data in local_model_update_list[0].items():
                update[name] = torch.zeros_like(data)

            for j in range(0, len(local_model_update_list)):
                local_model_update_dict = local_model_update_list[j]
                for name, data in local_model_update_dict.items():
                    weight_accumulator[name].add_(local_model_update_dict[name])
                    update[name].add_(local_model_update_dict[name])
                    detached_data = data.cpu().detach().numpy()
                    # print(detached_data.shape)
                    detached_data = detached_data.tolist()
                    # print(detached_data)
                    local_model_update_dict[name] = detached_data  # from gpu to cpu

            updates[state_keys[i]] = (num_samples, update)

    return weight_accumulator, updates

def FedAvg(net, w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def average_shrink_models(weight_accumulator, target_model, epoch_interval):
    """
    Perform FedAvg algorithm and perform some clustering on top of it.

    """
    for name, data in target_model.state_dict().items():
        # if self.params.get('tied', False) and name == 'decoder.weight':
        #     continue

        eta=1
        update_per_layer = weight_accumulator[name] * (eta / (args.num_users))
        # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

        # update_per_layer = update_per_layer * 1.0 / epoch_interval
        # if self.params['diff_privacy']:
        #     update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
        data = data.float()
        update_per_layer = update_per_layer.float()
        data.add_(update_per_layer)

    return True, copy.deepcopy(target_model.state_dict())

def geometric_median_update(target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm= None):
    """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
           """
    points = []
    alphas = []
    names = []
    for name, data in updates.items():
        points.append(data[1]) # update
        alphas.append(data[0]) # num_samples
        names.append(name)

    alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
    alphas = torch.from_numpy(alphas).float()

    # alphas.float().to(config.device)
    median = weighted_average_oracle(points, alphas)
    num_oracle_calls = 1

    # logging
    obj_val = geometric_median_objective(median, points, alphas)
    logs = []
    log_entry = [0, obj_val, 0, 0]
    logs.append(log_entry)
    # start
    wv=None
    for i in range(maxiter):
        prev_median, prev_obj_val = median, obj_val
        weights = torch.tensor([alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alphas, points)],
                             dtype=alphas.dtype)
        weights = weights / weights.sum()
        median = weighted_average_oracle(points, weights)
        num_oracle_calls += 1
        obj_val = geometric_median_objective(median, points, alphas)
        log_entry = [i + 1, obj_val,
                     (prev_obj_val - obj_val) / obj_val,
                     l2dist(median, prev_median)]
        logs.append(log_entry)
        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break
        wv=copy.deepcopy(weights)
    alphas = [l2dist(median, p) for p in points]

    update_norm = 0
    for name, data in median.items():
        update_norm += torch.sum(torch.pow(data, 2))
    update_norm= math.sqrt(update_norm)

    eta = 0.1
    if max_update_norm is None or update_norm < max_update_norm:
        for name, data in target_model.state_dict().items():
            update_per_layer = median[name] * (eta)
            # if self.params['diff_privacy']:
            #     update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
            data.add_(update_per_layer)
        is_updated = True
    else:
        is_updated = False

    return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(), alphas, copy.deepcopy(target_model.state_dict())

def weighted_average_oracle(points, weights):
    """Computes weighted average of atoms with specified weights

    Args:
        points: list, whose weighted average we wish to calculate
            Each element is a list_of_np.ndarray
        weights: list of weights of the same length as atoms
    """
    tot_weights = torch.sum(weights)

    weighted_updates= dict()

    for name, data in points[0].items():
        weighted_updates[name]=  torch.zeros_like(data)
    for w, p in zip(weights, points): # 对每一个agent
        for name, data in weighted_updates.items():
            temp = (w / tot_weights).float().to(args.device)
            temp= temp* (p[name].float())
            # temp = w / tot_weights * p[name]
            if temp.dtype!=data.dtype:
                temp = temp.type_as(data)
            data.add_(temp)

    return weighted_updates

def l2dist(p1, p2):
    """L2 distance between p1, p2, each of which is a list of nd-arrays"""
    squared_sum = 0
    for name, data in p1.items():
        squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
    return math.sqrt(squared_sum)

def geometric_median_objective(median, points, alphas):
    """Compute geometric median objective."""
    temp_sum= 0
    for alpha, p in zip(alphas, points):
        temp_sum += alpha * l2dist(median, p)
    return temp_sum

    # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

def model_dist_norm_var(model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var= sum_var.to(args.device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

def foolsgold_update(target_model, updates):
    client_grads = []
    alphas = []
    names = []
    eta = 0.1
    for name, data in updates.items():
        client_grads.append(data[1])  # gradient
        alphas.append(data[0])  # num_samples
        names.append(name)

    target_model.train()
    # train and update
    optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1,
                                momentum=0.9,
                                weight_decay=0.0005)

    optimizer.zero_grad()
    fg = FoolsGold(use_memory=True)
    agg_grads, wv,alpha = fg.aggregate_gradients(client_grads,names)
    for i, (name, params) in enumerate(target_model.named_parameters()):
        agg_grads[i]=agg_grads[i] * eta
        if params.requires_grad:
            params.grad = agg_grads[i].to(args.device)
    optimizer.step()
    wv=wv.tolist()
    return True, names, wv, alpha, copy.deepcopy(target_model.state_dict())

class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def foolsgold(self,grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = 1
        self.n_params = n_params
        self.cum_net_mov = 0


    def aggregate_updates(self, global_model, agent_updates_dict):
        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)

        aggregated_updates = self.agg_avg(agent_updates_dict)
        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_parameters(new_global_params, global_model.parameters())

        return global_model.state_dict()

    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

def gaussian_noise_ls(data_shape, s, sigma, device=None):
    """
    Gaussian noise for DP-FedAVG0-LS Algorithm
    """
    # print("sigma:", sigma * s)
    return torch.normal(0, sigma * s, data_shape).long().to(device)