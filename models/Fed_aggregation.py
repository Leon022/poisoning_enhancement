#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import copy
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from models.helper import accumulate_weight, geometric_median_update, \
    init_weight_accumulator, foolsgold_update, average_shrink_models, \
    Aggregation



def aggregation(args, glob_model, agent_name_keys, epochs_submit_update_dict, num_samples_dict, agent_updates_dict):
    weight_accumulator = init_weight_accumulator(glob_model)
    weight_accumulator, updates = accumulate_weight(args, weight_accumulator, epochs_submit_update_dict,
                                                           agent_name_keys, num_samples_dict)

    if args.aggregation_methods == 'fedavg':
        is_updated, aggre_model = average_shrink_models(weight_accumulator=weight_accumulator,
                                                  target_model=glob_model,
                                                  epoch_interval=1)
        # n_model_params = len(parameters_to_vector(glob_model.parameters()))
        # aggregator = Aggregation(num_samples_dict, n_model_params, args)
        # aggre_model = aggregator.aggregate_updates(glob_model, agent_updates_dict)

    elif args.aggregation_methods == 'RFA':
        maxiter = 10
        num_oracle_calls, is_updated, names, weights, alphas, aggre_model = geometric_median_update(glob_model, updates,
                                                                                              maxiter=maxiter)
    elif args.aggregation_methods == 'foolsgold':
        is_updated, names, weights, alphas, aggre_model = foolsgold_update(glob_model, updates)

    elif args.aggregation_methods == 'RLR':
        n_model_params = len(parameters_to_vector(glob_model.parameters()))
        aggregator = Aggregation(num_samples_dict, n_model_params, args)
        aggre_model = aggregator.aggregate_updates(glob_model, agent_updates_dict)

    return aggre_model