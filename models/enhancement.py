#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import torch
import torch.nn.functional as F
from torch import nn, autograd

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingTarget(nn.Module):
    def __init__(self):
        super(LabelSmoothingTarget, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        # smooth_loss = -logprobs.mean(dim=-1)
        smooth_loss = torch.zeros(logprobs.shape[0])
        for i in range(logprobs.shape[0]):
            smooth_loss[i] = abs(1.0 - logprobs[i][9])
        smooth_loss.requires_grad_(True)
        smooth_loss = smooth_loss.to(args.device)
        loss = confidence * nll_loss + smooth_loss*smoothing*0.1
        return loss.mean()