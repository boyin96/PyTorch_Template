#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 7/31/2023 1:41 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
metric.py: Description of metric functions.
"""

import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.eq(pred, target).sum().item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    """
    Top-K Accuracy.
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.eq(pred[:, i], target).sum().item()
    return correct / len(target)
