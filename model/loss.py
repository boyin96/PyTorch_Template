#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 7/31/2023 1:41 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
loss.py: Description of loss functions.
"""

import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)
