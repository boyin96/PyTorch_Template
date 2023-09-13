#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 7/31/2023 1:56 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
logger.py: Description of logger.
"""

import os
import pandas as pd

from itertools import product
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.util import get_logger


class TensorboardWriter:
    r"""
    Class for Tensorboard writer.
    """
    def __init__(self, config, log_dir, enabled):
        self.logger = get_logger(config, name="tensorboard-writer", state="train")
        self.writer = SummaryWriter(log_dir) if enabled else None

        self.selected_module = ""
        self.step = 0
        self.tb_writer_ftns = {
            "add_scalar", "add_scalars", "add_image", "add_images", "add_audio",
            "add_text", "add_histogram", "add_pr_curve", "add_embedding"
        }
        self.timer = datetime.now()

    def set_step(self, step):
        r"""
        Add scale to tensorboard.
        Args:
            step: step
        """
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        r"""
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            attr = getattr(self.writer, name)
            return attr


class BatchMetrics:
    r"""
    Class for the metrics of a batch.
    """
    def __init__(self, *keys, postfix="", writer=None):
        self.writer = writer
        self.postfix = postfix
        if postfix:
            keys = [k + postfix for k in keys]
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        r"""
        Update DataFrame.
        Args:
            key: index of row
            value: loss/acc/or
            n: 1.
        """
        if self.postfix:
            key = key + self.postfix
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        if self.postfix:
            key = key + self.postfix
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class EpochMetrics:
    r"""
    Class for the metrics of each epoch.
    """
    def __init__(self, config, metric_names, phases=("train", "valid"), monitoring="off"):
        self.logger = get_logger(config, name="epoch-metrics", state="train")
        columns = list(product(metric_names, phases))
        self._data = pd.DataFrame(columns=columns)
        self.monitor_mode, self.monitor_metric = self._parse_monitoring_mode(monitoring)
        self.topk_idx = []

    def minimizing_metric(self, idx):
        r"""
        The key function of sorted to sort dataframe based on specific metrics such as loss/valid.
        Args:
            idx: the index of epoch such as epoch-i
        """
        if self.monitor_mode == "off":
            return 0
        try:
            metric = self._data[self.monitor_metric].loc[idx]
        except KeyError:
            self.logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = "off"
            return 0
        if self.monitor_mode == "min":
            return metric
        else:
            return - metric

    @staticmethod
    def _parse_monitoring_mode(monitor_mode):
        r"""
        Split model and metric.
        Args:
            monitor_mode: such as "min loss/valid"
        """
        if monitor_mode == "off":
            return "off", None
        else:
            monitor_mode, monitor_metric = monitor_mode.split()
            monitor_metric = tuple(monitor_metric.split('/'))
            assert monitor_mode in ["min", "max"]
        return monitor_mode, monitor_metric

    def is_improved(self):
        r"""
        Check whether metric performance is improved.
        """
        if self.monitor_mode == "off":
            return True

        # Judge latest is equal the best or not.
        last_epoch = self._data.index[-1]
        best_epoch = self.topk_idx[0]
        return last_epoch == best_epoch

    def keep_topk_checkpt(self, checkpt_dir, k=3):
        r"""
        Keep top-k checkpoints by deleting k+1'th the best epoch index from dataframe for every epoch.
        """
        if len(self.topk_idx) > k and self.monitor_mode != "off":
            last_epoch = self._data.index[-1]
            self.topk_idx = self.topk_idx[:(k + 1)]
            if last_epoch not in self.topk_idx:
                to_delete = last_epoch
            else:
                to_delete = self.topk_idx[-1]

            # delete checkpoint having out-of topk metric
            filename = str(checkpt_dir / "checkpoint-epoch{}.pth".format(to_delete.split('-')[1]))
            try:
                os.remove(filename)
            except FileNotFoundError:
                # this happens when current model is loaded from checkpoint
                # or target file is already removed somehow
                pass

    def update(self, epoch, result):
        r"""
        Save the results of each epoch.
        Args:
            epoch: epoch
            result: results
        """
        epoch_idx = "epoch-{}".format(epoch)
        self._data.loc[epoch_idx] = {tuple(k.split('/')): v for k, v in result.items()}

        # Sort the topk.
        self.topk_idx.append(epoch_idx)
        self.topk_idx = sorted(self.topk_idx, key=self.minimizing_metric)

    def latest(self):
        return self._data[-1:]

    def to_csv(self, save_path=None):
        self._data.to_csv(save_path)

    def __str__(self):
        return str(self._data)
