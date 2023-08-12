#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 7/31/2023 1:47 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
trainer.py: Description of trainer class.
"""

import torch

from .base import BaseTrainer
from utils.util import inf_loop
from logger.logger import BatchMetrics


class Trainer(BaseTrainer):
    """
    Trainer class.
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader=None,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        # Initialize.
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        # Epoch-based training or iteration-based training.
        if len_epoch is None:
            # Epoch-based training.
            self.len_epoch = len(self.data_loader)
        else:
            # Iteration-based training.
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        # Set batch metric for train and valid data sets.
        self.train_metrics = BatchMetrics("loss", *[m.__name__ for m in self.metric_ftns], postfix="/train",
                                          writer=self.writer)
        self.valid_metrics = BatchMetrics("loss", *[m.__name__ for m in self.metric_ftns], postfix="/valid",
                                          writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.
        Args:
            epoch: current epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # Get the inputs.
            data, target = data.to(self.device), target.to(self.device)

            # Zero the parameter gradients.
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())

            if batch_idx % self.log_step == 0:
                for met in self.metric_ftns:
                    metric = met(output, target)
                    self.train_metrics.update(met.__name__, metric)
                # The loss here is not the average loss.
                self.logger.info(f"Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.6f}")

            # For two different training strategies.
            # For epoch-based training, it does not happen.
            # For iteration-based training, it happened depends on len_epoch.
            if batch_idx == self.len_epoch:
                break

        # Return train average results.
        log = self.train_metrics.result()

        # Validate model.
        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Add result metrics on entire epoch to tensorboard.
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k + "/epoch", v)

        # Return all average results.
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch.
        Args:
            epoch: integer, current training epoch.
        Returns:
            A log that contains information about validation.
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # Add histogram of model parameters to the tensorboard.
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        """
        Return process.
        """
        base = "[{}/{} ({:.0f}%)]"
        try:
            # Epoch-based training.
            total = len(self.data_loader.dataset)
            current = batch_idx * self.data_loader.batch_size
        except AttributeError:
            # Iteration-based training.
            total = self.len_epoch
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)
