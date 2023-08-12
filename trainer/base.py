#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 7/31/2023 1:47 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
base.py: Description of base trainer class.
"""

import os
import signal
import torch

from abc import abstractmethod, ABCMeta
from pathlib import Path
from shutil import copyfile
from numpy import inf

from utils.util import write_conf, get_logger
from logger.logger import TensorboardWriter, EpochMetrics


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers.
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.logger = get_logger(config, name="trainer", state="train")

        # Initialize.
        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.config = config

        cfg_trainer = config.trainer
        self.epochs = cfg_trainer.epochs
        self.log_step = cfg_trainer.logging_step

        # Setup metric monitoring for monitoring model performance and saving best-checkpoint.
        self.monitor = cfg_trainer.get("monitor", "off")
        metric_names = ["loss"] + [met.__name__ for met in self.metric_ftns]
        self.ep_metrics = EpochMetrics(config, metric_names, phases=("train", "valid"), monitoring=self.monitor)

        # Train epoch.
        self.start_epoch = 1
        self.checkpt_top_k = cfg_trainer.get("save_topk", -1)
        self.early_stop = cfg_trainer.get("early_stop", inf)

        # Save the final config.
        write_conf(self.config, "config.yaml")

        # Write to tensorboard.
        log_dir = Path(self.config.log_tensor_dir)
        if not log_dir.exists():
            log_dir.mkdir()
        self.writer = TensorboardWriter(config, log_dir, cfg_trainer.tensorboard)

        # Create ckpt directory.
        self.checkpt_dir = Path(self.config.save_ckpt_dir)
        path_latest = self.checkpt_dir / "latest"
        path_best = self.checkpt_dir / "best"

        if not self.checkpt_dir.exists():
            self.checkpt_dir.mkdir()
        if not path_latest.exists():
            path_latest.mkdir()
        if not path_best.exists():
            path_best.mkdir()

        # Whether resume model from checkpoint.
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.
        Args:
            epoch: current epoch.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            # Get one epoch results.
            result = self._train_epoch(epoch)

            # Update epoch metric.
            self.ep_metrics.update(epoch, result)

            # Print result metrics of this epoch.
            max_line_width = max(len(line) for line in str(self.ep_metrics).splitlines())

            self.logger.info('=' * max_line_width)
            self.logger.info('\n' + str(self.ep_metrics.latest()))
            self.logger.info('=' * max_line_width)

            # Check if model performance improved or not, for early stopping and topk saving.
            is_best = False
            improved = self.ep_metrics.is_improved()
            if improved:
                not_improved_count = 0
                is_best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance did not improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                # For linux system.
                os.kill(os.getppid(), signal.SIGTERM)

            # Save checkpoints.
            using_topk_save = self.checkpt_top_k > 0
            self._save_checkpoint(epoch, save_best=is_best, save_latest=using_topk_save)

            # Keep top-k checkpoints only, using monitoring metrics such as loss/valid.
            if using_topk_save:
                self.ep_metrics.keep_topk_checkpt(self.checkpt_dir, self.checkpt_top_k)

            # Save results.
            self.ep_metrics.to_csv("epoch-results.csv")

            self.logger.info('*' * max_line_width)

    def _save_checkpoint(self, epoch, save_best=False, save_latest=True):
        """
        Saving checkpoints.
        Args:
            epoch: current epoch number.
            save_best: save a copy of current checkpoint file as "model_best.pth".
            save_latest: save a copy of current checkpoint file as "model_latest.pth".
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch_metrics": self.ep_metrics,
            "config": self.config
        }

        # Save state.
        filename = str(self.checkpt_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        self.logger.info("Model checkpoint saved at: {}\{}".format(self.config.cwd, filename))

        if save_latest:
            latest_path = str(self.checkpt_dir / "latest" / "model_latest.pth")
            copyfile(filename, latest_path)
        if save_best:
            best_path = str(self.checkpt_dir / "best" / "model_best.pth")
            copyfile(filename, best_path)
            self.logger.info("Renewing best checkpoint: .\{}".format(best_path))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints.
        Args:
            resume_path: absolute checkpoint path to be resumed.
        """
        # Load start.
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        # Load checkpoints.
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1

        # Epoch metrics.
        self.ep_metrics = checkpoint["epoch_metrics"]

        # Load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["_target_"] != self.config["optimizer"]["_target_"]:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load end.
        self.logger.info("Checkpoint loaded.")
        self.logger.info("Resume training from epoch {}.".format(self.start_epoch))
