#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 6/6/2023 11:58 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
train.py: Description of main training.
"""

import os
import hydra
import torch
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from torchinfo import summary

from trainer.trainer import Trainer
from utils.util import instantiate, get_logger

# Fix random seeds for reproducibility.
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train_worker(config):
    logger = get_logger("train")

    max_len = 70

    # Log start.
    logger.info("=" * max_len)
    logger.info("Start training...")
    logger.info("@CopyRight Borg")
    logger.info("=" * max_len)

    # Setup data_loader instances.
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # Setup model instances.
    model = instantiate(config.arch)

    # Print the information of model.
    print("The details of model: ")
    summary(model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info("Trainable parameters: {}".format(sum([p.numel() for p in trainable_params])))

    # Setup metric and loss functions instances.
    criterion = instantiate(config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in config.metrics]

    # Setup optimizer instances.
    optimizer = instantiate(config.optimizer, model.parameters())

    # Setup lr_scheduler instances.
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    # Train process.
    trainer = Trainer(model=model,
                      criterion=criterion,
                      metric_ftns=metrics,
                      optimizer=optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

    # Log end.
    logger.info("=" * max_len)
    logger.info("Training is over.")
    logger.info("@CopyRight Borg")
    logger.info("=" * max_len)


def init_worker(config, working_dir):
    # Initialize config.
    config = OmegaConf.create(config)
    config.cwd = working_dir

    # For hydra errors.
    os.environ['HYDRA_FULL_ERROR'] = '1'

    # Prevent access to non-existing keys.
    OmegaConf.set_struct(config, True)

    # Start training process.
    train_worker(config)


@hydra.main(version_base=None, config_path="config/", config_name="train")
def main(config):
    # Decide device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    # Set number of num_works for Dataloader.
    config.cpus = 6

    # Set working dir.
    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))

    # Set absolute checkpoint path.
    if config.resume is not None:
        config.resume = hydra.utils.to_absolute_path(config.resume)
    config = OmegaConf.to_yaml(config, resolve=True)

    # Run process.
    init_worker(config, working_dir)


if __name__ == '__main__':
    main()
