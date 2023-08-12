#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 8/8/2023 12:54 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
evaluate.py: Description of evaluating.
"""

import os
import hydra
import torch

from pathlib import Path
from omegaconf import OmegaConf
from torchinfo import summary
from tqdm import tqdm

from utils.util import instantiate, get_logger


def eval_worker(config):
    logger = get_logger("train", state="evaluate")

    max_len = 70

    # Log start.
    logger.info("=" * max_len)
    logger.info("Start evaluating...")
    logger.info("@CopyRight Borg")
    logger.info("=" * max_len)

    logger.info("Loading checkpoint: {} ...".format(config.checkpoint))

    # Load checkpoint.
    checkpoint = torch.load(config.checkpoint)
    loaded_config = OmegaConf.create(checkpoint["config"])

    # Setup data_loader instances.
    data_loader = instantiate(config.data_loader)

    # Restore network architecture.
    model = instantiate(loaded_config.arch)
    summary(model)

    # Load trained weights.
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    # Instantiate loss and metrics.
    criterion = instantiate(loaded_config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]

    # Prepare model for testing.
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metrics):
                total_metrics[j] += metric(output, target) * batch_size

    # Equal to len(data_loader.dataset)
    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)})
    logger.info(log)

    # Log end.
    logger.info("=" * max_len)
    logger.info("Evaluating is over.")
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
    eval_worker(config)


@hydra.main(version_base=None, config_path="config/", config_name="evaluate")
def main(config):
    # Decide device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    # Set number of num_works for Dataloader.
    config.cpus = 6

    # Set working dir.
    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))

    config = OmegaConf.to_yaml(config, resolve=True)

    # Run process.
    init_worker(config, working_dir)


if __name__ == '__main__':
    main()
