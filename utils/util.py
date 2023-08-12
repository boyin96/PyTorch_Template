#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time      : 8/7/2023 11:00 PM
# Author    : Bo Yin
# Email     : bo.yin@ugent.be

r"""
util.py: Description of util functions.
"""

import yaml
import hydra
import logging
import logging.config

from omegaconf import OmegaConf
from pathlib import Path
from importlib import import_module
from itertools import repeat
from functools import partial, update_wrapper


def get_logger(config, name=None, state="train"):
    """
    Generate logger object.
    """
    hydra_conf = OmegaConf.load("outputs/{}/{}/.hydra/hydra.yaml".format(state, config.time_now))
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)


def inf_loop(data_loader):
    """
    Wrapper function for endless data loader.
    """
    for loader in repeat(data_loader):
        yield from loader


def instantiate(config, *args, is_func=False, **kwargs):
    """
    Wrapper function for hydra.utils.instantiate.
    Returns:
        1. return None if config.__target__ is None.
        2. return function handle if is_func is True.
        3. return if _target_ is a class name: the instantiated object with named arguments passing in __init__.
                  if _target_ is a callable function name: execute function and return value of the call.
    """
    assert "_target_" in config, "Config should have \'_target_\' for class instantiation."
    target = config["_target_"]
    if target is None:
        return None
    if is_func:
        # Get function handle.
        modulename, funcname = target.rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # Make partial function with arguments given in config.
        kwargs.update({k: v for k, v in config.items() if k != "_target_"})
        partial_func = partial(func, *args, **kwargs)

        # Update original func such as '__name__' and '__doc__' to partial function.
        update_wrapper(partial_func, func)
        return partial_func

    return hydra.utils.instantiate(config, *args, **kwargs)


def write_yaml(content, fname):
    """
    Write yaml.
    Args:
        content: yaml.
        fname: path.
    """
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)


def write_conf(config, save_path):
    """
    Save config file.
    Args:
        config: config.yaml.
        save_path: path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)
