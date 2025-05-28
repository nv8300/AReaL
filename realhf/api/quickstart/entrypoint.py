# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import datetime
import functools
import inspect
import json
import os
from typing import Callable

import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

import realhf.api.core.system_api as system_api
from realhf.base.constants import init_constants
from realhf.base.ray_utils import check_ray_availability
from realhf.base.slurm_utils import check_slurm_availability


def kind_reminder(config_name, logger, args):
    from realhf.base.constants import LOG_ROOT, MODEL_SAVE_ROOT

    logger.info(f"Running {config_name} experiment.")
    logger.info(
        f"Logs will be dumped to {os.path.join(LOG_ROOT, args.experiment_name, args.trial_name)}"
    )
    logger.info(
        f"Experiment configs will be dumped to {os.path.join(LOG_ROOT, args.experiment_name, args.trial_name, 'config.yaml')}"
    )
    logger.info(
        f"Model checkpoints will be saved to {os.path.join(MODEL_SAVE_ROOT, args.experiment_name, args.trial_name)}"
    )

    if args.mode == "slurm":
        slurm_available = check_slurm_availability()
        if slurm_available:
            logger.info("Launching experiments with SLURM...")
        else:
            logger.warning("Slurm is not available. Using local mode.")
            args.mode = "local"
    elif args.mode == "ray":
        ray_available = check_ray_availability()
        if ray_available:
            logger.info("Launching experiments with RAY...")
        else:
            logger.warning("Ray is not available. Using local mode.")
            args.mode = "local"
    elif args.mode == "local":
        logger.info("Launching experiments locally.")
    else:
        raise ValueError(f"Invalid mode {args.mode}")


cs = ConfigStore.instance()
QUICKSTART_CONFIG_CLASSES = {}
QUICKSTART_USERCODE_PATHS = {}
QUICKSTART_FN = {}


def register_quickstart_exp(config_name: str, exp_cls: Callable):
    usercode_path = os.path.abspath(inspect.getfile(inspect.currentframe().f_back))

    @hydra.main(version_base=None, config_name=config_name)
    def run(args):
        # NOTE: we import logging here to avoid hydra logging overwrite
        import realhf.base.logging as logging

        logger = logging.getLogger("quickstart", "colored")

        # print_runtime_helper(OmegaConf.to_object(args))

        exp_name = args.experiment_name
        if args.trial_name == MISSING:
            args.trial_name = trial_name = (
                f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            )
        else:
            trial_name = args.trial_name
        from realhf.apps.main import main_start, main_stop

        init_constants(args)
        from realhf.base.constants import LOG_ROOT, QUICKSTART_EXPR_CACHE_PATH

        config_save_path = os.path.join(
            LOG_ROOT, args.experiment_name, args.trial_name, "config.yaml"
        )
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, "w") as f:
            yaml.dump(
                dataclasses.asdict(OmegaConf.to_object(args)),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        kind_reminder(config_name, logger, args)

        exp_fn = functools.partial(exp_cls, **args)

        os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
        cache_file = os.path.join(
            QUICKSTART_EXPR_CACHE_PATH, f"{exp_name}_{trial_name}.json"
        )
        with open(cache_file, "w") as f:
            dict_args = OmegaConf.to_container(args)
            json.dump(
                dict(
                    args=dict_args,
                    usercode_path=usercode_path,
                    config_name=config_name,
                ),
                f,
                indent=4,
                ensure_ascii=False,
            )

        system_api.register_experiment(exp_name, exp_fn)

        try:
            main_start(args)
        except Exception as e:
            main_stop(args)
            logger.warning("Exception occurred. Stopping all workers.")
            raise e

    cs.store(name=config_name, node=exp_cls)

    # assert config_name not in QUICKSTART_CONFIG_CLASSES
    QUICKSTART_CONFIG_CLASSES[config_name] = exp_cls
    # assert config_name not in QUICKSTART_USERCODE_PATHS
    QUICKSTART_USERCODE_PATHS[config_name] = usercode_path
    # assert config_name not in QUICKSTART_FN
    QUICKSTART_FN[config_name] = run
    return run
