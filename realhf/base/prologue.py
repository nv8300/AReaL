# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import getpass
import sys
import datetime
import os
import json
from omegaconf import DictConfig, OmegaConf

PRESET_FLAG_NAME = "--config"
PRESET_FLAG_VAR_NAME = "config"
PRESET_EXTERNAL_CONFIG_NAME = "external_configs"

def global_init():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(PRESET_FLAG_NAME)
    args = vars(parser.parse_known_args()[0])
    if args[PRESET_FLAG_VAR_NAME] is None:
        return
    preset_path = args[PRESET_FLAG_VAR_NAME]

    config = OmegaConf.load(preset_path)
    external_configs = config.get(PRESET_EXTERNAL_CONFIG_NAME)

    if external_configs is None:
        return

    # add externel envs.
    if external_configs.envs is not None:
        for key, value in external_configs.envs.items():
            if key not in os.environ:
                os.environ[key] = value

    # resolve config path for cluster spec.
    cluster_spec_path = os.environ.get("CLUSTER_SPEC_PATH", "")
    if cluster_spec_path == "":
        if external_configs.cluster_config is not None:
            fileroot = external_configs.cluster_config.fileroot
            if fileroot is not None and fileroot != "":
                experiment_name = get_experiment_name(config.get("experiment_name"))
                trial_name = get_trial_name(config.get("trial_name"))
                config_dir = f"{fileroot}/configs/{getpass.getuser()}/{experiment_name}/{trial_name}"
                os.makedirs(config_dir, exist_ok=True)
                cluster_spec_path = f"{config_dir}/cluster_config.json"
                cluster_spec = OmegaConf.to_container(external_configs.cluster_config)
                if "cluster_type" not in cluster_spec:
                    cluster_spec["cluster_type"] = config.mode
                if "cluster_name" not in cluster_spec:
                    cluster_spec["cluster_name"] = f"{config.mode}_cluster"
                with open(cluster_spec_path, "w") as f:
                    json.dump(cluster_spec, f)
                os.environ["CLUSTER_SPEC_PATH"] = cluster_spec_path

def get_experiment_name(default_name: str = ""):
    if any("experiment_name=" in x for x in sys.argv):
        experiment_name = next(x for x in sys.argv if "experiment_name=" in x).split(
            "="
        )[1]
    else:
        experiment_name = default_name
        if experiment_name  == "":
            experiment_name = f"quickstart-experiment"

    if "_" in experiment_name:
        raise RuntimeError("experiment_name should not contain `_`.")
    return experiment_name

def get_trial_name(default_name: str = ""):
    if any("trial_name=" in x for x in sys.argv):
        trial_name = next(x for x in sys.argv if "trial_name=" in x).split("=")[1]
    else:
        trial_name = default_name
        if trial_name == "":
            trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if "_" in trial_name:
        raise RuntimeError("trial_name should not contain `_`.")
    return trial_name


global_init()
