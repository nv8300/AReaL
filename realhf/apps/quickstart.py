# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import datetime
import getpass
import pathlib
import re
import sys
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from realhf.base.prologue import PROLOGUE_FLAG_NAME, PROLOGUE_FLAG_VAR_NAME, PROLOGUE_EXTERNAL_CONFIG_NAME, get_experiment_name, get_trial_name
from realhf.api.quickstart.entrypoint import QUICKSTART_FN
from realhf.base.cluster import spec as cluster_spec
from realhf.base.importing import import_module

# NOTE: Register all implemented experiments inside ReaL.
import_module(
    str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "common"),
    re.compile(r".*_exp\.py$"),
)
import realhf.experiments.benchmark.profile_exp


def main():
    parser = argparse.ArgumentParser(prog="ReaL Quickstart")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True
    for k, v in QUICKSTART_FN.items():
        subparser = subparsers.add_parser(k)
        subparser.add_argument(
            "--show-args",
            action="store_true",
            help="Show all legal CLI arguments for this experiment.",
        )
        subparser.add_argument(
            PROLOGUE_FLAG_NAME,
            type=str,
            help="Set config (*.yaml) for this experiment.",
        )
        subparser.set_defaults(func=v)
    args = vars(parser.parse_known_args()[0])
    if args["show_args"]:
        sys.argv = [sys.argv[0], "--help"]
        QUICKSTART_FN[args["cmd"]]()
        return

    experiment_name = ""
    trial_name = ""

    if args[PROLOGUE_FLAG_VAR_NAME]:
        config_dir, experiment_name, trial_name = prepare_hydra_config(args["cmd"], args[PROLOGUE_FLAG_VAR_NAME])
        sys.argv.remove(PROLOGUE_FLAG_NAME)
        sys.argv.remove(args[PROLOGUE_FLAG_VAR_NAME])
        sys.argv += [f"--config-path", f"{config_dir}"]
    else:
        experiment_name = get_experiment_name()
        trial_name = get_trial_name()

    launch_hydra_task(args["cmd"], experiment_name, trial_name, QUICKSTART_FN[args["cmd"]])


def prepare_hydra_config(name: str, PROLOGUE_path: str):
    config = OmegaConf.load(PROLOGUE_path)
    experiment_name = get_experiment_name(config.get("experiment_name"))
    trial_name = get_trial_name(config.get("trial_name"))
    config_dir = f"{cluster_spec.fileroot}/configs/{getpass.getuser()}/{experiment_name}/{trial_name}"

    config.pop(PROLOGUE_EXTERNAL_CONFIG_NAME)
    with open(f"{config_dir}/{name}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    return (config_dir, experiment_name, trial_name)

def launch_hydra_task(name: str, experiment_name: str, trial_name: str, func: hydra.TaskFunction):
    # Disable hydra logging.
    if not any("hydra/job_logging=disabled" in x for x in sys.argv):
        sys.argv += ["hydra/job_logging=disabled"]

    if (
        "--multirun" in sys.argv
        or "hydra.mode=MULTIRUN" in sys.argv
        or "-m" in sys.argv
    ):
        raise NotImplementedError("Hydra multi-run is not supported.")

    # non-multirun mode, add hydra run dir
    sys.argv += [
        f"hydra.run.dir={cluster_spec.fileroot}/logs/{getpass.getuser()}/"
        f"{experiment_name}/{trial_name}/hydra-outputs/"
    ]

    sys.argv.pop(1)

    func()


if __name__ == "__main__":
    main()
