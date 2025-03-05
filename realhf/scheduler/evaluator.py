import json
import os
import re
import subprocess
import time
from typing import Dict

import wandb

import realhf.api.core.system_api as config_pkg
from realhf.base import cluster, constants, logging

logger = logging.getLogger("AutomaticEvaluator", "colored")


class AutomaticEvaluator:

    def __init__(
        self,
        config: config_pkg.AutomaticEvaluator,
        wandb_config: config_pkg.WandBConfig,
    ):
        self.__running_processes: Dict[int, subprocess.Popen] = {}
        self.__start_time = {}
        self.__done_steps = []
        self.__wandb_log_steps = []
        self.__pending_ckpts = {}
        self.__image = config.eval_job_image or cluster.spec.gpu_image
        self.__data_names = config.data_names
        self.__max_gen_tokens = config.max_gen_tokens
        self.__max_concurrent_jobs = config.max_concurrent_jobs
        self.__prompt_type = config.prompt_type
        self.__wandb_config = wandb_config
        self.__config = config
        self.__wandb_inited = False

        # Check evaluated checkpoints by logs
        former_output_dir = os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "eval_output",
        )
        if os.path.exists(former_output_dir):
            for log_dir in os.listdir(former_output_dir):
                match = re.match(r"globalstep(\d+)", log_dir)
                if not match:
                    continue
                global_step = int(match.groups()[0])
                self.__done_steps.append(global_step)

        logger.info(
            f"Initializing AutomaticEvaluator: \n"
            f"eval_job_image: {config.eval_job_image}\n"
            f"data_names: {config.data_names}\n"
            f"max_gen_tokens: {config.max_gen_tokens}\n"
            f"max_concurrent_jobs: {config.max_concurrent_jobs}\n"
            f"Existing eval outputs for global steps: {self.__done_steps}"
        )
        if self.__config.initial_checkpoint_path and 0 not in self.__done_steps:
            self.__pending_ckpts[0] = self.__config.initial_checkpoint_path

        if not cluster.spec.cluster_type == "slurm":
            raise NotImplementedError(
                "Currently only support automatic evaluation for slurm"
            )

    def __lazy_wandb_init(self):
        # Initializing wandb for evaluator
        wandb.login()
        wandb.init(
            mode=self.__wandb_config.mode,
            entity=self.__wandb_config.entity,
            project=self.__wandb_config.project or constants.experiment_name(),
            name=self.__wandb_config.name or f"{constants.trial_name()}_eval",
            job_type=self.__wandb_config.job_type,
            group=self.__wandb_config.group
            or f"{constants.experiment_name()}_{constants.trial_name()}",
            notes=self.__wandb_config.notes,
            tags=self.__wandb_config.tags,
            config=self.__wandb_config.config,
            dir=os.path.join(
                constants.LOG_ROOT, constants.experiment_name(), constants.trial_name()
            ),
            force=True,
            id=f"{constants.experiment_name()}_{constants.trial_name()}_eval",
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )

    def __check_new_ckpts(self):
        save_path = os.path.join(
            constants.MODEL_SAVE_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "actor",
        )
        if not os.path.exists(save_path):
            return
        for ckpt_dir in os.listdir(save_path):
            match = re.match(r"epoch(\d+)epochstep(\d+)globalstep(\d+)", ckpt_dir)
            if not match:
                continue
            _, _, global_step = map(int, match.groups())
            if not global_step in (
                list(self.__running_processes.keys())
                + list(self.__pending_ckpts.keys())
                + self.__done_steps
            ):
                abs_ckpt_dir = os.path.join(save_path, ckpt_dir)
                logger.info(
                    f"Found new checkpoint (globalstep{global_step}) at {abs_ckpt_dir}"
                )
                self.__pending_ckpts[global_step] = os.path.join(
                    save_path, abs_ckpt_dir
                )

    def __check_and_maybe_submit_jobs(self):
        for global_step, process in self.__running_processes.items():
            result = process.poll()
            if not result is None:
                self.__done_steps.append(global_step)
                start_time = self.__start_time[global_step]
                logger.info(
                    f"Evaluation of checkpoint (globalstep{global_step}) is done, returncode={process.returncode}, "
                    f"time passed {time.perf_counter() - start_time:.3f} s."
                )

        for done in self.__done_steps:
            if done in self.__running_processes:
                self.__running_processes.pop(done)
                self.__start_time.pop(done)

        submitted = []
        # Jobs should be submitted by the order of global steps
        ordered_steps = sorted(self.__pending_ckpts.keys())
        for global_step in ordered_steps:
            ckpt_path = self.__pending_ckpts[global_step]
            if len(self.__running_processes) >= self.__max_concurrent_jobs:
                return
            self.__submit_one(global_step, ckpt_path)
            submitted.append(global_step)

        for global_step in submitted:
            self.__pending_ckpts.pop(global_step)

    def __submit_one(self, global_step, ckpt_path):
        output_path = self.eval_output_path(global_step)
        os.makedirs(output_path, exist_ok=True)
        log_file = open(os.path.join(output_path, "output.log"), "w")
        if cluster.spec.cluster_type == "slurm":
            cmd = self.slurm_eval_cmd(global_step, ckpt_path)
        else:
            raise NotImplementedError(
                "AutomaticEvaluator does only support slurm cluster."
            )

        logger.info(
            f"Submitting evaluation job of checkpoint at {ckpt_path} (globalstep{global_step}), "
            f"command: {cmd}"
        )
        self.__running_processes[global_step] = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            shell=True,
        )
        self.__start_time[global_step] = time.perf_counter()

    def __maybe_parse_and_log_to_wandb(self):
        # Note that after recover, all previous done steps will be
        # logged again in case some data points are missing.
        # If the data point is already logged, wandb will raise
        # a warning.
        to_log = list(
            filter(
                lambda x: x not in self.__wandb_log_steps,
                (
                    self.__done_steps
                    + list(self.__running_processes.keys())
                    + list(self.__pending_ckpts.keys())
                ),
            )
        )
        while to_log:
            # The wandb should always log the minimal global step
            # whose eval job has been submitted but not logged.
            # If this minimal step is not logged, other steps should wait.
            global_step = min(to_log)
            result_path = os.path.join(
                self.eval_output_path(global_step),
                f"math_eval_{self.__max_gen_tokens}",
                f"aggregate_parallel_{self.__prompt_type}.json",
            )
            if not os.path.exists(result_path):
                break

            if not self.__wandb_inited:
                self.__lazy_wandb_init()
                self.__wandb_inited = True

            try:
                with open(result_path, "r") as fp:
                    data = json.load(fp)
            except json.JSONDecodeError:
                logger.warning(f"JSON decoding for eval result in {result_path} failed")
                continue
            except FileNotFoundError:
                logger.warning(
                    f"{result_path} not found, but the eval job is done. "
                    "Maybe the eval job abnormally exited and did not output the result."
                )
                continue
            wandb_data = {}
            for data_name, d in data.items():
                for k, v in d.items():
                    wandb_data[f"{data_name}_{k}"] = v
            wandb.log(wandb_data, step=global_step)
            logger.info(f"Logging eval result {wandb_data} to step {global_step}")
            self.__wandb_log_steps.append(global_step)
            to_log.remove(global_step)

    def step(self):
        self.__check_new_ckpts()
        self.__check_and_maybe_submit_jobs()
        self.__maybe_parse_and_log_to_wandb()

    def slurm_eval_cmd(self, global_step, ckpt_path):
        slurm_job_name = f"{constants.experiment_name()}_{constants.trial_name()}:eval_globalstep{global_step}"
        cmd = (
            f"srun --mpi=pmi2 -J {slurm_job_name} --ntasks=1 --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G "
            f"singularity exec --nv --pid --writable-tmpfs --bind /storage:/storage "
            f"{self.__image} "
            f"bash ./evaluation/sh/install_deps_and_eval.sh {ckpt_path} {self.eval_output_path(global_step)} "
            f"{self.__data_names} {self.__max_gen_tokens} {self.__prompt_type}"
        )
        return cmd

    def eval_output_path(self, global_step):
        return os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "eval_output",
            f"globalstep{global_step}",
        )
