import getpass
import os
import re
import signal as signal_module
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import psutil

from areal.api.cli_args import (
    ClusterSpecConfig,
    LauncherConfig,
    RecoverConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.api.io_struct import AllocationMode, AllocationType
from areal.utils.launcher import get_env_vars
from areal.utils.network import find_free_ports, gethostip
from areal.utils.recover import check_if_recover
from realhf.base import gpu_utils, logging, name_resolve, names
from realhf.scheduler.client import JobException, JobInfo, JobState

logger = logging.getLogger("Local Scheduler")
JOB_STATE_TO_PROCESS_STATUS = {
    JobState.NOT_FOUND: [],
    JobState.PENDING: [psutil.STATUS_PARKED],
    JobState.RUNNING: [
        psutil.STATUS_RUNNING,
        psutil.STATUS_SLEEPING,
        psutil.STATUS_DISK_SLEEP,
        psutil.STATUS_TRACING_STOP,
        psutil.STATUS_WAKING,
        psutil.STATUS_WAITING,
        psutil.STATUS_LOCKED,
        psutil.STATUS_IDLE,
    ],
    JobState.COMPLETED: [
        psutil.STATUS_DEAD,
        psutil.STATUS_STOPPED,
        psutil.STATUS_ZOMBIE,
    ],
    JobState.FAILED: [],
    JobState.CANCELLED: [],
}
RECOVER_TIME_INTERVAL = 10  # seconds

PROCESS_STATUS_TO_JOB_STATE = {}
for job_state, process_statuses in JOB_STATE_TO_PROCESS_STATUS.items():
    for process_status in process_statuses:
        PROCESS_STATUS_TO_JOB_STATE[process_status] = job_state


def terminate_process_and_children(pid: int, signal: Optional[Union[str, int]] = None):
    if signal is None:
        signal = signal_module.SIGKILL
    if isinstance(signal, str):
        signal = getattr(signal_module, signal)
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            terminate_process_and_children(child.pid)
        parent.send_signal(signal)
    except psutil.NoSuchProcess:
        pass


class LocalLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        self._jobs: Dict[str, subprocess.Popen] = {}
        self._job_counter: Dict[str, int] = defaultdict(int)
        self._job_states = {}

        self._gpu_counter = 0
        self._cuda_devices: List[str] = os.environ.get(
            "CUDA_VISIBLE_DEVICES", ",".join(map(str, range(gpu_utils.gpu_count())))
        ).split(",")
        if len(self._cuda_devices) < 1:
            raise RuntimeError(
                f"Local mode can only run when there is at least one GPU. "
                f"CUDA_VISIBLE_DEVICES is currently set to {os.environ['CUDA_VISIBLE_DEVICES']}."
            )

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")

    def __del__(self):
        self.wait()

    def submit_array(
        self,
        job_name: str,
        cmd: str | List[str],
        count: int = 1,
        gpu: int = 0,
        env_vars: Optional[Dict] = None,
    ):
        if env_vars is None:
            env_vars = {}
        if not isinstance(cmd, list):
            cmd = [cmd] * count
        offset = self._job_counter[job_name]
        for i in range(count):
            if gpu > 0:
                # Allocate GPUs in a round-robin manner
                visible_devices = []
                for _ in range(gpu):
                    available_device_id = self._gpu_counter % len(self._cuda_devices)
                    self._gpu_counter += 1
                    visible_devices.append(available_device_id)
                env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(self._cuda_devices[j]) for j in visible_devices
                )
            c = (
                " ".join(str(k) + "=" + str(v) for k, v in env_vars.items())
                + " stdbuf -oL "
                + cmd[i]
            )
            c = f"{c} 2>&1 | tee -a {self.log_path_of(job_name)}"
            logger.info("Starting local process with command: %s", c)
            process = subprocess.Popen(c, shell=isinstance(c, str))
            self._jobs[f"{job_name}/{offset + i}"] = process
            self._job_counter[job_name] += 1

    def submit(
        self,
        job_name: str,
        cmd: str | List[str],
        gpu: int = 0,
        env_vars: Optional[Dict] = None,
    ):
        self.submit_array(job_name=job_name, cmd=cmd, gpu=gpu, env_vars=env_vars)

    def stop(self, job_name, signal=None):
        assert any(k.startswith(job_name) for k in self._jobs)
        keys = [k for k, p in self._jobs.items() if k.startswith(job_name)]
        procs = [p for k, p in self._jobs.items() if k.startswith(job_name)]
        logger.info(
            f"Stopping local process with signal {signal if signal else 'SIGKILL'}, "
            f"pid: {[p.pid for p in procs]}"
        )
        for p in procs:
            terminate_process_and_children(p.pid, signal=signal)
        for p in procs:
            p.wait()
        for k, p in zip(keys, procs):
            self._jobs.pop(k)
            del p

    def stop_all(self, signal=None):
        # signal argument is ignored in local stop_all
        for name in self._job_counter:
            self.stop(name, signal=signal)

    def find(self, job_name):
        if job_name in self._jobs:
            return JobInfo(name=job_name, state=JobState.RUNNING, host="localhost")
        else:
            return JobInfo(name=job_name, state=JobState.NOT_FOUND)

    def find_all(self, job_name_regex=".*"):
        rs = []
        for name in self._jobs:
            if re.fullmatch(job_name_regex, name):
                rs.append(self.find(name))
        return rs

    def wait(
        self,
        timeout=None,
        check_status: Tuple[JobState, ...] = (
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ),
        remove_status: Tuple[JobState, ...] = (JobState.COMPLETED,),
        update=False,
    ):
        deadline = None if timeout is None else time.time() + timeout
        logger.info(
            "Waiting for %d local running processes, pids: %s",
            len(self._jobs),
            " ".join(str(job.pid) for job in self._jobs.values()),
        )
        left = set(self._jobs.keys())
        num_jobs_left = len(left)

        while len(left) > 0:
            to_remove = []
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.run_name}: {', '.join(sorted(left))}"
                )
            # update job states
            for job_name in list(left):
                job = self._jobs[job_name]
                pid = job.pid
                process = psutil.Process(pid)
                self._job_states[job_name] = PROCESS_STATUS_TO_JOB_STATE.get(
                    process.status(), JobState.NOT_FOUND
                )

            for job_name in list(left):
                state = self._job_states[job_name]
                if state in check_status:
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=job_name.split("/")[0],
                        host="local",
                        reason=state,
                    )
                if state in remove_status:
                    logger.info(f"Job {job_name} is {state}.(Removed)")
                    left.remove(job_name)
                    to_remove.append(job_name)

            if update:
                for k in to_remove:
                    self._jobs.pop(k)
                    worker_type = k.split("/")[0]
                    assert worker_type in self._job_counter
                    self._job_counter[worker_type] -= 1
                    if self._job_counter[worker_type] <= 0:
                        self._job_counter.pop(worker_type)

            time.sleep(2)


def main():
    config, _ = parse_cli_args(sys.argv[2:])
    local_main(config, run_id=0)


def local_main(config, run_id: int = 0):
    config.launcher = to_structured_cfg(config.launcher, LauncherConfig)
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    launcher = LocalLauncher(
        config.experiment_name, config.trial_name, config.cluster.fileroot
    )

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )
    alloc_mode = AllocationMode.from_str(config.allocation_mode)

    logger.info(
        f"LocalLauncher: experiment_name={config.experiment_name}, "
        f"trial_name={config.trial_name}, fileroot={config.cluster.fileroot}, "
        f"run_id={run_id}, is_recover_run={is_recover_run}"
    )

    server_cmd = []
    server_addrs = []
    if alloc_mode.gen_backend == "sglang":
        base_seed = config.sglang.random_seed
        config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
        ports = find_free_ports(alloc_mode.gen_dp_size * 2, port_range=(10000, 50000))
        host_ip = gethostip()
        host = "localhost" if not config.sglang.enable_metrics else host_ip
        for i in range(alloc_mode.gen_dp_size):
            config.sglang.random_seed = base_seed + i
            cmd = SGLangConfig.build_cmd(
                config.sglang,
                host=host,
                tp_size=alloc_mode.gen_tp_size,
                base_gpu_id=0,
                port=ports[i * 2],
                dist_init_addr=f"localhost:{ports[i*2+1]}",
            )
            server_cmd.append(cmd)
            server_addrs.append(f"{host}:{ports[i * 2]}")

        # Launch inference servers.
        launcher.submit_array(
            job_name="llm_server",
            cmd=server_cmd,
            count=alloc_mode.gen_dp_size,
            gpu=alloc_mode.gen_pp_size * alloc_mode.gen_tp_size,
            env_vars=get_env_vars(
                config.cluster.cluster_name,
                config.launcher.inference_server_env_vars,
            ),
        )
        logger.info(
            f"LLM inference server launched at: AREAL_LLM_SERVER_ADDRS={','.join(server_addrs)}"
        )

    # Launch trainer entrypoint
    if alloc_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        if alloc_mode.type_ == AllocationType.DECOUPLED_EVAL:
            gpu = 0
            nprocs = 1
        else:
            gpu = nprocs = alloc_mode.train_world_size
        launcher.submit(
            job_name="trainer",
            cmd=f"torchrun --nnodes 1 --nproc-per-node {nprocs} --master-addr localhost --master-port {find_free_ports(1, (10000, 50000))[0]} {' '.join(sys.argv[1:])}",
            gpu=gpu,
            env_vars=dict(
                **get_env_vars(
                    config.cluster.cluster_name,
                    config.launcher.trainer_env_vars,
                ),
                AREAL_LLM_SERVER_ADDRS=",".join(server_addrs),
                AREAL_RECOVER_RUN=str(int(is_recover_run)),
            ),
        )

    try:
        launcher.wait(
            check_status=(
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.NOT_FOUND,
                JobState.COMPLETED,
            ),
            remove_status=(),
        )
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        launcher.stop_all("SIGTERM")
        # NOTE: For local launcher, We cannot distinguish between a completed job and a failed job.
        # So we will always try to recover the job if it is finished or failed.
        recover_states = [JobState.FAILED, JobState.NOT_FOUND, JobState.COMPLETED]
        if isinstance(e, JobException):
            recover_this = (
                e.reason in recover_states
                and run_id < config.recover.retries
                and config.recover.mode in ["auto", "fault"]
            )
            if recover_this:
                time.sleep(RECOVER_TIME_INTERVAL)
                local_main(config, run_id=run_id + 1)
            else:
                raise e
        else:
            raise e


if __name__ == "__main__":
    main()
