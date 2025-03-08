# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import os
from typing import Dict, List, Optional, Tuple, Union

import realhf.api.core.dfg as dfg
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelAbstraction,
    ModelName,
    ModelShardID,
    StandaloneModelShardAbstraction,
)
from realhf.base import constants, topology
from realhf.base.cluster import spec as cluster_spec

_LLM_GPU_IMAGE = cluster_spec.gpu_image
_LLM_CPU_IMAGE = cluster_spec.cpu_image


@dataclasses.dataclass
class Scheduling:
    cpu: int
    gpu: int
    mem: int
    gpu_type: str = "tesla"
    node_type: str = None
    nodelist: str = None
    exclude: str = None
    container_image: str = _LLM_CPU_IMAGE
    env_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format

    @staticmethod
    def master_worker_default(**kwargs):
        return Scheduling(
            **{
                "cpu": 16,
                "mem": 20 * 1024,
                "gpu": 0,
                "container_image": _LLM_CPU_IMAGE,
                **kwargs,
            }
        )

    @staticmethod
    def model_worker_default(**kwargs):
        return Scheduling(
            **{
                "cpu": 2,
                "gpu": 1,
                "mem": 60 * 1024,
                "container_image": _LLM_GPU_IMAGE,
                **kwargs,
            }
        )


@dataclasses.dataclass
class WorkerInformation:
    """The basic information of an worker.

    To improve config readability, the experiment starter will fill the
    fields, instead of letting the users do so in experiment configs.
    """

    experiment_name: str = ""
    trial_name: str = ""  # Name of the trial of the experiment; e.g. "{USER}-0".
    worker_type: str = ""  # E.g. "policy", "actor", or "trainer".
    worker_index: int = (
        -1
    )  # The index of the worker of the specific type, starting from 0.
    worker_count: int = (
        0  # Total number of workers; hence, 0 <= worker_index < worker_count.
    )
    worker_tag: Optional[str] = (
        None  # For actor and policy worker, can be "training" or "evaluation".
    )
    host_key: Optional[str] = None  # Worker will update and keep this key alive.
    watch_keys: Union[str, List[str]] = (
        None  # Worker will exit if all of the watching keys are gone.
    )

    def system_setup(
        self,
        experiment_name,
        trial_name,
        worker_type,
        worker_index,
        worker_count,
    ):
        """Setup system related worker information, while leaving the rest
        untouched."""
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        self.worker_count = worker_count


@dataclasses.dataclass
class ModelWorker:
    base_seed: int
    shards: List[StandaloneModelShardAbstraction]
    # dataset, for source model workers
    tokenizer_name_or_path: Optional[str] = None
    datasets: Optional[List[Union[str, DatasetAbstraction]]] = None
    use_dataset_cache: bool = False
    dataset_cahce_root: str = constants.DATASET_CACHE_PATH
    cuda_cache_cleanliness: bool = True
    cuda_cache_clear_freq: int = 10
    torch_cache_mysophobia: bool = False
    # model_topos and worker_info will be configured automatically
    model_rpcs: List[dfg.MFCDef] = None
    model_topos: Dict[ModelName, topology.PipeModelDataParallelTopology] = None
    msid2mwid: Dict[ModelShardID, int] = None
    data_transfer_pairs: List[Tuple[str, str]] = None
    sync_param_pairs: List[Tuple[str, str]] = None
    # profiling
    profile_mode: bool = False
    worker_info: Optional[WorkerInformation] = None

    def __post_init__(self):
        model_names = [s.id.model_name for s in self.shards]
        if len(set(model_names)) != len(model_names):
            raise ValueError(
                f"ModelWorker cannot have multiple shards of the same model name: {model_names}."
            )


@dataclasses.dataclass
class ExperimentSaveEvalControl:
    """Utility object for controlling the frequency of saving and evaluation
    during training.

    ``Epoch`` refers to the number of times the training loop iterates over the entire dataset.
    ``Step`` refers to the number of iterations running the algorithm dataflow.

    This object manages independent counters for epochs, steps, and seconds. The model will
    be saved or evaluated when any of the following conditions are met.

    :param total_train_epochs: The total number of epochs to train the model.
    :type total_train_epochs: int
    :param save_freq_epochs: Frequency in epochs at which to save the model. If None,
        the model will not be saved based on epoch changes during training.
    :type save_freq_epochs: Optional[int]
    :param save_freq_steps: Frequency in steps at which to save the model. If None,
        the model will not be saved based on step changes during training.
    :type save_freq_steps: Optional[int]
    :param save_freq_secs: Frequency in seconds at which to save the model. If None,
        the model will not be saved based on time changes during training.
    :type save_freq_secs: Optional[int]
    :param ckpt_freq_epochs: Frequency in epochs at which to save the model for recover.
        The preivous checkpoint will be overwritten to reduce disk usage. If None, use save_freq_epochs.
    :type ckpt_freq_epochs: Optional[int]
    :param ckpt_freq_steps: Frequency in steps at which to save the model for recover. If None,
        the model will not be saved based on step changes during training.
    :type ckpt_freq_steps: Optional[int]
    :param ckpt_freq_secs: Frequency in seconds at which to save the model for recover. If None,
        the model will not be saved based on time changes during training.
    :type ckpt_freq_secs: Optional[int]
    :param eval_freq_epochs: Frequency in epochs at which to evaluate the model. If None,
        the model will not be evaluated based on epoch changes during training.
    :type eval_freq_epochs: Optional[int]
    :param eval_freq_steps: Frequency in steps at which to evaluate the model. If None,
        the model will not be evaluated based on step changes during training.
    :type eval_freq_steps: Optional[int]
    :param eval_freq_secs: Frequency in seconds at which to evaluate the model. If None,
        the model will not be evaluated based on time changes during training.
    :type eval_freq_secs: Optional[int]
    :param benchmark_steps: Terminate training after this number of steps. Used for system
        benchmarking only. Set to None for normal training.
    :type benchmark_steps: Optional[int]
    """

    total_train_epochs: int = 1
    # save control
    save_freq_epochs: Optional[int] = None
    save_freq_steps: Optional[int] = None
    save_freq_secs: Optional[int] = None
    # checkpointing control, only used for recover
    ckpt_freq_epochs: Optional[int] = None
    ckpt_freq_steps: Optional[int] = None
    ckpt_freq_secs: Optional[int] = None
    # eval control
    eval_freq_epochs: Optional[int] = None
    eval_freq_steps: Optional[int] = None
    eval_freq_secs: Optional[int] = None
    # benchmark
    benchmark_steps: Optional[int] = None


@dataclasses.dataclass
class MasterWorker:
    base_seed: int
    exp_ctrl: ExperimentSaveEvalControl
    # main components
    n_model_workers: int
    model_rpcs: List[dfg.MFCDef] = None
    model_topos: Dict[ModelName, topology.PipeModelDataParallelTopology] = None
    msid2mwid: Dict[ModelShardID | str, int] = None
    data_transfer_pairs: List[Tuple[str, str]] = None
    sync_param_pairs: List[Tuple[str, str]] = None
    worker_info: Optional[WorkerInformation] = None


@dataclasses.dataclass
class TasksGroup:
    count: int
    scheduling: Scheduling


@dataclasses.dataclass
class ExperimentScheduling:
    model_worker: Union[List[TasksGroup], TasksGroup] = dataclasses.field(
        default_factory=list
    )
    master_worker: Union[List[TasksGroup], TasksGroup] = dataclasses.field(
        default_factory=list
    )
    controller_image: str = _LLM_CPU_IMAGE


@dataclasses.dataclass
class AutomaticEvaluator:
    """Configuration for automatic evaluation.
    :param data_names: Dataset for evaluation seperated by comma. Currently support datasets stored under ./evaluation/data,
        including "aime24", "amc23" and "math_500". For example, if "aime24" and "amc23" are required for evaluation,
        this field should be set to "aime24,amc23".
    :type data_names: str
    :param max_gen_tokens: Maximum number of tokens to be generated in evaluation.
    :type max_gen_tokens: int
    :param max_concurrent_jobs: Maximum number of concurrent evaluation jobs to submit. If number of existing jobs is equal to
        `max_concurrent_jobs` and a new checkpoint is saved, the evaluation job will wait until former jobs complete.
    :type max_concurrent_jobs: int
    :param eval_job_image: Container image used to launch evaluation job. If set to None, evaluation jobs will use
        GPU image for training.
    :type eval_job_image: Optional[str]
    :param initial_checkpoint_path: Initial checkpoint path to evaluate. If specified, this initial checkpoint will be evaluated,
        results will be stored as global_step = 0.
    :type initial_checkpoint_path: Optional[str]
    :param prompt_type: Prompt format used in evaluation.
    :type prompt_type: str
    """

    data_names: str = "aime24"
    max_gen_tokens: int = 32768
    max_concurrent_jobs: int = 3
    eval_job_image: Optional[str] = None
    initial_checkpoint_path: Optional[str] = None
    prompt_type: str = "deepscaler"


@dataclasses.dataclass
class WandBConfig:
    mode: str = "disabled"
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    job_type: Optional[str] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    config: Optional[Dict] = None


@dataclasses.dataclass
class TensorBoardConfig:
    path: Optional[str] = None


@dataclasses.dataclass
class ExperimentConfig:
    exp_ctrl: ExperimentSaveEvalControl
    wandb: WandBConfig
    tensorboard: TensorBoardConfig
    # dataflow
    model_rpcs: List[dfg.MFCDef]
    model_worker: List[ModelWorker] = dataclasses.field(default_factory=list)
    # master_worker will be set automatically
    master_worker: Optional[List[MasterWorker]] = None
    # automatic evaluation
    auto_eval: bool = False
    evaluator: AutomaticEvaluator = dataclasses.field(
        default_factory=AutomaticEvaluator
    )

    def __post_init__(self):
        self.master_worker = [
            MasterWorker(
                base_seed=self.model_worker[0].base_seed,
                exp_ctrl=self.exp_ctrl,
                n_model_workers=len(self.model_worker),
            )
        ]

    def lazy_init(self):
        assert self.master_worker is not None and len(self.master_worker) == 1
        model_names = set()
        for w in self.model_worker:
            model_names = model_names.union([s.id.model_name for s in w.shards])
        model_names = sorted(list(model_names))

        assert constants.trial_name() is not None
        assert constants.experiment_name() is not None
        graph_path = os.path.join(
            constants.LOG_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "dataflow_graph.png",
        )
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        # If verbose set to True here, every worker will print the graph once
        # due to lazy init on workers.
        G = dfg.build_graph(self.model_rpcs, verbose=False, graph_path=graph_path)
        for rpc in self.model_rpcs:
            rpc._G = G

        self._validate_model_names(model_names)

        model_topos = self._collect_topos(model_names)
        model_configs = self._collect_model_configs(model_names)

        data_transfer_pairs = self._resolve_data_transfer_pairs(model_names)

        sync_param_pairs = self._resolve_param_realloc_pairs(model_configs, model_topos)

        model_names_to_instantiate = self._resolve_model_names_to_instantiate(
            model_names
        )

        for mw in self.model_worker:
            for s in mw.shards:
                s.should_instantiate = s.id.model_name in model_names_to_instantiate

        msid2mwid = {}
        for i, mw in enumerate(self.model_worker):
            mw.model_topos = model_topos
            for m in mw.shards:
                msid2mwid[m.id] = i
        for m in self.model_worker:
            m.msid2mwid = msid2mwid
            m.data_transfer_pairs = data_transfer_pairs
            m.sync_param_pairs = sync_param_pairs

        for m in self.model_worker:
            m.model_rpcs = self.model_rpcs

        # setup master worker config
        self.master_worker[0].model_rpcs = self.model_rpcs
        self.master_worker[0].model_topos = model_topos
        self.master_worker[0].msid2mwid = msid2mwid
        self.master_worker[0].sync_param_pairs = sync_param_pairs
        self.master_worker[0].data_transfer_pairs = data_transfer_pairs

    def resolve_worker_config(self, worker_type, worker_index):
        return getattr(self, worker_type)[worker_index]

    def set_worker_information(self, experiment_name, trial_name):
        if len(self.model_worker) > 0:
            assert len(self.master_worker) == 1

        for worker_type, workers in [
            ("model_worker", self.model_worker),
            ("master_worker", self.master_worker),
        ]:
            if len(workers) == 0:
                continue
            for i, worker in enumerate(workers):
                system_worker_info = dict(
                    experiment_name=experiment_name,
                    trial_name=trial_name,
                    worker_type=worker_type,
                    worker_index=i,
                    worker_count=len(workers),
                )
                if worker.worker_info is not None:
                    worker.worker_info.system_setup(**system_worker_info)
                else:
                    worker.worker_info = WorkerInformation(**system_worker_info)

    def _collect_topos(
        self, model_names: List[ModelName]
    ) -> Dict[ModelName, topology.PipeModelDataParallelTopology]:
        model_topos = {}
        model_allocations = {}
        for model_name in model_names:
            _this_mws_with_indicies = list(
                filter(
                    lambda i_mw: any(
                        x.id.model_name == model_name for x in i_mw[1].shards
                    ),
                    enumerate(self.model_worker),
                )
            )
            _this_mw_indices, _this_mws = zip(*_this_mws_with_indicies)
            _this_mw_indices = tuple(sorted(_this_mw_indices))
            all_shards: List[StandaloneModelShardAbstraction] = [
                next(filter(lambda x: x.id.model_name == model_name, mw.shards))
                for mw in _this_mws
            ]
            model_topos[model_name] = all_shards[0].id.topo
            model_allocations[model_name] = tuple(sorted(_this_mw_indices))

            ##### Sanity check of parallelism ranks. #####
            ranks = [s.id.parallelism_rank for s in all_shards]
            _topos = [s.id.topo for s in all_shards]
            if set(ranks) != set(list(range(len(_this_mws)))) or any(
                _t.world_size() != _topos[0].world_size() for _t in _topos
            ):
                raise ValueError(
                    f"Parallelism rank check failed: model name {model_name}, "
                    f"model shard ids={[s.id for s in all_shards]}."
                )
            ##### Sanity check of parallelism ranks. #####
        return model_topos

    def _collect_model_configs(
        self, model_names: List[ModelName]
    ) -> Dict[ModelName, ModelAbstraction]:
        model_configs = {}
        for model_name in model_names:
            _this_mws = list(
                filter(
                    lambda mw: any(x.id.model_name == model_name for x in mw.shards),
                    self.model_worker,
                )
            )
            all_shards: List[StandaloneModelShardAbstraction] = [
                next(filter(lambda x: x.id.model_name == model_name, mw.shards))
                for mw in _this_mws
            ]
            model_configs[model_name] = all_shards[0].model
        return model_configs

    def _validate_model_names(self, model_names: List[ModelName]):
        model_names = sorted(model_names)
        _roles = set(mn.role for mn in model_names)
        _replica_ids = {
            _role: sorted([mn.replica_id for mn in model_names if mn.role == _role])
            for _role in _roles
        }
        for v in _replica_ids.values():
            if list(sorted(v)) != list(range(len(v))):
                raise ValueError(
                    f"Model replica ids should be 0, 1, 2, ... for each role: {_replica_ids}."
                )

    def _resolve_data_transfer_pairs(
        self, model_names: List[ModelName]
    ) -> List[Tuple[ModelName, ModelName]]:
        data_transfer_pairs: List[Tuple[ModelName, ModelName]] = []
        G = self.model_rpcs[0]._G
        for edge in G.edges():
            mn1 = G.nodes[edge[0]]["object"].model_name
            mn2 = G.nodes[edge[1]]["object"].model_name
            data_transfer_pairs.append((mn1, mn2))
        src_rpcs = [rpc for rpc in self.model_rpcs if rpc.is_src]
        data_src_rpc = src_rpcs[0]
        for r in src_rpcs:
            if (
                data_src_rpc.model_name,
                r.model_name,
            ) not in data_transfer_pairs:
                data_transfer_pairs.append((data_src_rpc.model_name, r.model_name))
        return data_transfer_pairs

    def _resolve_param_realloc_pairs(
        self, model_configs, model_topos
    ) -> List[Tuple[ModelName, ModelName]]:
        sync_param_pairs: List[Tuple[ModelName, ModelName]] = []
        for rpc in self.model_rpcs:
            for hook in rpc._pre_hooks + rpc._post_hooks:
                if not isinstance(hook, dfg.ParamReallocHook):
                    continue
                other_model_name = (
                    hook.target if hook.target is not None else hook.source
                )
                other_topo = (
                    model_topos[hook.target]
                    if hook.target is not None
                    else model_topos[hook.source]
                )
                self_topo = model_topos[rpc.model_name]
                if (
                    self_topo.get_dim("model") % other_topo.get_dim("model") != 0
                    and other_topo.get_dim("model") % self_topo.get_dim("model") != 0
                ):
                    raise ValueError(
                        "To synchronize parameters between two models, "
                        "their model parallel size must be a multiple of each other."
                    )
                if rpc.model_name == other_model_name:
                    raise ValueError(
                        f"Cannot synchronize parameters within the same model "
                        f"(in {rpc}, {rpc.model_name} and {hook.target})."
                    )
                if hook.target is not None:
                    if not (rpc.model_name, hook.target) in sync_param_pairs:
                        sync_param_pairs.append((rpc.model_name, hook.target))
                else:
                    if not (hook.source, rpc.model_name) in sync_param_pairs:
                        sync_param_pairs.append((hook.source, rpc.model_name))
        return sync_param_pairs

    def _resolve_model_names_to_instantiate(
        self, model_names: List[ModelName]
    ) -> List[ModelName]:
        # Mark which shard of the same role should be instantiated.
        roles = set([model_name.role for model_name in model_names])
        role_is_trainable = {role: False for role in roles}
        role_trainable_idx = {}
        role_idx_collection = {role: set() for role in roles}
        for role in roles:
            for rpc in self.model_rpcs:
                if rpc.role != role:
                    continue
                if rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
                    if role_is_trainable[role]:
                        raise ValueError(
                            f"Multiple train_step for the same role {role} is not allowed."
                        )
                    role_is_trainable[role] = True
                    role_trainable_idx[role] = rpc.model_name.replica_id
                role_idx_collection[role].add(rpc.model_name.replica_id)
        role_cnt = {role: len(v) for role, v in role_idx_collection.items()}

        model_names_to_instantiate = []
        for role in roles:
            if role_is_trainable[role]:
                model_names_to_instantiate.append(
                    ModelName(role, role_trainable_idx[role])
                )
            else:
                model_names_to_instantiate += [
                    ModelName(role, i) for i in range(role_cnt[role])
                ]

        return model_names_to_instantiate


class Experiment:
    """Base class for defining the procedure of an experiment."""

    def scheduling_setup(self) -> ExperimentScheduling:
        """Returns the Scheduling of all workers."""
        raise NotImplementedError()

    def initial_setup(self) -> ExperimentConfig | List[ExperimentConfig]:
        """Returns a list of workers to create when a trial of the experiment
        is initialized."""
        raise NotImplementedError()


ALL_EXPERIMENT_CLASSES = {}


def register_experiment(name, cls):
    assert name not in ALL_EXPERIMENT_CLASSES
    ALL_EXPERIMENT_CLASSES[name] = cls


def make_experiment(name) -> Experiment:
    cls = ALL_EXPERIMENT_CLASSES[name]
    return cls()
