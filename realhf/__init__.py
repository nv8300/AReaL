# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

# Initialize preset config before all submodules.
from .base import prologue

# Re-import these classes for clear documentation,
# otherwise the name will have a long prefix like
# realhf.api.quickstart.model.ModelTrainEvalConfig.
from .api.core.config import ModelFamily, ModelName, ModelShardID
from .api.core.data_api import SequenceSample
from .api.core.dfg import MFCDef
from .api.core.model_api import (
    FinetuneSpec,
    GenerationHyperparameters,
    Model,
    ModelBackend,
    ModelInterface,
    PipelinableEngine,
    ReaLModelConfig,
)
from .api.quickstart.dataset import (
    PairedComparisonDatasetConfig,
    PromptAnswerDatasetConfig,
    PromptOnlyDatasetConfig,
)
from .api.quickstart.device_mesh import MFCConfig
from .api.quickstart.model import (
    ModelTrainEvalConfig,
    OptimizerConfig,
    ParallelismConfig,
)
from .experiments.common.common import CommonExperimentConfig, ExperimentSaveEvalControl
from .experiments.common.ppo_math_exp import PPOHyperparameters, PPOMATHConfig
from .experiments.common.sft_exp import SFTConfig

__version__ = "0.3.0"
