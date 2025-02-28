# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
import random

import numpy as np
import torch
import transformers

_SEED = None


def set_random_seed(seed):
    global _SEED
    _SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_seed() -> int:
    global _SEED
    return _SEED
