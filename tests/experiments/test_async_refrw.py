import functools
import gc
import json
import os
import pickle
import time
from typing import *

import numpy as np
import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from torch.cuda import is_initialized

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(root_dir)
import sys

sys.path.insert(0, root_dir)

from realhf.api.core import data_api, dfg, model_api
from realhf.api.core.config import ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import constants, logging
from realhf.base.network import find_free_port
from realhf.base.testing import (
    _DEFAULT_EXPR_NAME,
    _DEFAULT_TRIAL_NAME,
    init_global_constants,
)

logger = logging.getLogger("test async ref-rew")
os.environ["REAL_MATH_METADATA_PATH"] = "/storage/datasets/id2info.json"


def loadJson():
    dataDir = os.environ["REAL_MATH_METADATA_PATH"]
    with open(dataDir, "r") as f:
        if dataDir.endswith(".jsonl"):
            samples = [json.loads(line) for line in f.readlines()]
        else:
            samples = json.load(f)

    return samples


def _mock_input(batch_size: int, seq_len):
    vocab_size = 100
    torch.manual_seed(1)
    seqs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    samples = loadJson()
    id_list = list(samples.keys())
    # id_tensor = torch.tensor([id_list[i] for i in range(seqs.shape[0])], dtype=torch.long)  # 使用哈希值编码

    return data_api.SequenceSample.from_default(
        seqlens=[seq_len for _ in range(seqs.shape[0])],
        ids=[id_list[i] for i in range(seqs.shape[0])],
        data=dict(
            packed_input_ids=seqs.view(-1),
            # prompt_mask=torch.zeros_like(seqs.view(-1), dtype=torch.bool),
            packed_prompts=seqs[:, :seq_len].contiguous().view(-1),
        ),
    )


def funcion_call(
    rpc_name: str,
    rank: int,
    world_size: int,
    model_path: str,
    model_family_name: str,
    dp: int,
    pp: int,
    tp: int,
    interface_type: dfg.ModelInterfaceType,
    interface_impl: dfg.ModelInterfaceAbstraction,
    batch_size: int,
    prompt_len: int,
    input_: data_api.SequenceSample | None,
    port: int,
):

    # assert not torch.cuda.is_initialized()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    assert world_size == (
        dp * pp * tp
    ), f"dp={dp}, pp={pp}, tp={tp}, world_size={world_size}"
    assert batch_size % dp == 0, (batch_size, dp)

    # Initialize distributed environment.
    model_name = ModelName("default", 0)
    if not dist.is_initialized():
        logger.info("Setting up distributed environment...")
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://localhost:{port}",
        )
        logger.info("Initialized distributed environment.")
        init_global_constants(
            num_dp=dp,
            num_mp=tp,
            num_pp=pp,
            sequence_parallel=interface_type == dfg.ModelInterfaceType.TRAIN_STEP,
            model_name=model_name,
            max_prompt_len=prompt_len,
        )
    torch.cuda.set_device(0)

    # NOTE: import here to avoid CUDA re-initialization

    from realhf.impl.model.nn.real_llm_api import ReaLModel, add_helper_functions

    # Call a method like `config_from_llama` to get the config.
    mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(
        transformers.AutoConfig.from_pretrained(model_path)
    )
    is_critic = rpc_name in ["critic_inf", "critic_train", "rew_inf"]
    mconfig.is_critic = is_critic
    with constants.model_scope(model_name):
        # Construct the model.
        logger.info(f"Loading model from {model_path}...")
        module = ReaLModel(mconfig, dtype=torch.bfloat16, device="cuda")
        setattr(ReaLModel, "save_to_hf", getattr(ReaLModel, f"to_{model_family_name}"))
        setattr(
            ReaLModel, "load_from_hf", getattr(ReaLModel, f"from_{model_family_name}")
        )
        module._instantiation_hooks.append(
            lambda: getattr(module, f"from_{model_family_name}")(
                load_dir=model_path,
                init_critic_from_actor=is_critic,
            )
        )
        add_helper_functions(module)
        module.instantiate()
        module.eval()

        tokenizer = data_api.load_hf_tokenizer(model_path)

        model = model_api.Model(
            name=model_name,
            module=module,
            tokenizer=tokenizer,
            device=module.device,
            dtype=module.dtype,
        )
        if interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
            from realhf.impl.model.backend.megatron import MegatronTrainBackend

            backend = MegatronTrainBackend()
        else:
            from realhf.impl.model.backend.inference import PipelineInferenceBackend

            backend = PipelineInferenceBackend()

        logger.info("Running backend initialization...")
        ft_spec = model_api.FinetuneSpec(
            total_train_epochs=1,
            dataset_size=128,
            train_batch_size=128,
        )
        model = backend.initialize(model, ft_spec)

        interface = model_api.make_interface(interface_impl)

        if input_ is None:
            input_ = _mock_input(batch_size, prompt_len)

        input_ = input_.cuda()

        mb_spec = model_api.MicroBatchSpec()

        logger.info("Running interface computation...")
        start = time.perf_counter_ns()
        if interface_type == dfg.ModelInterfaceType.GENERATE:
            res = interface.generate(model, input_, mb_spec)
        elif interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
            res = interface.train_step(model, input_)
        else:
            res = interface.inference(model, input_, mb_spec)

        if constants.model_parallel_rank() == 0 and constants.is_last_pipe_stage():
            if isinstance(res, data_api.SequenceSample):
                res = res.cpu()

        comsumed = time.perf_counter_ns() - start
        logger.info(f"{rpc_name} Computation done. {comsumed} ns")
    return res


def run_function_call(
    rpc_name: str,
    model_path: str,
    model_family_name: str,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    input_: data_api.SequenceSample | None,
) -> data_api.SequenceSample | None:
    assert rpc_name in [
        "actor_gen",
        "actor_train",
        "critic_inf",
        "rew_inf",
        "critic_train",
        "ref_inf",
        "ref_rw",
    ]

    ref_rw_interface = dfg.ModelInterfaceAbstraction(
        "ref_rw",
        args=dict(
            generation_config=dict(
                max_new_tokens=gen_len, min_new_tokens=gen_len, greedy=True
            ),
            rew_inf_args=dict(
                tokenizer_path=model_path,
            ),
        ),
    )

    ppo_actor_interface = dfg.ModelInterfaceAbstraction(
        "ppo_actor",
        args=dict(
            generation_config=dict(
                max_new_tokens=gen_len, min_new_tokens=gen_len, greedy=True
            ),
            rew_inf_args=dict(
                tokenizer_path=model_path,
            ),
        ),
    )
    ppo_critic_interface = dfg.ModelInterfaceAbstraction("ppo_critic")
    rw_interface = dfg.ModelInterfaceAbstraction(
        "paired_rw",
    )
    if rpc_name == "actor_gen":
        interface_type = dfg.ModelInterfaceType.GENERATE
        interface_impl = ppo_actor_interface
    elif rpc_name == "actor_train":
        interface_type = dfg.ModelInterfaceType.TRAIN_STEP
        interface_impl = ppo_actor_interface
    elif rpc_name == "critic_inf":
        interface_type = dfg.ModelInterfaceType.INFERENCE
        interface_impl = ppo_critic_interface
    elif rpc_name == "ref_inf":
        interface_type = dfg.ModelInterfaceType.INFERENCE
        interface_impl = ppo_actor_interface
    elif rpc_name == "ref_rw":
        interface_type = dfg.ModelInterfaceType.INFERENCE
        interface_impl = ref_rw_interface
    elif rpc_name == "critic_train":
        interface_type = dfg.ModelInterfaceType.TRAIN_STEP
        interface_impl = ppo_critic_interface
    else:
        interface_type = dfg.ModelInterfaceType.INFERENCE
        interface_impl = rw_interface

    logger.info(f"Running RPC {rpc_name}...")

    port = find_free_port()
    res = funcion_call(
        rank=0,
        rpc_name=rpc_name,
        world_size=1,
        model_path=model_path,
        model_family_name=model_family_name,
        dp=1,
        pp=1,
        tp=1,
        interface_type=interface_type,
        interface_impl=interface_impl,
        batch_size=batch_size,
        prompt_len=prompt_len,
        input_=input_,
        port=port,
    )

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    if isinstance(res, data_api.SequenceSample):
        return res
    else:
        logger.info(f"RPC {rpc_name} stats: {res}")


def main():
    mp.set_start_method("spawn", force=True)

    model_family_name = "qwen2"
    batch_size = 16
    prompt_len = 128
    gen_len = 4096
    model_path = "/storage/models/DeepSeek-R1-Distill-Qwen-1.5B"

    constants.set_experiment_trial_names(_DEFAULT_EXPR_NAME, _DEFAULT_TRIAL_NAME)

    for i in range(2):
        ref_rw_res = run_function_call(
            "ref_rw",
            model_family_name=model_family_name,
            model_path=model_path,
            batch_size=batch_size,
            prompt_len=prompt_len,
            gen_len=gen_len,
            input_=None,
        )


if __name__ == "__main__":
    main()
