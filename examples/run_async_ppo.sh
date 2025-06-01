#!/bin/bash
python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=/storage/testing/experiments \
    actor.type._class=qwen3 \
    actor.path=/storage/testing/models/Qwen__Qwen3-1.7B \
    ref.type._class=qwen3 \
    ref.path=/storage/testing/models/Qwen__Qwen3-1.7B \
    dataset.path=/storage/testing/dataset/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=4096 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=32768 \
    actor_inf.mb_spec.max_tokens_per_mb=32768 \
    max_concurrent_rollouts=16 \
    max_head_offpolicyness=4