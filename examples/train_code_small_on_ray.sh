#!/bin/sh
MODEL_FAMILY=qwen2

EXP_NAME="$1"
MODEL_NAME="$2"
DATASET_NAME="$3"
TRAIN_BATCH_SIZE="$4"
GROUP_SIZE="$5"
NODES="$6"
ALLOCATION_MODE="$7"
MAX_NEW_TOKENS=$8
MAX_NUM_SEQS=$9
PPO_MBS=${10}
KL_CTL=${11}

MAX_TOKEN_PER_MB=$(expr 2048 + ${MAX_NEW_TOKENS} + 1024)
MAX_SEQ_LEN_TO_CAPTURE=$(expr 2048 + ${MAX_NEW_TOKENS})

BASE_MODEL_PATH="/storage/models/${MODEL_NAME}"

# original data
DATA_PATH="/storage/datasets/${DATASET_NAME}"
REAL_CODE_METADATA_PATH="/storage/datasets/codeparrot-apps-test.jsonl"

# Option 1: The experiment runs locally with subprocesses.
# MODE=local
# Option 2: The experiment runs in a Ray cluster
# MODE=ray
# Option 3: The experiment runs in a SLURM + pyxis cluster
# Using the slurm mode requires a cluster spec file
# and setting CLUSTER_SPEC_PATH to the path of it.
MODE=ray

# `experiment_name` and `trial_name` can be arbitrary.
# Logs and saved checkpoints will be indexed by them.
#EXP_NAME=ppo-zero--${MODEL_NAME}--${DATASET_NAME}
#EXP_NAME=ppo-zero-distill-1.5B-default
TRIAL_NAME="${TRAIN_BATCH_SIZE}x${GROUP_SIZE}-n${NODES}"

# We use the "heuristic" allocation mode here to automatically determine the parallelism strategy
# for each model function call, i.e., actor generation, critic inference, actor train, etc.
# The number of GPUs is `n_nodes` * `n_gpus_per_node` (not set explictly here, defaults to 8).
# ReaL will make full use of these available GPUs to design allocations.
# This does not ensure the optimal throughput, but it is a good starting point.

# The `heuristic` allocation mode is not ensured to run with every model configurations.
# For example, if the vocabulary size is an odd number, the model parallelism may not work.
# In these cases, you can use the `ppo_manual.sh` to specify the parallelism strategy manually.

# The `ppo` subcommand specifies that this is a PPO experiment.
# The `save_freq_steps` is set to `null` to disable saving checkpoints.
# Enable it if you want to save checkpoints.
# The `ppo` option is used to control the generation and PPO algorithm hyperparameters.
# Note that the performance of PPO is sensitive to the the pre-trained model and hyperparameters.
# It's the user's responsibility to tune them appropriately.
unset CLUSTER_SPEC_PATH
CLUSTER_SPEC_PATH=/storage/ray/cluster_config_on_ray.json \
REAL_CODE_METADATA_PATH=${REAL_CODE_METADATA_PATH} \
FUNCTIONCALL_SERVICE_DOMAIN="" \
REAL_GPU_MEMORY_KILL_THRESHOLD=1 \
python3 -m realhf.apps.quickstart ppo-code \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    wandb.mode=disabled \
    exp_ctrl.total_train_epochs=1 \
    exp_ctrl.save_freq_epochs=1 \
    exp_ctrl.ckpt_freq_secs=600 \
    group_size=${GROUP_SIZE} \
    group_adv_norm=False \
    use_dense_reward=False \
    reward_delta=True \
    rw_type=sparse \
    check_xml_format=False \
    actor.type._class=$MODEL_FAMILY \
    actor.path=$BASE_MODEL_PATH \
    actor.vllm.hybrid_train=False \
    actor.vllm.enforce_eager=False \
    actor.vllm.max_seq_len_to_capture=${MAX_SEQ_LEN_TO_CAPTURE} \
    actor.vllm.max_num_seqs=${MAX_NUM_SEQS} \
    actor.vllm.gpu_memory_utilization=1 \
    actor.vllm.swap_space=64 \
    critic.type._class=$MODEL_FAMILY \
    critic.type.is_critic=True \
    critic.init_critic_from_actor=True \
    critic.path=$BASE_MODEL_PATH\
    ref.type._class=$MODEL_FAMILY \
    ref.path=$BASE_MODEL_PATH \
    rew.type._class=$MODEL_FAMILY \
    rew.type.is_critic=True \
    rew.init_critic_from_actor=True \
    rew.path=$BASE_MODEL_PATH \
    dataset.path=$DATA_PATH \
    dataset.max_prompt_len=2048 \
    dataset.train_bs_n_seqs=${TRAIN_BATCH_SIZE} \
    ppo.gen.max_new_tokens=${MAX_NEW_TOKENS} \
    ppo.gen.min_new_tokens=0 \
    ppo.disable_value=True \
    ppo.gen.top_p=1 ppo.gen.top_k=1000000 \
    ppo.ppo_n_minibatches=${PPO_MBS} \
    ppo.gen.temperature=0.6 \
    ppo.kl_ctl=${KL_CTL} \
    ppo.value_eps_clip=0.2 \
    ppo.reward_output_scaling=5 \
    ppo.reward_output_bias=0.0 \
    ppo.adv_norm=True ppo.value_norm=True \
    mask_too_long=False \
    ppo.discount=1.0 \
    actor.optimizer.lr=1e-6 \
    critic.optimizer.lr=5e-6 \
    actor.optimizer.lr_scheduler_type=constant \
    actor_gen.mb_spec.max_tokens_per_mb=${MAX_TOKEN_PER_MB} \
    ref_inf.mb_spec.max_tokens_per_mb=${MAX_TOKEN_PER_MB} \
    rew_inf.mb_spec.max_tokens_per_mb=${MAX_TOKEN_PER_MB} \
    critic_inf.mb_spec.max_tokens_per_mb=${MAX_TOKEN_PER_MB} \
    actor_train.mb_spec.max_tokens_per_mb=${MAX_TOKEN_PER_MB} \
    critic_train.mb_spec.max_tokens_per_mb=${MAX_TOKEN_PER_MB} \
    cache_clear_freq=1 \
    n_nodes=${NODES} \
    allocation_mode="'${ALLOCATION_MODE}'" n_gpus_per_node=8 \
    recover_mode=auto \
    recover_retries=10 \
    torch_cache_mysophobia=True
