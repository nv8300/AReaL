import itertools
import os
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import AllocationMode, FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.multi_turn_bfcl import MultiTurnWorkflow
from areal.utils import logging


#os.environ["NCCL_DEBUG"] = "TRACE"  # 或 "TRACE"（更详细）
#os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
os.environ["NCCL_P2P_LEVEL"] = "NVL"  # H800 支持 NVLink，强制使用 NVLink 通信（减少 PCIe 冲突）

logger = logging.getLogger(__name__)


def bfcl_reward_fn(multi_turn_model_result_list_decoded: list[list[list[str]]], multi_turn_ground_truth_list: list[list[str]], test_entry: dict, test_category="multi_turn_base", model_name="qwen") -> int:
    from areal.reward.bfcl_checker import multi_turn_checker

    return int(multi_turn_checker(multi_turn_model_result_list_decoded, multi_turn_ground_truth_list, test_entry, test_category, model_name))


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    logger.info("finish tokenizer")


    train_dataset = get_custom_dataset(
        path="bfcl",
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        path="bfcl",
        rank=rank,
        world_size=world_size,
        split="test",
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
    )

    logger.info("finish get_custom_dataset")

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.valid_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    logger.info("finish StatefulDataLoader")

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize(None, ft_spec)

    logger.info("finish rollout")

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    logger.info("finish actor init")
    rank = dist.get_rank()

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    event = torch.cuda.Event(enable_timing=True, blocking=True)  # blocking=True 确保事件完成后才触发
    event.record()
    weight_update_meta = [WeightUpdateMeta.from_fsdp_nccl(AllocationMode.from_str(config.allocation_mode), actor)]
    logger.info(f"Rank {rank}: finish weight_update_meta")
    event.wait()
    logger.info(f"Rank {rank}: FSDP internal CUDA ops finished (via event)")

    key_stream = torch.cuda.current_stream()
    key_stream.synchronize()
    #torch.cuda.synchronize()
    logger.info(f"Rank {rank}: All CUDA ops synced before broadcast")

    #dist.barrier()
    #logger.info(f"Rank {rank}: Passed barrier before broadcast")
    dist.broadcast_object_list(weight_update_meta, src=0)
    logger.info(f"Rank {rank}: After broadcast")
    weight_update_meta = weight_update_meta[0]

    logger.info("finish weight_update_meta")

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = MultiTurnWorkflow(
        reward_fn=bfcl_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        max_steps=20,
        turn_discount=0.95,
        model_name="qwen",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    logger.info("finish workflow init")

    eval_workflow = MultiTurnWorkflow(
        reward_fn=bfcl_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        max_steps=20,
        turn_discount=0.95,
        model_name="qwen",
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    logger.info("finish eval_workflow init")

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    logger.info("finish evaluator init")

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )

    logger.info("finish recover init")

    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )
    logger.info("finish start_step")

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = itertools.cycle(train_dataloader)
    logger.info("start StepInfo")
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )
        logger.info(f"get step_info: {step_info}")

        with stats_tracker.record_timing("rollout"):
            logger.info(f"start step_info: {step_info} rollout")
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        logger.info(f"finish step_info: {step_info} rollout")

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        logger.info("finish synchronize")

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        logger.info("finish recompute logp log")
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        logger.info("finish ref log")
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")
        
        logger.info("finish compute advantages log")
        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")
        logger.info("finish ppo_update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)
        logger.info("finish update_weights")

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)
        logger.info("finish save")
        
        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                # Stats are logged in the workflow
                # and will be exported later
                cnt = 0
                for data in valid_dataloader:
                    for item in data:
                        eval_rollout.submit(item, eval_workflow)
                        cnt += 1
                eval_rollout.wait(cnt, timeout=None)

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )
        logger.info("finish evaluate")

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )
        logger.info("finish checkpoint_for_recover")

        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(stats_tracker.export_all(reduce_group=actor.parallelism_group))
        stats_logger.commit(epoch, step, global_step, stats)

        logger.info("finish stats_logger commit")


        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        # Resume rollout
        rollout.resume()
        logger.info("finish rollout all")

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])

