import os
import sys

from torch.utils.tensorboard import SummaryWriter
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import RMPairedConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.dataset import get_custom_dataset
from areal.engine.rm_paired.rm_paired_engine import FSDPRMPairedEngine
from areal.utils import seeding, stats_tracker
from areal.utils.data import pad_rm_paired_sequences_to_tensors
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.utils import logging


logger = logging.getLogger(__name__)


def main(args):
    config, _ = load_expr_config(args, RMPairedConfig)
    config: RMPairedConfig

    writer = SummaryWriter(log_dir=config.cluster.fileroot+'/tf-logs')

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    logger.info("finish tokenizer")

    seeding.set_random_seed(config.seed, f"trainer{rank}")

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        path=config.valid_dataset.path,
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
        collate_fn=pad_rm_paired_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=pad_rm_paired_sequences_to_tensors,
        drop_last=config.valid_dataset.drop_last,
    )

    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    logger.info("finish StatefulDataLoader")

    engine = FSDPRMPairedEngine(config=config.model)
    engine.initialize(None, ft_spec)
    logger.info("finish engine.initialize")

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        engine,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
    )
    logger.info("finish saver&evaluator&recover_handler")
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    len(train_dataloader)
    logger.info("finish start_step")

    global_step = 0
    for epoch in range(total_epochs):
        logger.info(f"start epoch: {epoch}")
        for step, data in enumerate(train_dataloader):
            logger.info(f"start step: {step}")
            if global_step < start_step:
                global_step += 1
                continue
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=len(train_dataloader),
            )
            logger.info(f"finish {epoch}/{step} step_info")
            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("sft"),
            ):
                stats = engine.train_rm_paired(data)
                engine.step_lr_scheduler()
                stats_tracker.scalar(**stats)
            logger.info(f"finish {epoch}/{step} train_rm_paired")

            with stats_tracker.record_timing("save"):
                saver.save(engine, epoch, step, global_step, tokenizer=tokenizer)
            logger.info(f"finish {epoch}/{step} save")

            with stats_tracker.record_timing("eval"):
                # No need to log anything. Logging will be handled outside
                # via stats_tracker.export().
                def evaluate_fn():
                    with stats_tracker.scope("sft-eval"):
                        for data in valid_dataloader:
                            engine.evaluate_rm_paired(data)

                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )
            logger.info(f"finish {epoch}/{step} evaluate")

            with stats_tracker.record_timing("checkpoint_for_recover"):
                recover_handler.dump(
                    engine,
                    step_info,
                    saver,
                    evaluator,
                    stats_logger,
                    train_dataloader,
                    tokenizer=tokenizer,
                )

            stats_logger.commit(
                writer,
                epoch,
                step,
                global_step,
                stats_tracker.export(reduce_group=engine.parallelism_group),
            )
            logger.info(f"finish {epoch}/{step} stats_logger")
            global_step += 1
    
    logger.info("finish all epoch")
    stats_logger.close()
    engine.destroy()
    logger.info("FINISH")


if __name__ == "__main__":
    main(sys.argv[1:])