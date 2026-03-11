# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

"""
Launch FDFO training.

Usage:
    # Sanity check using a single GPU
    python train.py --num-epochs 5 --total-pairs-per-epoch 16

    # Reproducing our results using torchrun multi-GPU
    torchrun --nproc_per_node=8 train.py
"""

import os
import re
import math
import tyro
from fdfo.train_loop import train_fdfo
from fdfo.utils import get_rank, get_world_size, print0
from fdfo.config import FDFOConfig


def configure_training_batches(config: FDFOConfig):
    world_size = get_world_size()
    print0(f"Configuring hyperparameters for {world_size} GPUs")

    # Choose max_batch_size.
    if config.max_batch_size is None:
        config.max_batch_size = 6 if world_size < 8 else 13

    # Choose pairs_per_gpu so that its value * 2 is divisible by gradient_steps_per_epoch.
    pairs_per_gpu = config.total_pairs_per_epoch // world_size
    pair_alignment = config.gradient_steps_per_epoch // math.gcd(2, config.gradient_steps_per_epoch)
    pairs_per_gpu -= pairs_per_gpu % pair_alignment
    assert pairs_per_gpu > 0, \
        f"Not enough pairs per GPU: total_pairs_per_epoch={config.total_pairs_per_epoch}, world_size={world_size}, gradient_steps_per_epoch={config.gradient_steps_per_epoch}"

    # Choose sample.batch_size.
    if config.sample.batch_size is None:
        config.sample.batch_size = 1
        for bs in range(config.max_batch_size, 0, -1):
            if pairs_per_gpu % bs == 0:
                config.sample.batch_size = bs
                break

    # Finalize sampling config.
    if config.sample.num_batches_per_epoch is None:
        config.sample.num_batches_per_epoch = pairs_per_gpu // config.sample.batch_size
    print0(f"Sampling config: {config.sample.batch_size} pairs/batch * {config.sample.num_batches_per_epoch} batches * {world_size} GPUs" +
           f" = {config.sample.batch_size * config.sample.num_batches_per_epoch * world_size} pairs/epoch")

    # Choose train.batch_size.
    samples_per_step_per_gpu = (pairs_per_gpu * 2) // config.gradient_steps_per_epoch  # Exact division.
    if config.train.batch_size is None:
        config.train.batch_size = 1
        for bs in range(config.max_batch_size, 0, -1):
            if samples_per_step_per_gpu % bs == 0:
                config.train.batch_size = bs
                break

    # Finalize training config.
    if config.train.gradient_accumulation_steps is None:
        config.train.gradient_accumulation_steps = samples_per_step_per_gpu // config.train.batch_size
    print0(f"Training config: {config.train.batch_size} samples/batch * {config.train.gradient_accumulation_steps} accum * {world_size} GPUs" +
           f" = {config.train.batch_size * config.train.gradient_accumulation_steps * world_size} samples/step")


def get_next_run_id(run_dir_root: str) -> int:
    """Get next available run ID by scanning existing directories."""
    os.makedirs(run_dir_root, exist_ok=True)
    dir_names = [entry.name for entry in os.scandir(run_dir_root) if entry.is_dir()]
    pattern = re.compile(r"^\d+")
    run_id = 0
    for name in dir_names:
        match = pattern.match(name)
        if match:
            run_id = max(run_id, int(match.group()) + 1)
    return run_id


def create_run_dir(run_dir_root: str, run_desc: str) -> tuple[str, int]:
    """Create a new run directory with incrementing ID. Only call from rank 0."""
    run_id = get_next_run_id(run_dir_root)
    run_name = f"{run_id:05d}-{run_desc}"
    run_dir = os.path.join(run_dir_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def create_run_dir_distributed(run_dir_root: str, run_desc: str) -> tuple[str, int]:
    """Create run directory from rank 0 and broadcast to all ranks."""
    import torch.distributed as dist
    if get_world_size() == 1:
        return create_run_dir(run_dir_root, run_desc)

    if not dist.is_initialized():
        dist.init_process_group(backend='gloo')

    result = [create_run_dir(run_dir_root, run_desc) if get_rank() == 0 else (None, None)]
    dist.broadcast_object_list(result, src=0)
    dist.destroy_process_group()
    return result[0]


def main():
    config = tyro.cli(FDFOConfig)
    configure_training_batches(config)

    # Auto-generate run name from config if not specified.
    if not config.run_name:
        config.run_name = f'{config.reward_preset}-cfg_{str(config.sample.guidance_scale).replace(".", "p")}'

    run_dir_root = './runs'
    run_dir, run_id = create_run_dir_distributed(run_dir_root, config.run_name)
    print0(f'Run directory: {run_dir}')

    train_fdfo(config, run_dir, run_id)


if __name__ == "__main__":
    main()
