# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import os
import json
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.distributed
import wandb

@dataclass
class RolloutLogData:
    """Data from rollouts for logging purposes only."""
    epoch: int
    rewards: torch.Tensor # Interleaved [branch 1 reward 1, branch 2 reward 1, branch 1 reward 2, branch 2 reward 2, ...]
    images: List[torch.Tensor] # Interleaved [branch 1 image 1, branch 2 image 1, branch 1 image 2, branch 2 image 2, ...]
    prompts: List[str] # One per pair (not interleaved)


@dataclass
class TrainingStepMetrics:
    clipfrac: float
    policy_loss: float
    loss: float
    epoch: int


class MetricsAccumulator:

    def __init__(self):
        self.data = defaultdict(list)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v.detach() if hasattr(v, 'detach') else v)

    def reduce_and_clear(self, world_size: int = 1) -> Dict[str, float]:
        result = {k: torch.mean(torch.stack(v)) for k, v in self.data.items()}
        if world_size > 1:
            for v in result.values():
                torch.distributed.all_reduce(v)
                v /= world_size
        self.data.clear()
        return {k: v.item() for k, v in result.items()}


class TrainingLogger:

    def __init__(self, run_dir: str, run_name: str, config: Any, rank: int = 0, world_size: int = 1):
        self.run_dir = run_dir
        self.use_wandb = config.log_wandb
        self.rank = rank
        self.world_size = world_size
        self.jsonl_path = os.path.join(run_dir, "training_log.jsonl")

        if self.is_main and self.use_wandb:
            config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else dict(config)
            wandb.init(
                project="fdfo",
                name=run_name,
                config=config_dict,
            )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def log_rollout(self, data: RolloutLogData, global_step: int):
        rewards = self._gather_rewards(data.rewards)
        if not self.is_main:
            return

        log_data = {"epoch": data.epoch, "reward": rewards, "reward_mean": rewards.mean(), "reward_std": rewards.std()}
        self._log_to_jsonl(log_data, global_step)
        self._log_to_wandb(log_data, global_step)

    def log_images(self, data: RolloutLogData, global_step: int, resolution: int, log_every_n_epochs: int = 10):
        if not self.is_main or not self.use_wandb:
            return
        if data.epoch % log_every_n_epochs != 0:
            return

        pairs = list(zip(data.images[0::2], data.images[1::2], data.rewards[0::2], data.rewards[1::2], data.prompts))

        images_for_wandb = []
        for img1, img2, r1, r2, prompt in random.sample(pairs, min(15, len(pairs))):
            combined = torch.cat([img1, img2], dim=2)
            pil = Image.fromarray((combined.float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil = pil.resize((resolution * 2, resolution))

            r1 = r1.item() if hasattr(r1, 'item') else r1
            r2 = r2.item() if hasattr(r2, 'item') else r2

            images_for_wandb.append(wandb.Image(pil, caption=f"reward left: {r1:.2f} | reward right: {r2:.2f} | {prompt:.100}"))

        wandb.log({"images": images_for_wandb}, step=global_step)

    def log_training_step(self, metrics: TrainingStepMetrics, global_step: int):
        if not self.is_main:
            return

        log_data = {
            "clipfrac": metrics.clipfrac, # Clip frac reports as if we used hard PPO clipping even though we use SPO clipping
            "policy_loss": metrics.policy_loss,
            "loss": metrics.loss,
            "epoch": metrics.epoch,
        }
        self._log_to_wandb(log_data, global_step)

    def _log_to_wandb(self, data: Dict, step: int):
        if not self.use_wandb:
            return
        wandb.log(data, step=step)

    def _log_to_jsonl(self, data: Dict, step: int):
        log_entry = self._convert_for_json(dict(data))
        log_entry["step"] = step
        log_entry["_timestamp"] = time.time()

        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _convert_for_json(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        return obj

    def _gather_rewards(self, rewards: torch.Tensor) -> np.ndarray:
        if self.world_size > 1:
            gathered = [torch.zeros_like(rewards) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered, rewards)
            rewards = torch.cat(gathered)
        return rewards.float().cpu().numpy()
