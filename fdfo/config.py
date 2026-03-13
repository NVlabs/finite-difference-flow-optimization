# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class SampleConfig:
    num_solver_steps: int = 40 # Number of sampling steps
    guidance_scale: float = 1.0 # CFG scale (1.0 = no guidance)
    batch_size: Optional[int] = None # Per-rank
    num_batches_per_epoch: Optional[int] = None # Multiplies epoch batch size with same peak memory usage
    resolution: int = 512

@dataclass
class TrainConfig:
    batch_size: Optional[int] = None # Per-rank
    learning_rate: float = 3e-5
    clip_range: float = 3e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: Optional[int] = None
    max_grad_norm: float = 1.0
    kl_weight: float = 0.0 # KL strength w.r.t. base model

@dataclass
class VLMRewardPromptConfig:
    prompt_template: str = ""
    target_token: str = "Yes"
    contrast_token: str = "No"
    reward_name: Optional[str] = None # Convenience for logging, etc

@dataclass
class RewardSpec:
    kind: Literal["vlm", "pickscore"] = "vlm"
    weight: float = 1.0
    prompt_config: Optional[VLMRewardPromptConfig] = None # Only needed for vlm kind

@dataclass
class ChurnScheduleConfig:
    churn_weight: float = 1.0

@dataclass
class IntervalScheduleConfig(ChurnScheduleConfig):
    mu: float = 1.0
    sigma: float = 1.5
    churn_weight: float = 1.0
    churn_weight_sigma: float = 0.0
    soft_sigma: float = 0.25

@dataclass
class PriorScheduleConfig(ChurnScheduleConfig):
    grad_mu: float = 1.0
    grad_sigma: float = 1.5
    churn_weight: float = 0.1
    churn_weight_sigma: float = 1.0

@dataclass
class RawChurnScheduleConfig(ChurnScheduleConfig):
    churn_weight: float = 0.03
    grad_mu: float = 0.5
    grad_sigma: float = 1.0


VLM_ALIGNMENT_CONFIG = VLMRewardPromptConfig(
    reward_name="caption_matching",
    prompt_template='Does this image match the caption "{prompt}"? Answer Yes or No.',
    target_token="Yes",
    contrast_token="No",
)

REWARD_PRESETS = {
    "pickscore": [
        RewardSpec(kind="pickscore", weight=1.0),
    ],
    "vlm_alignment": [
        RewardSpec(kind="vlm", weight=1.0, prompt_config=VLM_ALIGNMENT_CONFIG),
    ],
    "combined": [
        RewardSpec(kind="pickscore", weight=1.0),
        RewardSpec(kind="vlm", weight=0.1, prompt_config=VLM_ALIGNMENT_CONFIG),
    ],
}

def get_reward_preset(name: str) -> list[RewardSpec]:
    if name not in REWARD_PRESETS:
        raise ValueError(f"Unknown reward preset: {name}. Available: {list(REWARD_PRESETS.keys())}")
    return REWARD_PRESETS[name]


@dataclass
class FDFOConfig:
    """Main configuration"""
    run_name: str = ""
    log_wandb: bool = False
    seed: int = 6 # There will still be some run-to-run variation
    checkpoint_subdir: str = "checkpoints"
    num_epochs: int = 1000
    save_freq: int = 5 # Checkpoint frequency
    resume_from: str = ""
    use_lora: bool = True # Non-LoRA training is not supported
    lora_rank: int = 32
    lora_alpha: int = 64
    vlm_model_name: str = "qwenvl-7b"
    vlm_use_fsdp: bool = True # Use FSDP to shard VLM reward model across GPUs to save memory
    pretrained_model: str = "stabilityai/stable-diffusion-3.5-medium"
    memory_efficient: bool = True # Enable memory optimizations (e.g., offload text encoders during training)
    prompt_fn: str = "pickscore"
    prompt_fn_kwargs: dict = field(default_factory=lambda: {'dataset_path': 'prompt_sets', 'split': 'pickscore_train'})
    total_pairs_per_epoch: int = 432
    gradient_steps_per_epoch: int = 4
    max_batch_size: Optional[int] = None # Max micro-batch size (for both sampling and training); determines peak VRAM
    sample: SampleConfig = field(default_factory=SampleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward_preset: Literal["pickscore", "vlm_alignment", "combined"] = "combined"
    reward_specs: list[RewardSpec] = field(default_factory=list) # Overrides reward_preset when non-empty
    churn_schedule: ChurnScheduleConfig = field(default_factory=lambda: RawChurnScheduleConfig(churn_weight=0.0025))
