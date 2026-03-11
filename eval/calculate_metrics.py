# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from collections import defaultdict
import numpy as np
import torch
from fdfo import utils
from eval import models, prompts

import logging
logging.getLogger('diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3').setLevel(logging.ERROR)


METRIC_MODELS = {
    'pickscore': 'yuvalkirstain/PickScore_v1',
    'hpsv2':     'adams-story/HPSv2-hf',
    'clip_h14':  'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    'clip_l14':  'openai/clip-vit-large-patch14',
}

VLM_QUESTIONS = {
    'vlm_alignment': 'Does this image match the caption "{prompt}"? Answer with Yes or No.',
    'vlm_quality':   'Is this image of professional quality? Answer with Yes or No.',
    'vlm_photo':     'Does this image look photorealistic? Answer with Yes or No.',
}


@dataclass
class MetricsConfig:
    """Configuration for metric calculation."""
    # Checkpoint path, or None to use the base model
    checkpoint: Optional[str] = None
    # Prompt set to use ('pickscore_train' or 'hpdv2')
    prompt_set: Literal['pickscore_train', 'hpdv2'] = 'hpdv2'
    # Number of prompts to evaluate
    num_prompts: int = 500
    # Number of images per prompt
    num_repeats: int = 1
    # Metrics to compute
    metrics: List[Literal['pickscore', 'hpsv2', 'clip_h14', 'clip_l14', 'vlm_alignment', 'vlm_quality', 'vlm_photo', 'dreamsim_diversity']] = field(default_factory=lambda: ['pickscore'])
    # Output file for results (JSON)
    out: Optional[str] = None
    # Random seeds
    prompt_seed: int = 0
    image_seed: int = 0
    # Generation parameters
    width: int = 512
    height: int = 512
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    batch_size: int = 4
    # VLM model (for vlm_* metrics)
    vlm_model: str = 'Qwen/Qwen2.5-VL-7B-Instruct'


def calculate_metrics(config: MetricsConfig) -> Dict[str, float]:
    """
    Calculate evaluation metrics on generated images.

    Returns:
        Dictionary of metric names to average scores
    """
    start_time = time.time()
    device = torch.device('cuda')

    # Validate metrics
    all_metrics = set(METRIC_MODELS.keys()) | set(VLM_QUESTIONS.keys()) | {'dreamsim_diversity'}
    for m in config.metrics:
        if m not in all_metrics:
            raise ValueError(f"Unknown metric: {m}. Available: {sorted(all_metrics)}")

    clip_metrics = [m for m in config.metrics if m in METRIC_MODELS]
    vlm_metrics = [m for m in config.metrics if m in VLM_QUESTIONS]
    use_dreamsim = 'dreamsim_diversity' in config.metrics

    # DreamSim diversity requires multiple images per prompt
    if use_dreamsim and config.num_repeats < 2:
        raise ValueError("dreamsim_diversity requires num_repeats >= 2")

    # Load prompts
    utils.print0(f"Loading prompts: {config.prompt_set}")
    all_prompts = prompts.load_prompts(config.prompt_set)

    # Shuffle and limit prompts
    rng = np.random.RandomState(config.prompt_seed)
    rng.shuffle(all_prompts)
    all_prompts = all_prompts[:config.num_prompts]

    # Partition work across ranks
    rank = utils.get_rank()
    world_size = utils.get_world_size()
    all_prompts = [p for i, p in enumerate(all_prompts) if i % world_size == rank]
    utils.print0(f"Using {config.num_prompts} prompts ({len(all_prompts)} per rank, {world_size} ranks)")

    # Load checkpoint to evaluate
    utils.print0(f"Loading checkpoint: {config.checkpoint}")
    pipe = models.load_pipe(config.checkpoint, device=device)

    # Load CLIP models
    clip_models = {}
    for metric in clip_metrics:
        model_name = METRIC_MODELS[metric]
        utils.print0(f"Loading {metric}: {model_name}")
        clip_models[metric] = models.CLIP(model_name, device=device)

    # Load VLM model
    vlm = None
    if vlm_metrics:
        utils.print0(f"Loading VLM: {config.vlm_model}")
        vlm = models.VLM(config.vlm_model, device=device)

    # Load DreamSim
    dreamsim = None
    if use_dreamsim:
        utils.print0("Loading DreamSim...")
        dreamsim = models.DreamSim(device=device)

    scores = defaultdict(list)
    num_total = len(all_prompts) * config.num_repeats
    prompts_per_batch = max(config.batch_size // config.num_repeats, 1)
    num_batches = (len(all_prompts) + prompts_per_batch - 1) // prompts_per_batch

    utils.print0(f"\nGenerating {num_total} images ({len(all_prompts)} prompts x {config.num_repeats} repeats per rank)...")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * prompts_per_batch
        batch_end = min(batch_start + prompts_per_batch, len(all_prompts))
        batch_prompts_base = all_prompts[batch_start:batch_end]

        # Repeat prompts
        batch_prompts = [p for p in batch_prompts_base for _ in range(config.num_repeats)]
        batch_seeds = [
            config.image_seed + (batch_start + i) * config.num_repeats + r
            for i in range(len(batch_prompts_base))
            for r in range(config.num_repeats)
        ]

        elapsed = time.time() - start_time
        utils.print0(f"Batch {batch_idx + 1}/{num_batches} ({elapsed:.1f}s elapsed)")

        # Generate images
        with torch.no_grad():
            images = models.run_pipe(
                pipe,
                prompts=batch_prompts,
                seed=batch_seeds,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                width=config.width,
                height=config.height,
                batch_size=config.batch_size,
            )

        # CLIP-based metrics
        for metric, clip_model in clip_models.items():
            with torch.no_grad():
                batch_scores = clip_model.calc_scores(batch_prompts, images)
                scores[metric].extend(batch_scores.cpu().tolist())

        # VLM-based metrics
        if vlm is not None:
            for metric in vlm_metrics:
                question_template = VLM_QUESTIONS[metric]
                questions = [question_template.format(prompt=p) for p in batch_prompts]
                with torch.no_grad():
                    batch_scores = vlm.calc_yes_no_scores(questions, images)
                    scores[metric].extend(batch_scores.cpu().tolist())

        # DreamSim diversity
        if dreamsim is not None:
            with torch.no_grad():
                image_embs = dreamsim.calc_image_embs(images)
                for k in range(len(batch_prompts_base)):
                    values = []
                    for i in range(config.num_repeats - 1):
                        for j in range(i + 1, config.num_repeats):
                            emb_i = image_embs[k * config.num_repeats + i]
                            emb_j = image_embs[k * config.num_repeats + j]
                            diff = emb_i - emb_j
                            values.append(diff.square().sum())
                    scores['dreamsim_diversity'].append(torch.stack(values).mean().cpu().item())

    if world_size > 1:
        utils.print0("Gathering scores across ranks...")
        for metric in list(scores.keys()):
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, scores[metric])
            scores[metric] = [s for rank_scores in gathered for s in rank_scores]

    results = {metric: np.mean(score_list) for metric, score_list in scores.items()}

    results['_config'] = {
        'checkpoint':  config.checkpoint,
        'prompt_set':  config.prompt_set,
        'num_prompts': config.num_prompts,
        'num_repeats': config.num_repeats,
        'prompt_seed': config.prompt_seed,
        'image_seed':  config.image_seed,
    }
    results['_elapsed_seconds'] = time.time() - start_time

    return results


def print_results(results: Dict[str, float]):
    """Pretty-print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    if '_config' in results:
        cfg = results['_config']
        print(f"Checkpoint: {cfg['checkpoint']}")
        print(f"Prompt set: {cfg['prompt_set']} ({cfg['num_prompts']} prompts x {cfg['num_repeats']} repeats)")
        print()

    for key, value in sorted(results.items()):
        if not key.startswith('_'):
            print(f"  {key:<20} {value:.4f}")

    if '_elapsed_seconds' in results:
        print(f"\nCompleted in {results['_elapsed_seconds']:.1f} seconds")

    print("="*60)
