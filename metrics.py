# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

"""
Evaluate a given FDFO checkpoint using various metrics.

Usage:
    export CHECKPOINT=https://huggingface.co/nvidia/finite-difference-flow-optimization/tree/main/fdfo-combined-reward-no-cfg/epoch-0000100

    # Training-time rewards
    torchrun --nproc_per_node=8 python metrics.py --checkpoint $CHECKPOINT \
        --prompt-set pickscore_train --num-prompts 4096 --num-repeats 1 --metrics pickscore vlm_alignment

    # External control metrics
    torchrun --nproc_per_node=8 python metrics.py --checkpoint $CHECKPOINT \
        --prompt-set hpdv2 --num-prompts 3200 --num-repeats 4 --metrics hpsv2 clip_h14 clip_l14 dreamsim_diversity
"""

import os
import json
import tyro
from eval.calculate_metrics import MetricsConfig, calculate_metrics, print_results
from fdfo import utils

def main():
    config = tyro.cli(MetricsConfig)

    # Initialize distributed (works single-GPU too)
    utils.init()

    # Calculate metrics
    results = calculate_metrics(config)

    if utils.get_rank() == 0:
        print_results(results)

        if config.out:
            os.makedirs(os.path.dirname(config.out) or '.', exist_ok=True)
            with open(config.out, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {config.out}")

if __name__ == "__main__":
    main()
