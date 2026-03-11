# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

"""
Generate a set of images for a given FDFO checkpoint.

Usage:
    export CHECKPOINT=https://huggingface.co/nvidia/finite-difference-flow-optimization/tree/main/fdfo-combined-reward-no-cfg/epoch-0000100
    python generate.py --checkpoint $CHECKPOINT --out out.jpg
"""

import tyro
from eval.generate_images import GenerateConfig, generate_prompt_grid

def main():
    config = tyro.cli(GenerateConfig)
    generate_prompt_grid(config)

if __name__ == "__main__":
    main()
