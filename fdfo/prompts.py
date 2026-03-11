# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

import os
import numpy as np
from fdfo.config import FDFOConfig

class PromptDataLoader:
    """
    Prompt dataloader that preloads all prompts into system memory
    """

    def __init__(self, config: FDFOConfig, shuffle=True, seed=None):
        self.prompt_fn_name = config.prompt_fn
        self.prompt_fn_kwargs = config.prompt_fn_kwargs
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        print(f"Loading prompts for '{self.prompt_fn_name}'...")
        self.prompts = self._load_all_prompts()
        print(f"Loaded {len(self.prompts)} prompts into memory")

        self._indices = np.arange(len(self.prompts))
        self._position = 0
        if self.shuffle:
            self.rng.shuffle(self._indices)

    def _load_all_prompts(self):
        """Load all prompts based on prompt function name. Only pickscore is supported here, but could be extended to other functions."""
        if self.prompt_fn_name == "pickscore":
            dataset_path = self.prompt_fn_kwargs.get('dataset_path')
            split = self.prompt_fn_kwargs.get('split')
            file_path = os.path.join(dataset_path, f'{split}.txt')
            with open(file_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            raise ValueError(f"Unknown prompt function: {self.prompt_fn_name}")

    def get_batch(self, batch_size):
        prompts = []
        for _ in range(batch_size):
            if self._position >= len(self._indices): # Reshuffle when reaching end of data
                if self.shuffle:
                    self.rng.shuffle(self._indices)
                self._position = 0

            idx = self._indices[self._position]
            prompts.append(self.prompts[idx])
            self._position += 1

        return prompts
