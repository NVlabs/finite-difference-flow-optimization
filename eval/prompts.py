# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

"""
Prompt loading for evaluation.

Supported prompt sets:
- pickscore_train: PickScore training prompts (local txt file, see README for download instructions)
- hpdv2: HPD v2 benchmark prompts (download with: python -m eval.prompts)
"""

import os
import json
from typing import List
from fdfo import utils


PROMPT_DIR = 'prompt_sets'

PROMPT_GRID = [
    ['Prompts used in Figures 3-11',
        'A girl with pigtails is holding a giant sunflower.',
        'a green tractor plowing a field at sunrise.',
        'A quaint cottage nestled in a vibrant flower-filled meadow.',
        'A cat wearing ski goggles is exploring in the snow.',
        'A cat-dragon hybrid. Photograph.',
    ],
    ['Prompts used in Figure 12',
        'Lunch in Bavaria - oil painting',
        'A cat dressed as a wizard in broad daylight.',
        'Budapest as a beautiful flowerpunk city, flowerpunk, hyper realistic, high quality, 8k',
        'Kids race their bikes down the hill as their friends cheer from the sidelines, and a kite flutters in the breeze above them.',
        'A quiet suburban cul-de-sac, where children play in its enclosed street.',
    ],
    ['Prompts used in Figures 13-14',
        "A soft, fabric teddy bear sitting on a child's wooden chair, under the warm glow of a brass lamp.",
        'outdoor full body shot on Canon DS of a toddler dressed as a medieval emperor, unforgettable dress, intricate details, insane details, v 5,',
        'A beach scene where the sandcastle appears taller than the nearby cooler.',
        'A single rose growing through a crack.',
        'A sphinx talking with travelers in front of the Pyramids at sunset.',
    ],
]


def load_prompts(name: str) -> List[str]:
    """Load prompts by name."""
    if name == 'pickscore_train':
        path = os.path.join(PROMPT_DIR, 'pickscore_train.txt')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PickScore prompts not found at {path}. "
                f"Download with: mkdir -p prompt_sets && curl -o prompt_sets/pickscore_train.txt "
                f"https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/train.txt"
            )
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif name == 'hpdv2':
        path = os.path.join(PROMPT_DIR, 'hpdv2.json')
        if not os.path.exists(path) and utils.get_rank() == 0:
            print(f"HPD v2 prompts not found, downloading...")
            download_hpdv2()
        utils.barrier()
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown prompt set: {name}. Supported: 'pickscore_train', 'hpdv2'")


def download_hpdv2():
    """Download HPD v2 prompts from HuggingFace."""
    from huggingface_hub import hf_hub_download

    print("Downloading HPD v2 prompts from HuggingFace...")
    prompts = []
    for category in ['anime', 'concept-art', 'paintings', 'photo']:
        path = hf_hub_download('zhwang/HPDv2', filename=f'benchmark/{category}.json', repo_type='dataset')
        with open(path, 'rb') as f:
            prompts += json.load(f)

    os.makedirs(PROMPT_DIR, exist_ok=True)
    path = os.path.join(PROMPT_DIR, 'hpdv2.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(prompts)} prompts to {path}")


if __name__ == '__main__':
    download_hpdv2()
