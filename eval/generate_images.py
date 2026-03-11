# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

import os
import functools
from dataclasses import dataclass
from typing import Optional
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from eval import models, prompts


def save_image(path: str, image: np.ndarray):
    """Save image to disk."""
    print(f'Saving to {path} ...')
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)
    PIL.Image.fromarray(image, 'RGB').save(path, quality=95)


@functools.lru_cache(None)
def get_ttf_font_path():
    """Get path to a TrueType font in a cross-platform way."""
    import urllib.request
    url = "http://fonts.gstatic.com/s/opensans/v17/mem8YaGs126MiZpBA-U1UpcaXcl0Aw.ttf" # Open Sans regular
    path, _ = urllib.request.urlretrieve(url)
    return path


def draw_text_pil(canvas: np.ndarray, text: str, x: int, y: int, w: int, h: int, font_size: int = 20):
    """Draw text on canvas using PIL."""
    img = PIL.Image.fromarray(canvas)
    draw = PIL.ImageDraw.Draw(img)
    font = PIL.ImageFont.truetype(get_ttf_font_path(), font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    xx = x + (w - bbox[0] - bbox[2]) // 2
    yy = y + (h - bbox[1] - bbox[3]) // 2
    draw.text((xx, yy), text, fill=(0, 0, 0), font=font)
    return np.array(img)


@dataclass
class GenerateConfig:
    """Configuration for image generation."""
    # Checkpoint path, or None to use the base model
    checkpoint: Optional[str] = None
    # Output path
    out: str = "out.jpg"
    # Random seed
    seed: int = 0
    # Number of images per prompt in grid
    grid_rows: int = 1
    grid_cols: int = 3

    # Pipeline parameters
    width: int = 512
    height: int = 512
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    batch_size: int = 4


def generate_prompt_grid(config: GenerateConfig):
    """Generate visualization grid from prompts."""

    cols = prompts.PROMPT_GRID
    all_prompts = [prompt.replace('\n', ' ') for col in cols for prompt in col[1:]
                   for _ in range(config.grid_cols * config.grid_rows)]

    print(f'Loading checkpoint: {config.checkpoint}')
    pipe = models.load_pipe(config.checkpoint)

    print(f'Generating {len(all_prompts)} images...')
    images = models.run_pipe(
        pipe,
        prompts=all_prompts,
        seed=config.seed,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
        batch_size=config.batch_size,
    )
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    # Layout parameters
    border = 2
    num_rows = max(len(col) - 1 for col in cols)
    image_h, image_w = config.height, config.width
    grid_w = (image_w + border) * config.grid_cols + border
    grid_h = (image_h + border) * config.grid_rows + border
    col_w = int(grid_w * 1.05 + 0.5)
    row_h = int(grid_h * 1.15 + 0.5)
    title_h = int(image_h * 0.13 + 0.5)
    prompt_y = int(grid_h * 1.02 + 0.5)

    # Create canvas
    canvas = np.full([title_h + row_h * num_rows, col_w * len(cols), 3], 255, dtype=np.uint8)

    # Paint grid
    image_iter = iter(images)
    for col_idx, col in enumerate(cols):
        # Draw column title
        canvas = draw_text_pil(canvas, col[0], x=col_w*col_idx, y=0, w=col_w, h=title_h,
                               font_size=int(image_h * 0.12))

        for row_idx, prompt in enumerate(col[1:]):
            grid_x = col_w * col_idx + (col_w - grid_w) // 2
            grid_y = title_h + row_h * row_idx

            for grid_row in range(config.grid_rows):
                for grid_col in range(config.grid_cols):
                    x = grid_x + (image_w + border) * grid_col + border
                    y = grid_y + (image_h + border) * grid_row + border
                    canvas[y - border : y + image_h + border, x - border : x + image_w + border] = 127
                    canvas[y : y + image_h, x : x + image_w] = next(image_iter)

            # Draw prompt text (truncated)
            prompt_display = prompt[:80] + '...' if len(prompt) > 80 else prompt
            canvas = draw_text_pil(canvas, prompt_display, x=col_w*col_idx, y=grid_y+prompt_y,
                                   w=col_w, h=row_h-prompt_y, font_size=int(image_h * 0.08))

    # Save
    save_image(config.out, canvas)
    print('Done.')
