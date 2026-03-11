# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

"""
Model loading and evaluation utilities for FDFO checkpoints.

Supports loading SD3.5 with LoRA weights, and evaluation models (CLIP, VLM, DreamSim).
"""

import os
import re
import copy
import warnings
import functools
import contextlib
import numpy as np
import torch
import torch.distributed.fsdp
import PIL.Image
import huggingface_hub
import transformers
import diffusers
import peft
diffusers.logging.set_verbosity_error()


@contextlib.contextmanager
def rank0_first():
    """Context manager that ensures rank 0 executes first in distributed setting."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if rank == 0:
        yield
    if world_size > 1:
        torch.distributed.barrier()
    if rank != 0:
        yield
    if world_size > 1:
        torch.distributed.barrier()


@functools.lru_cache(None)
def load_orig_pipe(repo='stabilityai/stable-diffusion-3.5-medium', dtype=torch.bfloat16, device=torch.device('cuda')):
    return diffusers.StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=dtype).to(device)


def load_pipe(checkpoint=None, dtype=torch.bfloat16, device=torch.device('cuda')):
    """Load SD3.5-medium pipeline, optionally with LoRA weights from checkpoint."""

    # If the checkpoint is a HuggingFace URL, download it via ``huggingface_hub.snapshot_download()``.
    m = re.fullmatch(r"(hf:|https://huggingface.co/)([^/]+/[^/]+)(/tree/([^/]+))?(.*)", checkpoint) if checkpoint else None
    if m:
        repo_id = m.group(2)
        revision = m.group(4) or "main"
        subdir = m.group(5).strip("/")
        allow_patterns = [f"{subdir}/**"] if subdir else None
        root = huggingface_hub.snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        checkpoint = os.path.join(root, *subdir.split("/"))

    # Load original pipeline.
    orig_pipe = load_orig_pipe(dtype=dtype, device=device)
    if checkpoint is None:
        return orig_pipe

    # Take a copy of the pipeline.
    lora_pipe = copy.copy(orig_pipe)
    lora_pipe.orig_transformer = lora_pipe.transformer
    lora_pipe.transformer = copy.deepcopy(lora_pipe.transformer)

    # Flow-GRPO original implementation.
    lora_pipe.transformer = peft.PeftModel.from_pretrained(lora_pipe.transformer, checkpoint).eval()
    return lora_pipe


@torch.no_grad()
def run_pipe(pipe, prompts, seed=0, num_inference_steps=30, guidance_scale=4.5, width=512, height=512, batch_size=16, **pipe_kwargs):
    if isinstance(seed, int):
        seed = range(seed, seed + len(prompts))
    assert len(seed) == len(prompts)
    pipe.set_progress_bar_config(disable=True)

    images = []
    for batch in np.array_split(np.arange(len(prompts)), max((len(prompts) - 1) // batch_size + 1, 1)):
        batch_prompts = [prompts[idx] for idx in batch]
        batch_generators = [torch.Generator().manual_seed(int(seed[idx]) % (1 << 31)) for idx in batch]
        out = pipe(prompt=batch_prompts, generator=batch_generators, output_type='pt',
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width, height=height,
            **pipe_kwargs)
        images.append((out.images * 255 + 0.5).clip(0, 255).to(torch.uint8))
    return torch.cat(images)


class VLM:
    """Vision-Language Model for yes/no question scoring (e.g., prompt adherence)."""

    MODEL_SPECS = {
        'Qwen/Qwen2.5-VL-7B-Instruct':  dict(dtype=torch.bfloat16, fsdp=False),
        'Qwen/Qwen2.5-VL-72B-Instruct': dict(dtype=torch.bfloat16, fsdp=True),
    }
    CHAT_TEMPLATE = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n'

    def __init__(self, name='Qwen/Qwen2.5-VL-7B-Instruct', device=torch.device('cuda')):
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.name = name
        self.device = device
        self.spec = self.MODEL_SPECS[name]

        # Load processor
        self.processor = transformers.AutoProcessor.from_pretrained(name)
        self.processor.tokenizer.padding_side = 'left'
        self.vocab = self.processor.tokenizer.vocab

        # Load model
        load_device = 'cpu' if self.spec['fsdp'] else device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name,
            torch_dtype=self.spec['dtype'],
            device_map=load_device if load_device == 'cpu' else None,
        )
        if load_device != 'cpu':
            self.model = self.model.to(device)

        # Wrap with FSDP for large models
        if self.spec['fsdp']:
            self.model = torch.distributed.fsdp.FullyShardedDataParallel(
                self.model,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
                mixed_precision=torch.distributed.fsdp.MixedPrecision(
                    param_dtype=self.spec['dtype'], 
                    buffer_dtype=self.spec['dtype']
                ),
                auto_wrap_policy=torch.distributed.fsdp.wrap.size_based_auto_wrap_policy,
                device_id=torch.cuda.current_device(),
            )

    def calc_yes_no_scores(self, texts, images):
        """Calculate yes/no scores for given texts and images."""
        texts = [self.CHAT_TEMPLATE.format(text) for text in texts]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors='pt')
        outputs = self.model(**inputs.to(self.device), return_dict=True)
        logits = torch.stack([xo[len(xi) - 1] for xi, xo in zip(inputs.input_ids, outputs.logits)]).to(torch.float32)
        scores = torch.sigmoid(logits[:, self.vocab['Yes']] - logits[:, self.vocab['No']])
        return scores * 100


class CLIP:
    """CLIP-based scoring models (CLIP, PickScore, HPSv2)."""

    MODEL_SPECS = {
        'laion/CLIP-ViT-H-14-laion2B-s32B-b79K':    dict(dtype=torch.float32, max_length=77),
        'openai/clip-vit-large-patch14':            dict(dtype=torch.float32, max_length=77),
        'yuvalkirstain/PickScore_v1':               dict(dtype=torch.float32, max_length=77),
        'adams-story/HPSv2-hf':                     dict(dtype=torch.float32, max_length=77),
    }

    def __init__(self, name='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', device=torch.device('cuda')):
        self.name = name
        self.device = device
        self.spec = self.MODEL_SPECS[name]

        # Load model and processor
        self.model = transformers.CLIPModel.from_pretrained(name, torch_dtype=self.spec['dtype']).to(device)
        self.processor = transformers.CLIPProcessor.from_pretrained(name)

    def calc_text_embs(self, texts):
        """Calculate normalized text embeddings."""
        inputs = self.processor(text=texts, padding=True, truncation=True, max_length=self.spec['max_length'], return_tensors='pt')
        text_embs = self.model.get_text_features(**inputs.to(self.device))
        text_embs = getattr(text_embs, 'pooler_output', text_embs) # transformers backwards compatibility
        text_embs = text_embs.to(torch.float32)
        return text_embs / torch.linalg.norm(text_embs, dim=1, keepdim=True)

    def calc_image_embs(self, images):
        """Calculate normalized image embeddings."""
        inputs = self.processor(images=images, return_tensors='pt')
        image_embs = self.model.get_image_features(**inputs.to(self.device))
        image_embs = getattr(image_embs, 'pooler_output', image_embs) # transformers backwards compatibility
        image_embs = image_embs.to(torch.float32)
        return image_embs / torch.linalg.norm(image_embs, dim=1, keepdim=True)

    def calc_scores(self, texts, images):
        """Calculate text-image similarity scores (scaled by logit_scale)."""
        cosine_sim = torch.linalg.vecdot(self.calc_text_embs(texts), self.calc_image_embs(images))
        logit_scale = self.model.logit_scale.exp()
        return cosine_sim * logit_scale


class DreamSim:
    """DreamSim perceptual distance model for diversity metrics."""

    def __init__(self, device=torch.device('cuda')):
        import dreamsim # pip install dreamsim
        warnings.filterwarnings('ignore', re.escape('`torch.nn.utils.weight_norm` is deprecated'))
        warnings.filterwarnings('ignore', re.escape('Already found a `peft_config` attribute in the model.'))
        self.device = device
        with rank0_first(): # avoid race conditions in distributed setting
            self.model, self.preprocess = dreamsim.dreamsim(pretrained=True, device=device)

    def calc_image_embs(self, images):
        """Calculate DreamSim embeddings for images."""
        images = [img if isinstance(img, PIL.Image.Image) else PIL.Image.fromarray(img.cpu().numpy().transpose(1, 2, 0), 'RGB') for img in images]
        images = torch.cat([self.preprocess(img) for img in images]).to(self.device)
        return self.model.embed(images)

    def calc_distances(self, img1, img2):
        """Calculate perceptual distances between image pairs."""
        return 1 - torch.linalg.vecdot(self.calc_image_embs(img1), self.calc_image_embs(img2))
