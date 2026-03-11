# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

from typing import List
import warnings
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from fdfo.config import RewardSpec, VLMRewardPromptConfig
import transformers
from transformers import AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="The following layers were not sharded")
warnings.filterwarnings("ignore", message=".*image processor.*fast processor.*")
warnings.filterwarnings("ignore", message="Using a slow image processor")

class RewardEngine:

    def __init__(self, reward_specs: List[RewardSpec], device, vlm_model_name, use_fsdp: bool = False):
        self.reward_specs = reward_specs
        self.device = device
        self.use_fsdp = use_fsdp

        needs_vlm = any(s.kind == "vlm" for s in reward_specs)
        needs_pickscore = any(s.kind == "pickscore" for s in reward_specs)

        if needs_vlm:
            self.vlm, self.vlm_processor = self._init_vlm(vlm_model_name)

        if needs_pickscore:
            self.pickscore_processor, self.pickscore_model = self._init_pickscore()

    def _init_vlm(self, model_name: str):

        vlm_name_map = {
            "qwenvl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "qwenvl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        }
        assert model_name in vlm_name_map, f"VLM name {model_name} not found in {list(vlm_name_map.keys())}"
        model_path = vlm_name_map[model_name]

        # Load to CPU first when using FSDP, so FSDP can shard and move to GPU incrementally
        # without needing double the GPU memory during initialization
        load_device = "cpu" if self.use_fsdp else f"cuda:{torch.cuda.current_device()}"
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=load_device,
        )
        vlm.eval().requires_grad_(False)

        if self.use_fsdp:
            mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
            vlm = FSDP(vlm, sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=mixed_precision, auto_wrap_policy=size_based_auto_wrap_policy, device_id=torch.cuda.current_device())

        vlm_processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

        return vlm, vlm_processor

    def _init_pickscore(self):
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"

        pickscore_processor = AutoProcessor.from_pretrained(processor_path)
        pickscore_model = AutoModel.from_pretrained(model_path).eval().to(self.device, dtype=torch.float32)
        pickscore_model.eval().requires_grad_(False)

        return pickscore_processor, pickscore_model

    @torch.no_grad()
    def _compute_vlm_reward(
        self,
        prompts: List[str],
        images: torch.Tensor,
        vlm_config: VLMRewardPromptConfig,
    ) -> torch.Tensor:

        vocab = self.vlm_processor.tokenizer.vocab
        target, contrast = vlm_config.target_token, vlm_config.contrast_token

        qwen_image_size = 504
        qwen_template = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n'
        texts = [qwen_template.format(vlm_config.prompt_template.format(prompt=prompt)) for prompt in prompts]
        images = images.to(torch.float32)
        images = F.interpolate(images, size=(qwen_image_size, qwen_image_size), mode="bilinear", align_corners=False)
        inputs = self.vlm_processor(
            text=texts, images=images, padding=True, images_kwargs={"do_rescale": False, "do_resize": False}, return_tensors="pt"
        )
        outputs = self.vlm(**inputs.to(self.device, dtype=torch.bfloat16), return_dict=True)

        logits = outputs.logits[:, -1, :].to(torch.float32)
        scores = torch.sigmoid(logits[:, vocab[target]] - logits[:, vocab[contrast]])
        return scores * 100

    @torch.no_grad()
    def _compute_pickscore(
        self,
        prompts: List[str],
        images: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(torch.float32)
        image_inputs = self.pickscore_processor(
            images=images, padding=True, truncation=True, max_length=77, images_kwargs={"do_rescale": False}, return_tensors="pt"
        )

        # Fix for transformers returning list instead of tensor for pixel_values
        if "pixel_values" in image_inputs and isinstance(image_inputs["pixel_values"], list):
            image_inputs["pixel_values"] = torch.stack(image_inputs["pixel_values"])

        text_inputs = self.pickscore_processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )

        image_embs = self.pickscore_model.get_image_features(**image_inputs.to(self.device))
        image_embs = getattr(image_embs, 'pooler_output', image_embs) # transformers backwards compatibility
        image_embs = image_embs.to(torch.float32)
        image_embs = image_embs / torch.linalg.norm(image_embs, dim=1, keepdim=True)

        text_embs = self.pickscore_model.get_text_features(**text_inputs.to(self.device))
        text_embs = getattr(text_embs, 'pooler_output', text_embs) # transformers backwards compatibility
        text_embs = text_embs.to(torch.float32)
        text_embs = text_embs / torch.linalg.norm(text_embs, dim=1, keepdim=True)

        cosine_sim = torch.linalg.vecdot(text_embs, image_embs)
        logit_scale = self.pickscore_model.logit_scale.exp()
        return cosine_sim * logit_scale

    @torch.no_grad()
    def compute_weighted_rewards(
        self,
        prompts: List[str],
        images: torch.Tensor,
    ) -> torch.Tensor:
        weighted_sum = torch.zeros(len(images), dtype=torch.float32, device=self.device)

        for spec in self.reward_specs:
            if spec.kind == "pickscore":
                r = self._compute_pickscore(prompts, images)
            elif spec.kind == "vlm":
                r = self._compute_vlm_reward(prompts, images, spec.prompt_config)
            else:
                raise ValueError(f"Invalid reward kind: {spec.kind}")

            weighted_sum = weighted_sum + spec.weight * r.to(torch.float32)

        return weighted_sum
