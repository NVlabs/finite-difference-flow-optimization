# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

import os
import datetime
import contextlib
import torch
import torch.nn.parallel
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import diffusers
diffusers.logging.set_verbosity_error()
from fdfo.prompts import PromptDataLoader
from fdfo.noise_schedules import build_churn_scheduler
from fdfo.sampling import sample_image_pairs
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from fdfo.rewards import RewardEngine
from fdfo.logging import TrainingLogger, RolloutLogData, TrainingStepMetrics, MetricsAccumulator
from fdfo.utils import init, get_rank, get_world_size, barrier, print0, deinit, CheckpointIO, set_random_seed, VRAMProfiler

def predict_v(transformer, sample, embeds, pooled_embeds, config):
    guidance_scale = config.sample.guidance_scale
    do_cfg = abs(guidance_scale - 1) > 1e-6
    if do_cfg:
        v_neg, v_pos = transformer(hidden_states=torch.cat([sample["xs"]] * 2), timestep=torch.cat([sample["timesteps"]] * 2), encoder_hidden_states=embeds, pooled_projections=pooled_embeds).sample.chunk(2)
        v = v_neg + guidance_scale * (v_pos - v_neg)
    else:
        v = transformer(hidden_states=sample["xs"], timestep=sample["timesteps"], encoder_hidden_states=embeds, pooled_projections=pooled_embeds).sample
    return v

def train_fdfo(config, run_dir: str, run_id: int):
    init()
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    vram = VRAMProfiler(os.path.join(run_dir, "vram_profile.jsonl"), device)

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    logger = TrainingLogger(
        run_dir=run_dir,
        run_name=f"{run_id:05d}-{config.run_name}",
        config=config,
        rank=get_rank(),
        world_size=get_world_size(),
    )

    set_random_seed(config.seed, get_rank())

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained_model, torch_dtype=torch.bfloat16)
    for module in [pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]:
        module.requires_grad_(False)
        module.to(device, dtype=torch.bfloat16)
    pipeline.safety_checker = None
    pipeline.transformer.to(device)

    vram.checkpoint("After model + text encoders loaded")

    target_modules = ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out", "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"]
    transformer_lora_config = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, init_lora_weights="gaussian", target_modules=target_modules)
    pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    transformer = pipeline.transformer

    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    wrapped_model = torch.nn.parallel.DistributedDataParallel(pipeline.transformer, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(transformer_trainable_parameters, lr=config.train.learning_rate, betas=(config.train.adam_beta1, config.train.adam_beta2), weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)
    scaler = GradScaler()

    vram.checkpoint("After LoRA + DDP + optimizer")

    from fdfo.config import get_reward_preset
    reward_specs = config.reward_specs if config.reward_specs else get_reward_preset(config.reward_preset)
    reward_engine = RewardEngine(reward_specs, device, config.vlm_model_name, use_fsdp=config.vlm_use_fsdp)
    vram.checkpoint("After reward engine loaded")

    churn_scheduler = build_churn_scheduler(config.churn_schedule)
    prompt_dataloader = PromptDataLoader(config=config, shuffle=True, seed=config.seed + get_rank())

    # Each pair produces 2 samples (two perturbed branches).
    pairs_per_gpu = config.sample.batch_size * config.sample.num_batches_per_epoch
    samples_per_gpu = pairs_per_gpu * 2
    samples_per_step_per_gpu = config.train.batch_size * config.train.gradient_accumulation_steps
    assert samples_per_gpu >= samples_per_step_per_gpu, "Total samples per GPU must be >= training batch size per GPU"
    assert samples_per_gpu % samples_per_step_per_gpu == 0, "Total samples per GPU must divide evenly into training batches"

    state = dict(cur_epoch=0, global_step=0, total_elapsed_time=0)
    checkpoint = CheckpointIO(state=state, optimizer=optimizer)

    if config.resume_from:
        print0(f"Resuming from {config.resume_from}")
        checkpoint_dir = config.resume_from

        training_state_path = os.path.join(checkpoint_dir, 'training-state.pt')
        if os.path.exists(training_state_path):
            checkpoint.load(training_state_path)

        adapter_path = os.path.join(checkpoint_dir, 'adapter_model.safetensors')
        if os.path.exists(adapter_path):
            lora_state_dict = load_safetensors(adapter_path)
            set_peft_model_state_dict(transformer, lora_state_dict)
            for param in transformer.parameters():
                if param.requires_grad:
                    param.data = param.data.float()
            print0(f"Loaded PEFT adapter from {adapter_path}")

    first_epoch = state['cur_epoch']
    global_step = state['global_step']

    neg_prompt_embed, _, neg_pooled_prompt_embed, _ = pipeline.encode_prompt([""], None, None, do_classifier_free_guidance=False, device=device, max_sequence_length=128)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]

    for epoch in range(first_epoch, config.num_epochs):
        state['cur_epoch'] = epoch
        vram.reset_peak()

        if config.memory_efficient and epoch > first_epoch: # Move text encoders to VRAM
            for enc in text_encoders:
                enc.to(device)

        pipeline.transformer.eval()
        samples = []
        all_images = []
        all_prompts = []

        for batch_idx in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling", disable=(get_rank() != 0)):
            prompts = prompt_dataloader.get_batch(config.sample.batch_size)
            prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(prompts, None, None, do_classifier_free_guidance=False, device=device, max_sequence_length=128)

            with autocast('cuda'):
                images, xs, velocities = sample_image_pairs(
                    pipeline,
                    churn_scheduler,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embed.repeat(config.sample.batch_size, 1, 1),
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embed.repeat(config.sample.batch_size, 1),
                    config=config,
                )

            xs = torch.stack(xs, dim=1)
            velocities = torch.stack(velocities, dim=1)

            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)
            expanded_prompts = [p for p in prompts for _ in range(2)]
            rewards = reward_engine.compute_weighted_rewards(expanded_prompts, images)

            all_images.extend(images)
            all_prompts.extend(prompts)

            rewards_branch_1, rewards_branch_2 = rewards[0::2], rewards[1::2]
            reward_diff = (rewards_branch_1 - rewards_branch_2).float() / 2
            dR = torch.stack((reward_diff, -reward_diff), dim=1).reshape(-1)

            x_0_branch_1 = xs[0::2, -1].float()
            x_0_branch_2 = xs[1::2, -1].float()
            endpoint_difference = x_0_branch_2 - x_0_branch_1
            dx_dR = torch.stack((-endpoint_difference, endpoint_difference), dim=1).reshape(-1, *endpoint_difference.shape[1:]) / 2

            repeat = lambda x: x.repeat_interleave(2, dim=0)
            samples.append(
                {
                    "prompt_embeds": repeat(prompt_embeds),
                    "pooled_prompt_embeds": repeat(pooled_prompt_embeds),
                    "timesteps": repeat(timesteps),
                    "xs": xs[:, :-1],
                    "velocities": velocities,
                    "rewards": rewards,
                    "dR": dR,
                    "dx_dR": dx_dR,
                }
            )

            if batch_idx == 0:
                vram.checkpoint(f"Epoch {epoch}: after 1st sample batch + rewards")

        vram.checkpoint(f"Epoch {epoch}: after all {config.sample.num_batches_per_epoch} sample batches accumulated")
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        rollout_log_data = RolloutLogData(epoch=epoch, rewards=samples["rewards"], images=all_images, prompts=all_prompts)
        logger.log_rollout(rollout_log_data, global_step)
        logger.log_images(rollout_log_data, global_step, config.sample.resolution)

        if config.memory_efficient: # Offload text encoders to main memory
            for enc in text_encoders:
                enc.to('cpu')
            torch.cuda.empty_cache()
            vram.checkpoint(f"Epoch {epoch}: after text encoder offload")

        num_timesteps = config.sample.num_solver_steps

        # Flatten, permute globally, and batch samples for training
        local_size = num_timesteps * config.sample.batch_size * config.sample.num_batches_per_epoch * 2
        world_size = get_world_size()

        perm_generator = torch.Generator().manual_seed(config.seed + epoch * 12345) # Matches across ranks
        global_perm = torch.randperm(local_size * world_size, generator=perm_generator)
        rank = get_rank()
        local_indices = global_perm[rank * local_size : (rank + 1) * local_size]

        def all_gather_cat(x: torch.Tensor) -> torch.Tensor:
            ws = get_world_size()
            out = torch.empty((ws * x.shape[0],) + x.shape[1:], device=x.device, dtype=x.dtype)
            torch.distributed.all_gather_into_tensor(out, x.contiguous())
            return out

        local_indices = local_indices.to(device=device)
        base_indices = torch.div(local_indices, num_timesteps, rounding_mode="floor")

        # Batch time-dependent tensors as (xs, velocities, timesteps).
        # Keep non-time-dependent tensors (prompt_embeds, dx_dR, etc.) in compact form to avoid T-fold expansion.
        samples_batched_td = {}
        compact_data = {}
        for k, v in samples.items():
            v = v.to(device=device)

            is_time_dependent = (v.ndim >= 2 and v.shape[1] == num_timesteps)
            if is_time_dependent: # v: [B, T, ...] -> flat [B*T, ...]
                v_flat = v.flatten(0, 1)
                v_global = all_gather_cat(v_flat) # [global_B*T, ...]
                v_shuffled = v_global.index_select(0, local_indices)
                samples_batched_td[k] = v_shuffled.reshape(-1, config.train.batch_size, *v_shuffled.shape[1:])
                del v_flat, v_global, v_shuffled
            else: # v: [B, ...] - store compactly
                v_global = all_gather_cat(v) # [global_B, ...]
                compact_data[k] = v_global
            del v

        compact_indices = base_indices.reshape(-1, config.train.batch_size)
        samples_batched = [dict(zip(samples_batched_td, x)) for x in zip(*samples_batched_td.values())]
        for i, mb in enumerate(samples_batched):
            mb["_compact_idx"] = compact_indices[i]

        del samples, samples_batched_td  # Free original samples - no longer needed after batching
        torch.cuda.empty_cache()
        vram.checkpoint(f"Epoch {epoch}: after all-gather + batching (samples freed)")

        pipeline.transformer.train()
        metrics_acc = MetricsAccumulator()
        accumulation_counter = 0
        local_num_accumulation_rounds = num_timesteps * config.train.gradient_accumulation_steps

        # Training loop
        print0(f"Epoch {epoch} training: {len(samples_batched)} samples")
        for sample_td in tqdm(samples_batched, desc=f"Epoch {epoch} training", disable=get_rank() != 0):
            idx = sample_td["_compact_idx"]
            sample = {k: v for k, v in sample_td.items() if k != "_compact_idx"}
            for k, v in compact_data.items():
                sample[k] = v[idx]

            do_cfg = abs(config.sample.guidance_scale - 1) > 1e-6
            if do_cfg:
                embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                pooled_embeds = torch.cat([train_neg_pooled_prompt_embeds, sample["pooled_prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]
                pooled_embeds = sample["pooled_prompt_embeds"]

            accumulation_counter += 1
            is_accumulation_boundary = (accumulation_counter % local_num_accumulation_rounds == 0)

            with contextlib.nullcontext() if is_accumulation_boundary else wrapped_model.no_sync():
                with autocast('cuda'):
                    v = predict_v(wrapped_model, sample, embeds, pooled_embeds, config)
                    v_ref, dx_dR = sample["velocities"], sample["dx_dR"]
                    ms_norm = torch.sqrt(torch.mean(dx_dR ** 2, dim=tuple(range(1, dx_dR.ndim)), keepdim=True)) ** 2.0
                    dx_dR = dx_dR / (ms_norm + 1e-6)

                    v_target = (v_ref - dx_dR).detach()
                    ps_loss = -(v - v_target) ** 2
                    ps_loss = ps_loss.mean(dim=tuple(range(1, ps_loss.ndim)))
                    ps_loss_ref = -(dx_dR ** 2).mean(dim=tuple(range(1, dx_dR.ndim)))

                    if config.train.kl_weight > 0: # KL regularization w.r.t. base model
                        with torch.no_grad():
                            with transformer.disable_adapter():
                                v_base = predict_v(transformer, sample, embeds, pooled_embeds, config)

                dR = sample["dR"]
                ratio = torch.exp(ps_loss - ps_loss_ref)
                policy_loss = -(dR * ratio) + (dR.abs() / (2.0 * config.train.clip_range)) * ((ratio - 1.0) ** 2) # SPO surrogate objective
                loss = torch.mean(policy_loss)

                if config.train.kl_weight > 0:
                    loss += config.train.kl_weight * ((v - v_base) ** 2).mean()

                metrics_acc.add(policy_loss=policy_loss.mean(), loss=loss, clipfrac=torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                scaled_loss = loss / local_num_accumulation_rounds
                scaler.scale(scaled_loss).backward()

                if accumulation_counter == 1:
                    vram.checkpoint(f"Epoch {epoch}: after 1st training fwd+bwd")

            if is_accumulation_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), config.train.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                reduced_metrics = metrics_acc.reduce_and_clear(get_world_size())
                step_metrics = TrainingStepMetrics(clipfrac=reduced_metrics["clipfrac"], policy_loss=reduced_metrics["policy_loss"], loss=reduced_metrics["loss"], epoch=epoch)
                logger.log_training_step(step_metrics, global_step)

                global_step += 1
                state['global_step'] = global_step

        # Dereference variables from the training loop to allow tensors to be garbage collected.
        del sample, sample_td, samples_batched, compact_data
        del v_ref, dR, embeds, pooled_embeds, v, v_target, ps_loss, ps_loss_ref, ratio, policy_loss, loss, scaled_loss, ms_norm, dx_dR
        if config.train.kl_weight > 0:
            del v_base
        torch.cuda.empty_cache()

        vram.checkpoint(f"Epoch {epoch}: end (after cleanup)")

        if (epoch + 1) % config.save_freq == 0:
            checkpoint_name = f'epoch-{epoch+1:07d}' # how many epochs have been executed
            checkpoint_dir = os.path.join(run_dir, config.checkpoint_subdir, checkpoint_name)
            if get_rank() == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                transformer.save_pretrained(checkpoint_dir)
                print0(f"Saved PEFT checkpoint to {checkpoint_dir}")
            barrier()

            checkpoint_path = os.path.join(checkpoint_dir, 'training-state.pt')
            checkpoint.save(checkpoint_path)

    deinit()
