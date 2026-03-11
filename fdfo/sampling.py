# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from fdfo.noise_schedules import Scheduler
from fdfo.config import FDFOConfig

def edm_step(flow_schedule: FlowMatchEulerDiscreteScheduler, v: torch.Tensor, timestep: float, sample: torch.Tensor, gamma: float):

    step_idx = flow_schedule.index_for_timestep(timestep)
    t = flow_schedule.sigmas[step_idx].item() # Current timestep
    s = flow_schedule.sigmas[step_idx + 1].item() # Desired next timestep
    r = s / (-gamma*s + gamma + 1) # Overshoot timestep
    B = lambda x: x[:, None, None, None] # Broadcast to (-1, 1, 1, 1)

    overshoot_sample = sample + B(r - t) * v # Euler step which overshoots to timestep r
    new_noise_scale = r*((gamma + 1)**2 - 1) ** 0.5 / (gamma*r + 1) # Scale of fresh noise
    scale = 1 / (gamma*r + 1) # Scale down overshoot sample

    prev_sample = B(scale) * overshoot_sample + B(new_noise_scale) * torch.randn_like(overshoot_sample)
    return prev_sample

@torch.no_grad()
def sample_image_pairs(pipeline, churn_scheduler: Scheduler, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, config: FDFOConfig):

    batch_size = prompt_embeds.shape[0]
    do_cfg = abs(config.sample.guidance_scale - 1) > 1e-6
    height, width = config.sample.resolution, config.sample.resolution
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    x = pipeline.prepare_latents(batch_size, pipeline.transformer.config.in_channels, height, width, prompt_embeds.dtype, device, None)
    timesteps, _ = retrieve_timesteps(pipeline.scheduler, config.sample.num_solver_steps, device)

    def sample_churn_pdf():
        churn_density, _ = churn_scheduler.sample_densities()
        pdf = churn_density.pdf_discretized(ts=(timesteps / 1000), churn_scaling=True)
        return torch.cat([pdf, pdf.new_zeros(1)])

    churn_pdf = torch.stack([sample_churn_pdf() for _ in range(batch_size)]).to(device)

    # Repeat interleaved for paired sampling trajectories
    repeat = lambda x: x.repeat_interleave(2, dim=0)
    x = repeat(x)
    churn_pdf = repeat(churn_pdf)
    prompt_embeds = repeat(prompt_embeds)
    pooled_prompt_embeds = repeat(pooled_prompt_embeds)
    negative_prompt_embeds = repeat(negative_prompt_embeds)
    negative_pooled_prompt_embeds = repeat(negative_pooled_prompt_embeds)

    prior_churn, churns = churn_pdf[:, 0], churn_pdf[:, 1:]

    # Churn initial noise sample
    B = lambda x: x[:, None, None, None]
    scale = 1 / (prior_churn + 1)
    new_noise_scale = ((prior_churn + 1)**2 - 1) ** 0.5 / (prior_churn + 1)
    x = B(scale) * x + B(new_noise_scale) * torch.randn_like(x)

    xs = [x]
    velocities = []

    for t, churn_scale in zip(timesteps, churns.unbind(dim=1)):

        if do_cfg:
            v_neg, v_pos = pipeline.transformer(hidden_states=torch.cat([x] * 2), timestep=t.expand(batch_size * 4), encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds).sample.chunk(2)
            v = v_neg + config.sample.guidance_scale * (v_pos - v_neg)
        else:
            v = pipeline.transformer(hidden_states=x,timestep=t.expand(batch_size * 2), encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds).sample

        x = edm_step(pipeline.scheduler, v.float(), t, x.float(), churn_scale)
        xs.append(x)
        velocities.append(v)

    pipeline.vae.use_slicing = True
    x = (x / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(x, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pt")

    return image, xs, velocities
