"""Omni2Sound adapter with training-aligned TA/VTA/CFG-dropout branches."""
import math
import torch
from .projection import guided_fields, project


def build_branches(model, source_conditioning, target_text_condition):
    required = {"clip_feature", "sync_feature", "text_prompt", "seconds_start", "seconds_total"}
    if not required <= source_conditioning.keys():
        raise ValueError(f"Missing conditions: {sorted(required - source_conditioning.keys())}")
    source = source_conditioning
    duration = {k: source[k] for k in ("seconds_start", "seconds_total")}
    sync, mask = source["sync_feature"]
    zero_sync = [torch.zeros_like(sync), mask]
    empty = dict(model.get_conditioning_inputs(source))
    for key in ("cross_attn_cond", "additional_cond", "prepend_cond"):
        if isinstance(empty.get(key), torch.Tensor):
            empty[key] = torch.zeros_like(empty[key])
    return {
        "empty": empty,
        "source": model.get_conditioning_inputs({**duration, "sync_feature": zero_sync,
                                                  "text_prompt": source["text_prompt"]}),
        "target": model.get_conditioning_inputs({**duration, "sync_feature": zero_sync,
                                                  "text_prompt": target_text_condition}),
        "video_target": model.get_conditioning_inputs({**source, "text_prompt": target_text_condition}),
    }


def _cast(value, noise):
    if isinstance(value, torch.Tensor):
        return value.to(device=noise.device, dtype=noise.dtype if value.is_floating_point() else value.dtype)
    if isinstance(value, dict):
        return {k: _cast(v, noise) for k, v in value.items()}
    return value


def forward(model_fn, latent, time, branches, *, cfg=2.5, alpha=1.2):
    time = torch.as_tensor(time, device=latent.device, dtype=latent.dtype).expand(len(latent))
    raw = {key: model_fn(latent, time, **branches[key], cfg_scale=1.0,
                         batch_cfg=True, scale_phi=0.0)
           for key in ("empty", "video_target", "source", "target")}
    return project(*guided_fields(raw, cfg), alpha=alpha).to(latent.dtype)


@torch.inference_mode()
def sample(model_fn, noise, branches, *, steps=100, cfg=2.5, alpha=1.2):
    """Deterministic frozen-field v-ODE rotation, t=1 -> 0 (no SDE noise)."""
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    latent = noise.clone()
    branches = _cast(branches, noise)
    times = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=noise.dtype)
    for step in range(steps):
        field = forward(model_fn, latent, times[step], branches, cfg=cfg, alpha=alpha)
        theta, following = times[step] * (math.pi / 2), times[step + 1] * (math.pi / 2)
        data = latent * theta.cos() - field * theta.sin()
        epsilon = latent * theta.sin() + field * theta.cos()
        latent = data * following.cos() + epsilon * following.sin()
    return latent


@torch.inference_mode()
def generate(model, *, conditioning=None, conditioning_tensors=None,
             source_prompt=None, source_text_condition=None, steps=100, cfg_scale=2.5,
             sample_size=2097152, seed=42, device="cuda", return_latents=False, alpha=1.2):
    """conditioning carries target text and video; source_prompt supplies its source."""
    if model.diffusion_objective != "v":
        raise ValueError("TAP-Foley expects Omni2Sound v-prediction")
    if conditioning_tensors is None:
        if conditioning is None:
            raise ValueError("conditioning or conditioning_tensors is required")
        conditioning_tensors = model.conditioner(conditioning, device)
    target_condition = conditioning_tensors["text_prompt"]
    batch = target_condition[0].shape[0]
    if source_text_condition is None:
        prompts = [source_prompt] * batch if isinstance(source_prompt, str) else source_prompt
        if (not prompts or len(prompts) != batch
                or any(not isinstance(s, str) or not s.strip() for s in prompts)):
            raise ValueError("source_prompt must provide one nonempty string per sample")
        source_text_condition = model.conditioner.conditioners["text_prompt"](prompts, device)
    if source_text_condition[0].shape[0] != batch:
        raise ValueError("Source and target batch sizes differ")
    source = {**conditioning_tensors, "text_prompt": source_text_condition}
    branches = build_branches(model, source, target_condition)
    dtype = next(model.model.parameters()).dtype
    ratio = model.pretransform.downsampling_ratio if model.pretransform is not None else 1
    if sample_size <= 0 or sample_size % ratio:
        raise ValueError("sample_size must be positive and divisible by the latent downsampling ratio")
    rng = torch.Generator(device=device)
    rng.seed() if seed == -1 else rng.manual_seed(seed)
    noise = torch.randn(batch, model.io_channels, sample_size // ratio,
                        device=device, dtype=dtype, generator=rng)
    latent = sample(model.model, noise, branches, steps=steps, cfg=cfg_scale, alpha=alpha)
    if return_latents or model.pretransform is None:
        return latent
    return model.pretransform.decode(latent.to(next(model.pretransform.parameters()).dtype))
