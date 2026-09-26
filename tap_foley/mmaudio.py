"""MMAudio adapter; uses learned null video sequences for text-only fields."""
import torch
from .projection import guided_fields, project


def forward(net, latent, time, branches, *, cfg=4.5, alpha=1.2):
    """Four backbone forwards at the same latent/time, then TAP projection."""
    time = torch.as_tensor(time, device=latent.device, dtype=latent.dtype).expand(len(latent))
    raw = {key: net.predict_flow(latent, time, branches[key])
           for key in ("empty", "video_target", "source", "target")}
    return project(*guided_fields(raw, cfg, round_guided=True), alpha=alpha,
                   correction_roundtrip=True)


@torch.inference_mode()
def sample(net, noise, branches, *, steps=25, cfg=4.5, alpha=1.2):
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    latent = noise.clone()
    for step in range(steps):
        field = forward(net, latent, step / steps, branches, cfg=cfg, alpha=alpha)
        latent = latent + field / steps
    return latent


@torch.inference_mode()
def generate(clip_video, sync_video, text, *, source_text, feature_utils, net, fm,
             rng, cfg_strength=4.5, alpha=1.2, negative_text=None,
             clip_batch_size_multiplier=40, sync_batch_size_multiplier=40,
             image_input=False):
    if not text or source_text is None or len(source_text) != len(text):
        raise ValueError("source_text and target text must have equal nonzero batch lengths")
    if any(not isinstance(s, str) or not s.strip() for s in [*text, *source_text]):
        raise ValueError("Source and target prompts must be nonempty strings")
    if negative_text is not None and any(negative_text):
        raise ValueError("TAP-Foley uses source_text; negative_text must be empty")
    if clip_video is None or sync_video is None or image_input:
        raise ValueError("TAP-Foley requires both video streams (not image-only input)")
    if fm.inference_mode != "euler" or fm.min_sigma != 0:
        raise ValueError("TAP-Foley requires deterministic Euler with min_sigma=0")
    bs = len(text)
    device, dtype = feature_utils.device, feature_utils.dtype
    clip = feature_utils.encode_video_with_clip(
        clip_video.to(device, dtype), batch_size=bs * clip_batch_size_multiplier)
    sync = feature_utils.encode_video_with_sync(
        sync_video.to(device, dtype), batch_size=bs * sync_batch_size_multiplier)
    target = feature_utils.encode_text(text).to(device, dtype)
    source = feature_utils.encode_text(source_text).to(device, dtype)
    null_clip, null_sync = net.get_empty_clip_sequence(bs), net.get_empty_sync_sequence(bs)
    branches = {
        "empty": net.get_empty_conditions(bs),
        "video_target": net.preprocess_conditions(clip, sync, target),
        "source": net.preprocess_conditions(null_clip, null_sync, source),
        "target": net.preprocess_conditions(null_clip, null_sync, target),
    }
    noise = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                        generator=rng, device=device, dtype=dtype)
    latent = sample(net, noise, branches, steps=fm.num_steps, cfg=cfg_strength, alpha=alpha)
    return feature_utils.vocode(feature_utils.decode(net.unnormalize(latent)))
