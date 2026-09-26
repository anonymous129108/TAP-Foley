"""Per-example target-anchored half-space projection, reduced in FP32."""
import math
import torch


def project(video_target, source, target, *, alpha=1.2, correction_roundtrip=False):
    """Return V_VT - alpha * max(<V_VT-V_0T,d>/||d||²,0) * d.

    d = V_0S - V_0T. All dimensions except batch are reduced together.
    A zero direction is an exact no-op. Alpha is constant across sampling steps.
    """
    if not math.isfinite(alpha) or alpha < 1:
        raise ValueError("alpha must be finite and >= 1")
    if video_target.ndim < 2 or not video_target.is_floating_point():
        raise ValueError("Expected a floating-point field with a batch dimension")
    for field in (source, target):
        if (field.shape != video_target.shape or field.device != video_target.device
                or field.dtype != video_target.dtype):
            raise ValueError("All fields must share shape, device and dtype")
    base, src, anchor = video_target.float(), source.float(), target.float()
    if not all(torch.isfinite(v).all() for v in (base, src, anchor)):
        raise ValueError("TAP fields must be finite")
    direction = src - anchor
    norm_sq = direction.square().flatten(1).sum(1)
    valid = torch.isfinite(norm_sq) & (norm_sq > 0)
    shape = (-1,) + (1,) * (base.ndim - 1)
    direction = torch.where(valid.reshape(shape), direction, 0)
    denominator = torch.where(valid, norm_sq, 1)
    coordinate = ((base - anchor) * direction).flatten(1).sum(1) / denominator
    weight = torch.where(valid & torch.isfinite(coordinate), coordinate.clamp_min(0), 0)
    if correction_roundtrip:
        # Match MMAudio's established FP32 operation order: first project, then
        # recover and scale the correction. This affects final-bit rounding.
        nearest = base - weight.reshape(shape) * direction
        correction = alpha * (base - nearest)
    else:
        correction = (alpha * weight).reshape(shape) * direction
    result = (base - correction).to(video_target.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite projected field")
    return result


def guided_fields(raw, cfg, *, round_guided=False):
    if not math.isfinite(cfg) or cfg <= 0:
        raise ValueError("cfg must be finite and positive")
    empty = raw["empty"].float()
    fields = [empty + cfg * (raw[key].float() - empty)
              for key in ("video_target", "source", "target")]
    # MMAudio's maintained sampler publishes each guided field in model dtype.
    # Omni2Sound retains FP32 through CFG and projection.
    return [value.to(raw["empty"].dtype) if round_guided else value for value in fields]
