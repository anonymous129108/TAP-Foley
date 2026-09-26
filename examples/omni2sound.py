"""Inference with the patched Omni2Sound API and upstream video features."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-feature", type=Path, required=True)
    parser.add_argument("--sync-feature", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    import torchaudio
    from pyscripts.generate_v2a_cond import load_model, generate_diffusion_cond

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.model_config.read_text())
    with torch.inference_mode():
        model = load_model(model_config=config, model_ckpt_path=str(args.checkpoint),
                           device=args.device, model_half=False, use_ema=False)
        audio = generate_diffusion_cond(
            model, sampling_method="tap-foley", source_prompt=args.source,
            conditioning=[{"task": "VTA", "clip_feature": str(args.clip_feature.resolve()),
                           "sync_feature": str(args.sync_feature.resolve()),
                           "seconds_start": 0.0, "seconds_total": args.duration,
                           "text_prompt": args.target}],
            steps=100, cfg_scale=2.5, tap_alpha=1.2,
            sample_size=config["sample_size"], seed=args.seed, device=args.device,
        ).float()
    if not torch.isfinite(audio).all() or audio.abs().max() <= 1e-8:
        raise RuntimeError("Non-finite or silent decoded audio")
    # Same peak normalization as the research Omni2Sound output writer.
    audio = audio / audio.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    audio = audio[..., :round(args.duration * config["sample_rate"])].clamp(-1, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), audio[0].cpu(), config["sample_rate"])


if __name__ == "__main__":
    main()
