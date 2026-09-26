# TAP-Foley

Target-Anchored Projection for counterfactual video Foley generation.
Given a silent video, a source sound caption and a conflicting target caption,
TAP-Foley corrects the guided video–target field at every sampling step.

[Demo](https://anonymous129108.github.io/TAP-Foley/) ·
[Video/source/10-target benchmark](data/README.md)

## Method

```text
V_00 = F(empty)
V_VT = V_00 + cfg * (F(video, target) - V_00)
V_0S = V_00 + cfg * (F(no video, source) - V_00)
V_0T = V_00 + cfg * (F(no video, target) - V_00)
d = V_0S - V_0T
lambda = max(dot(V_VT - V_0T, d) / dot(d, d), 0)
V_TAP = V_VT - 1.2 * lambda * d
```

Dot products cover all non-batch dimensions independently for each sample.
A zero source–target direction leaves the field unchanged. The correction scale
**alpha=1.2 is constant at every step**. There is no transition or second phase.

| Backend | Sampling | Steps | Video/source/target CFG | Precision |
|---|---|---:|---:|---|
| MMAudio large 44k v2 | Deterministic Euler, 0 → 1 | 25 | 4.5 | BF16; FP32 projection |
| Omni2Sound | Deterministic v-ODE rotation, 1 → 0 | 100 | 2.5 | FP32 |

MMAudio uses learned null video embeddings for text-only branches. Omni2Sound
omits CLIP tokens in text-only branches, zeroes Synchformer features, and retains
duration conditioning. Its unconditional branch matches joint CFG dropout.

## Install and patch

Use each backbone's own environment, dependencies and pretrained weights:
[MMAudio](https://github.com/hkchengrex/MMAudio) and
[Omni2Sound](https://github.com/omni2sound/Omni2Sound).
Install this adapter into **each** environment, without replacing its PyTorch:

```bash
git clone https://github.com/anonymous129108/TAP-Foley.git
cd TAP-Foley
pip install -e . --no-deps
```

The patches target these exact upstream revisions:

| Backend | Commit |
|---|---|
| MMAudio | `974010a026c731054592d8f777218bd9d85a6c24` |
| Omni2Sound | `0222190690731627f5e98f6e47c43b3535879fcc` |

For fresh backbone checkouts:

```bash
git clone https://github.com/hkchengrex/MMAudio.git ../MMAudio
git -C ../MMAudio checkout 974010a026c731054592d8f777218bd9d85a6c24
python -m tap_foley.apply_patch mmaudio ../MMAudio --check
python -m tap_foley.apply_patch mmaudio ../MMAudio

git clone https://github.com/omni2sound/Omni2Sound.git ../Omni2Sound
git -C ../Omni2Sound checkout 0222190690731627f5e98f6e47c43b3535879fcc
python -m tap_foley.apply_patch omni2sound ../Omni2Sound --check
python -m tap_foley.apply_patch omni2sound ../Omni2Sound
```

The installer detects an already-applied patch. Existing upstream inference stays
the default; select `sampling_method="tap-foley"` to enable TAP. Patches do not
change weights, architecture, state-dict keys or training behavior.

## MMAudio

After completing MMAudio's upstream setup, run from its checkout:

```bash
python demo.py --video /path/to/video.mp4 \
  --prompt 'dog barking' --source_prompt 'hammering nails' \
  --sampling_method tap-foley --tap_alpha 1.2 \
  --num_steps 25 --cfg_strength 4.5 --duration 8 --seed 42
```

The patched `mmaudio.eval_utils.generate` also accepts
`sampling_method="tap-foley", source_text=[source_caption], tap_alpha=1.2`;
its existing `text` argument contains the target caption. Supply one source per
target in a batch. Leave `negative_text` empty and use both video streams.

## Omni2Sound

The patched `pyscripts.generate_v2a_cond.generate_diffusion_cond` accepts:

```python
audio = generate_diffusion_cond(
    model,
    sampling_method="tap-foley",
    source_prompt="hammering nails",
    conditioning=[{
        "task": "VTA",
        "clip_feature": "/path/to/video.clip.npy",
        "sync_feature": "/path/to/video.sync.npy",
        "text_prompt": "dog barking",
        "seconds_start": 0.0,
        "seconds_total": 10.0,
    }],
    steps=100, cfg_scale=2.5, tap_alpha=1.2,
    sample_size=model_config["sample_size"], seed=42, device="cuda",
)
```

Set steps and CFG explicitly: the upstream function retains its original native
sampling defaults. Omit native SDE sampler kwargs and negative conditioning.
`return_latents=True` skips decoding. Precomputed conditioner outputs are accepted
through `conditioning_tensors` and `source_text_condition`.

A complete single-example runner is provided. Run from the Omni2Sound checkout,
using features extracted with its official CLIP/Synchformer pipeline:

```bash
PYTHONPATH="$PWD" python ../TAP-Foley/examples/omni2sound.py \
  --model-config weights/omni2sound/vt2a-24-v55vt35-oa15-mq-td15/model_config.json \
  --checkpoint weights/omni2sound/vt2a-24-v55vt35-oa15-mq-td15/checkpoints/model.ckpt \
  --clip-feature /path/to/video.clip.npy --sync-feature /path/to/video.sync.npy \
  --source 'hammering nails' --target 'dog barking' --output output/tap.wav
```

This runner uses a single device with sufficient memory for the full FP32 model.
For an existing sharded deployment, `tap_foley.omni2sound.sample` accepts a model
callable and four precomputed branch dictionaries. Device placement remains the
caller's responsibility. No feature extraction or checkpoint download is bundled.

## Forward and sampler integration

`tap_foley.mmaudio.forward(net, latent, time, branches)` and
`tap_foley.omni2sound.forward(model_fn, latent, time, branches)` return the TAP field
at one latent/time. `branches` contains `empty`, `video_target`, `source`, and
`target`; build these with the backend's conditioning contract above.
The corresponding `sample` functions integrate this field over the whole trajectory.
Calling ordinary backbone `forward` directly remains unchanged.

## Benchmark and validation

[data/](data/README.md) contains 1,189 video identifiers, their source captions,
and ten reviewed target captions each, in CSV and JSONL. It contains no machine
paths, review notes or generated audio. The original video collection is obtained
separately.

```bash
python -m unittest discover -s tests -v
# Optional: verify patch applicability and dispatch on your upstream checkouts.
TAP_MMAUDIO_ROOT=/path/to/MMAudio TAP_OMNI2SOUND_ROOT=/path/to/Omni2Sound \
  python -m unittest discover -s tests -v
```

Tests cover projection geometry, degenerate cases, per-example independence,
conditioning, deterministic integrators, and patched API dispatch. Release
validation also compares the complete 25/100-step latent trajectories against
the research samplers with constant alpha=1.2. These are numerical integration
tests using deterministic test fields, not a new pretrained-model evaluation.

Pretrained models and upstream code retain their original licenses. See the
upstream repositories for model setup and usage terms.
