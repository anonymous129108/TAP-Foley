# VGGSound-Counterfactual manifest

The reviewed VGGSounder subset contains **1,189 videos**, one source caption and
**10 target captions per video**: **11,890 counterfactual triplets**.

- `vggsound-counterfactual.csv`: `Video ID`, `category` (source), `target_caption_01` … `target_caption_10`.
- `vggsound-counterfactual.jsonl`: one `{video, source, targets}` object per video.
- `manifest.json`: counts, schema and SHA-256 checksums.

Caption spelling and target order are preserved exactly. Each `video` is a
relative MP4 filename: the final underscore separates the YouTube ID from its
zero-padded start time in seconds. For example, `-23CeprtibU_000030.mp4` refers
to the VGGSound clip starting at 30 seconds in video `-23CeprtibU`.

Obtain the original videos through the [VGGSound project](https://www.robots.ox.ac.uk/~vgg/data/).
Resolve `video` against your local video directory; do not use the reference
audio as model conditioning. The release contains video identifiers and captions;
it does not bundle the original video collection or pretrained models. Source
videos and pretrained models retain their respective upstream terms.

```python
import json
from pathlib import Path

video_root = Path("/path/to/vggsound/video")
for line in Path("data/vggsound-counterfactual.jsonl").read_text().splitlines():
    example = json.loads(line)
    video = video_root / example["video"]
    for target in example["targets"]:
        # Generate from video, example["source"], target.
        print(video, example["source"], target)
```
