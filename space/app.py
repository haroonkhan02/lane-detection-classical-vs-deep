"""Hugging Face Space entry point: the repo's Gradio demo with bundled examples.

`scripts/deploy_space.py` copies this file next to the classical config and example images
and uploads the bundle; the `lanedet` package itself is installed from GitHub
(see requirements.txt), and the U-Net weights come from the Hub model repo.
"""

from pathlib import Path

from lanedet.demo import build_app
from lanedet.hub import resolve_checkpoint

HERE = Path(__file__).parent
NOTE = (
    "\n\nExamples 1–4 are TuSimple test frames. The last one is from Udacity's project "
    "video: a different camera, where the classical warp is not calibrated and the network "
    "is off its training domain. Code and results: "
    "[GitHub](https://github.com/haroonkhan02/lane-detection-classical-vs-deep)."
)

app = build_app(
    HERE / "tusimple_full.yaml",
    resolve_checkpoint(None),
    examples=sorted(str(p) for p in (HERE / "examples").glob("*.jpg")),
    description=NOTE,
)

if __name__ == "__main__":
    app.launch()
