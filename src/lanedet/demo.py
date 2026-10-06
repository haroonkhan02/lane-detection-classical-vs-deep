"""Gradio web demo: upload a road image and compare both detectors side by side.

Example:
    lanes-demo            # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .classical.config import ClassicalConfig
from .classical.pipeline import ClassicalLaneDetector
from .config import CONFIGS_DIR, MODELS_DIR, TUSIMPLE_DIR
from .viz import binary_to_bgr, draw_hud, draw_lanes


def build_app(config: Path, checkpoint: Path):
    import gradio as gr

    cfg = ClassicalConfig.from_yaml(config)
    classical = ClassicalLaneDetector(cfg)
    deep = None
    if checkpoint.exists():
        from .deep.model import DeepLaneDetector

        deep = DeepLaneDetector(checkpoint, classical_cfg=cfg)

    def run(rgb: np.ndarray):
        if rgb is None:
            return None, None, None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        res_c, debug = classical.detect(bgr)
        out_c = draw_hud(draw_lanes(bgr, res_c), res_c, "classical")
        bird = np.hstack([debug.warped, binary_to_bgr(debug.binary)])
        if deep is not None:
            res_d = deep(bgr)
            out_d = draw_hud(draw_lanes(bgr, res_d), res_d,
                             f"deep ({deep.last_inference_ms:.0f} ms)")
        else:
            out_d = np.zeros_like(bgr)
        to_rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # noqa: E731
        return to_rgb(out_c), to_rgb(out_d), to_rgb(bird)

    examples = sorted(str(p) for p in (TUSIMPLE_DIR / "clips").rglob("20.jpg"))[:4]
    with gr.Blocks(title="Lane detection: classical vs deep") as app:
        gr.Markdown("## Lane detection: classical CV vs. U-Net\nUpload a front-camera road "
                    "image (TuSimple-like viewpoint works best).")
        inp = gr.Image(label="Input", type="numpy")
        with gr.Row():
            out_c = gr.Image(label="Classical pipeline")
            out_d = gr.Image(label="Deep model" + ("" if deep else " (no checkpoint found)"))
        bird = gr.Image(label="Classical: bird's-eye view and thresholded binary")
        inp.change(run, inp, [out_c, out_d, bird])
        if examples:
            gr.Examples(examples, inp)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "tusimple.yaml")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "unet_resnet34.pt")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()
    build_app(args.config, args.checkpoint).launch(share=args.share)


if __name__ == "__main__":
    main()
