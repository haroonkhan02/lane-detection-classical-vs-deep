"""Publish the in-browser demo: ONNX files to the model repo, web/ to a free static Space.

Uses the Hugging Face login stored by `hf auth login`. The page loads the INT8 ONNX file from
the model repo, so the Space itself only holds HTML, JavaScript and the example frames.

Example:
    python scripts/deploy_static_space.py --dry-run   # build outputs/static_space only
    python scripts/deploy_static_space.py --models    # also upload models/onnx/*.onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from deploy_space import EXAMPLES

from lanedet.hub import HUB_REPO

ROOT = Path(__file__).resolve().parents[1]
SPACE_REPO = "haroonx02/lane-detection-browser"
WEB_FILES = ("index.html", "app.js", "lanes.js", "README.md")
ONNX_FILES = ("lane_unet_int8.onnx", "lane_unet_fp16.onnx")


def build_bundle(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "examples").mkdir(parents=True)
    for name in WEB_FILES:
        shutil.copy(ROOT / "web" / name, out / name)
    for name, src in EXAMPLES.items():
        shutil.copy(ROOT / src, out / "examples" / name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=SPACE_REPO)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "static_space")
    parser.add_argument("--models", action="store_true", help="upload the ONNX files too")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(args.out)
    print(f"Bundle ready at {bundle}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    if args.models:
        for name in ONNX_FILES:
            api.upload_file(path_or_fileobj=ROOT / "models" / "onnx" / name,
                            path_in_repo=f"onnx/{name}", repo_id=HUB_REPO,
                            commit_message=f"Add {name}")
    api.create_repo(args.repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=bundle,
                      commit_message="Deploy browser demo")
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
