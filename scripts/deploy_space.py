"""Bundle the Gradio Space (app, classical config, example frames) and upload it.

Uses the Hugging Face login stored by `hf auth login`. The U-Net weights are published
separately to the model repo in `lanedet.hub.HUB_REPO`.

Example:
    python scripts/deploy_space.py              # build outputs/space_bundle and upload
    python scripts/deploy_space.py --dry-run    # build the bundle only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_REPO = "haroonx02/lane-detection-classical-vs-deep"
EXAMPLES = {  # bundle name -> source frame
    "1_multilane.jpg": "data/tusimple_full/clips/0601/1495492754556334960/20.jpg",
    "2_curve.jpg": "data/tusimple_full/clips/0531/1492635040463935320/20.jpg",
    "3_truck_occlusion.jpg": "data/tusimple_full/clips/0601/1494452471554214540/20.jpg",
    "4_concrete_patch.jpg": "data/tusimple_full/clips/0530/1492638009277743415_0/20.jpg",
    "5_udacity_other_camera.jpg": "data/udacity/test_images/test5.jpg",
}


def build_bundle(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "examples").mkdir(parents=True)
    for name in ("app.py", "requirements.txt", "packages.txt", "README.md"):
        shutil.copy(ROOT / "space" / name, out / name)
    shutil.copy(ROOT / "configs" / "tusimple_full.yaml", out / "tusimple_full.yaml")
    for name, src in EXAMPLES.items():
        shutil.copy(ROOT / src, out / "examples" / name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=SPACE_REPO)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "space_bundle")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(args.out)
    print(f"Bundle ready at {bundle}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=bundle,
                      commit_message="Deploy demo")
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
