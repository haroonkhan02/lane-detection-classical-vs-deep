"""Model weights on the Hugging Face Hub, so the demo works without training first.

The full-TuSimple checkpoint is published at ``HUB_REPO``. ``resolve_checkpoint`` returns a
local file when one exists and otherwise downloads (and caches) the published one.
"""

from __future__ import annotations

from pathlib import Path

HUB_REPO = "haroonx02/lane-detection-unet-resnet34"
HUB_FILENAME = "unet_resnet34_full.pt"


def resolve_checkpoint(local: Path | None, use_hub: bool = True,
                       repo_id: str = HUB_REPO, filename: str = HUB_FILENAME) -> Path | None:
    """Local checkpoint if it exists, else the Hub copy, else None (deep model disabled)."""
    if local is not None and local.exists():
        return local
    if not use_hub:
        return None
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as err:  # offline, repo renamed, rate limit: run classical-only
        print(f"Could not fetch {repo_id}/{filename} from the Hub ({err}); deep model off.")
        return None
