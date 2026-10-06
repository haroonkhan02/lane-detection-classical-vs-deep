"""Download TuSimple (Hugging Face mirror) and the Udacity calibration/video assets.

Only the labelled frame of each TuSimple clip (``20.jpg``) is fetched. Note: the public
Hugging Face mirror contains only ~500 of the ~6,400 labelled clips, so this gives a
*subset* of TuSimple (frames that are missing from the mirror are skipped and counted).
If you have the full dataset (e.g. from Kaggle), extract it to ``data/tusimple/`` with
the standard layout (``clips/...``, ``label_data_*.json``, ``test_label.json``) and
everything else works unchanged.

Examples:
    lanes-download udacity
    lanes-download tusimple --split train
    lanes-download tusimple --split test --limit 300
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config import (
    TUSIMPLE_DIR,
    TUSIMPLE_REPO,
    TUSIMPLE_TEST_JSON,
    TUSIMPLE_TRAIN_JSONS,
    TUSIMPLE_VAL_JSONS,
    UDACITY_DIR,
    UDACITY_FILES,
    UDACITY_RAW,
)

SPLIT_JSONS = {
    "train": TUSIMPLE_TRAIN_JSONS,
    "val": TUSIMPLE_VAL_JSONS,
    "test": (TUSIMPLE_TEST_JSON,),
}


class _RateLimiter:
    """Hugging Face allows ~3000 anonymous file requests per 5 minutes per IP."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


_LIMITER = _RateLimiter(per_second=9.0)


def _fetch(url: str, dest: Path, retries: int = 6, timeout: float = 30.0) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        _LIMITER.wait()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as f:
                f.write(r.read())
            tmp.rename(dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
            if attempt == retries - 1:
                raise RuntimeError(f"failed to download {url}") from e
            time.sleep(60 if e.code == 429 else 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"failed to download {url}") from e
            time.sleep(2 ** attempt)


def _hf_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{TUSIMPLE_REPO}/resolve/main/{path}"


def _mirror_clip_dirs() -> set[str]:
    """Clip folders that actually exist in the Hugging Face mirror."""
    from huggingface_hub import HfApi

    api = HfApi()
    dirs = set()
    for top in api.list_repo_tree(TUSIMPLE_REPO, path_in_repo="clips", repo_type="dataset"):
        for clip in api.list_repo_tree(TUSIMPLE_REPO, path_in_repo=top.path,
                                       repo_type="dataset"):
            dirs.add(clip.path)
    return dirs


def download_tusimple(split: str, root: Path = TUSIMPLE_DIR, limit: int | None = None,
                      workers: int = 8) -> tuple[int, int]:
    """Download the labelled frames of ``split``. Returns (downloaded, labelled total)."""
    files: list[str] = []
    for name in SPLIT_JSONS[split]:
        _fetch(_hf_url(name), root / name)
        with (root / name).open() as f:
            files += [json.loads(line)["raw_file"] for line in f if line.strip()]
    total = len(files)
    available = _mirror_clip_dirs()
    files = [p for p in files if p.rsplit("/", 1)[0] in available]
    files = files[:limit] if limit else files

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(_fetch, _hf_url(p), root / p) for p in files]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc=f"tusimple/{split}",
                        unit="img"):
            try:
                job.result()
                done += 1
            except FileNotFoundError:
                pass
    return done, total


def download_udacity(root: Path = UDACITY_DIR) -> int:
    for rel in tqdm(UDACITY_FILES, desc="udacity", unit="file"):
        _fetch(f"{UDACITY_RAW}/{rel}", root / rel)
    return len(UDACITY_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("tusimple", "udacity"))
    parser.add_argument("--split", choices=tuple(SPLIT_JSONS), default="train")
    parser.add_argument("--limit", type=int, help="only the first N frames")
    args = parser.parse_args()
    if args.dataset == "udacity":
        print(f"Done: {download_udacity()} files")
    else:
        done, total = download_tusimple(args.split, limit=args.limit)
        print(f"Done: {done} of {total} labelled {args.split} frames available in the mirror")


if __name__ == "__main__":
    main()
