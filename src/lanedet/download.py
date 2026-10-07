"""Download TuSimple (Hugging Face mirror) and the Udacity calibration/video assets.

Only the labelled frame of each TuSimple clip (``20.jpg``) is fetched. Note: the public
Hugging Face mirror contains only ~500 of the ~6,400 labelled clips, so this gives a
*subset* of TuSimple (frames that are missing from the mirror are skipped and counted).
The ``kaggle`` source fetches the *full* dataset (~23 GB zip with every video frame,
needs ``~/.kaggle/kaggle.json``) and extracts only the labels and the labelled frames
(~1.3 GB) into ``data/tusimple_full/``.

Examples:
    lanes-download udacity
    lanes-download tusimple --split train
    lanes-download tusimple --split test --limit 300
    lanes-download kaggle            # full TuSimple -> data/tusimple_full
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config import (
    DATA_DIR,
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


KAGGLE_DATASET = "manideep1108/tusimple"
TUSIMPLE_FULL_DIR = DATA_DIR / "tusimple_full"


def extract_tusimple_zip(zip_path: Path, root: Path = TUSIMPLE_FULL_DIR) -> int:
    """Extract labels + labelled frames from the Kaggle archive into one flat root.

    The archive keeps train and test under ``*/train_set/`` and ``*/test_set/``; their
    clip folders don't collide, so both are merged under ``root/clips/`` to match the
    ``raw_file`` paths in the label files. Returns the number of frames extracted.
    """
    frames = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            is_label = name.endswith(".json") and parts[-1].startswith(("label_data", "test_"))
            is_frame = parts[-1] == "20.jpg" and "clips" in parts
            if not (is_label or is_frame):
                continue
            rel = Path(parts[-1]) if is_label else Path(*parts[parts.index("clips"):])
            dest = root / rel
            if dest.exists():
                frames += is_frame
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, dest.open("wb") as out:
                out.write(src.read())
            frames += is_frame
    return frames


def download_tusimple_kaggle(root: Path = TUSIMPLE_FULL_DIR, keep_zip: bool = False) -> int:
    cache = DATA_DIR / "kaggle"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "tusimple.zip"
    if not zip_path.exists():
        subprocess.run(["kaggle", "datasets", "download", KAGGLE_DATASET, "-p", str(cache)],
                       check=True)
    frames = extract_tusimple_zip(zip_path, root)
    if not keep_zip:
        zip_path.unlink()
    return frames


def download_udacity(root: Path = UDACITY_DIR) -> int:
    for rel in tqdm(UDACITY_FILES, desc="udacity", unit="file"):
        _fetch(f"{UDACITY_RAW}/{rel}", root / rel)
    return len(UDACITY_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("tusimple", "udacity", "kaggle"))
    parser.add_argument("--keep-zip", action="store_true", help="kaggle: keep the 23 GB zip")
    parser.add_argument("--split", choices=tuple(SPLIT_JSONS), default="train")
    parser.add_argument("--limit", type=int, help="only the first N frames")
    args = parser.parse_args()
    if args.dataset == "udacity":
        print(f"Done: {download_udacity()} files")
    elif args.dataset == "kaggle":
        frames = download_tusimple_kaggle(keep_zip=args.keep_zip)
        print(f"Done: {frames} labelled frames in {TUSIMPLE_FULL_DIR}")
    else:
        done, total = download_tusimple(args.split, limit=args.limit)
        print(f"Done: {done} of {total} labelled {args.split} frames available in the mirror")


if __name__ == "__main__":
    main()
