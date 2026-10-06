"""Train the U-Net lane-slot segmentation model on TuSimple.

Loss = cross-entropy (background down-weighted, lanes are thin) + multiclass Dice.
The checkpoint with the best validation TuSimple ego-lane accuracy is kept.

Example:
    lanes-train --epochs 40 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from ..config import (
    MODELS_DIR,
    NUM_CLASSES,
    REPORTS_DIR,
    TUSIMPLE_DIR,
    TUSIMPLE_TRAIN_JSONS,
    TUSIMPLE_VAL_JSONS,
)
from ..tusimple import available_records
from .dataset import TuSimpleDataset
from .model import DeepLaneDetector, build_model, pick_device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loss(device: torch.device):
    weights = torch.tensor([0.4] + [1.0] * (NUM_CLASSES - 1), device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights)
    dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(1, NUM_CLASSES)))
    return lambda logits, target: ce(logits, target) + dice(logits, target)


def save_checkpoint(model, path: Path, encoder: str, size, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "encoder": encoder, "input_size": list(size), **extra},
               path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=TUSIMPLE_DIR)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "unet_resnet34.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    from ..evaluate import evaluate_detector  # local import avoids a cycle at module load

    size = (args.width, args.height)
    train_recs = available_records(TUSIMPLE_TRAIN_JSONS, args.root)
    val_recs = available_records(TUSIMPLE_VAL_JSONS, args.root)
    if not train_recs or not val_recs:
        raise SystemExit("Missing TuSimple train/val images. Run `lanes-download tusimple`.")
    print(f"train {len(train_recs)} images, val {len(val_recs)} images")

    device = pick_device(args.device)
    loader = DataLoader(
        TuSimpleDataset(train_recs, args.root, size, augment=True, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        persistent_workers=args.workers > 0, drop_last=True)
    model = build_model(args.encoder).to(device)
    criterion = make_loss(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, pct_start=0.1,
                                                total_steps=args.epochs * len(loader))

    tmp_ckpt = args.output.with_suffix(".last.pt")
    history, best = [], -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            losses.append(loss.item())

        save_checkpoint(model, tmp_ckpt, args.encoder, size, {})
        summary, _ = evaluate_detector(DeepLaneDetector(tmp_ckpt, device=str(device)),
                                       val_recs, args.root, progress=False)
        val_acc = summary["ego"]["accuracy"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_ego_acc": val_acc,
                        "val_all_acc": summary["all"]["accuracy"]})
        flag = ""
        if val_acc > best:
            best = val_acc
            save_checkpoint(model, args.output, args.encoder, size,
                            {"epoch": epoch, "val_ego_acc": val_acc})
            flag = "  *best*"
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val ego acc {val_acc:.3f}  "
              f"val all acc {summary['all']['accuracy']:.3f}  "
              f"[{(time.time() - start) / 60:.1f} min]{flag}", flush=True)
    tmp_ckpt.unlink(missing_ok=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "training_history.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "device": str(device),
         "train_images": len(train_recs), "val_images": len(val_recs),
         "minutes": (time.time() - start) / 60, "history": history}, indent=2))
    print(f"Best val ego accuracy {best:.3f}; saved {args.output}")


if __name__ == "__main__":
    main()
