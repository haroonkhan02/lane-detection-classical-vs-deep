"""Drawing helpers: lane overlay, HUD and debug mosaics."""

from __future__ import annotations

import cv2
import numpy as np

from .types import LaneResult

SLOT_COLORS = {"left_left": (255, 160, 0), "left": (0, 220, 255), "right": (0, 220, 255),
               "right_right": (255, 160, 0)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_lanes(bgr: np.ndarray, result: LaneResult, fill: bool = True) -> np.ndarray:
    out = bgr.copy()
    if fill and "left" in result.lanes and "right" in result.lanes:
        left = result.lanes["left"]
        right = result.lanes["right"][::-1]
        poly = np.vstack([left, right]).astype(np.int32)
        layer = out.copy()
        color = (0, 0, 255) if result.departure else (0, 200, 0)
        cv2.fillPoly(layer, [poly], color)
        out = cv2.addWeighted(layer, 0.3, out, 0.7, 0)
    for slot, pts in result.lanes.items():
        cv2.polylines(out, [pts.astype(np.int32).reshape(-1, 1, 2)], False,
                      SLOT_COLORS.get(slot, (255, 255, 255)), 6, cv2.LINE_AA)
    return out


def draw_hud(bgr: np.ndarray, result: LaneResult, label: str = "", fps: float | None = None
             ) -> np.ndarray:
    out = bgr.copy()
    lines = [label] if label else []
    if result.curvature_m is not None:
        curv = result.curvature_m
        lines.append("Curvature: straight" if curv > 3000 else f"Curvature: {curv:,.0f} m")
    if result.offset_m is not None:
        side = "right" if result.offset_m > 0 else "left"
        lines.append(f"Offset: {abs(result.offset_m):.2f} m {side} of centre")
    if not result.lanes:
        lines.append("No lanes detected")
    if fps is not None:
        lines.append(f"{fps:.0f} FPS")
    for i, text in enumerate(lines):
        cv2.putText(out, text, (20, 40 + 34 * i), FONT, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (20, 40 + 34 * i), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if result.departure:
        w = out.shape[1]
        cv2.rectangle(out, (w // 2 - 230, 20), (w // 2 + 230, 75), (0, 0, 255), -1)
        cv2.putText(out, "LANE DEPARTURE WARNING", (w // 2 - 215, 58), FONT, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def binary_to_bgr(binary: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((binary > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)


def mosaic(images: list[np.ndarray], cols: int = 2, width: int = 640) -> np.ndarray:
    """Tile images (resized to the same width) into a grid."""
    tiles = []
    for img in images:
        img = binary_to_bgr(img) if img.ndim == 2 else img
        h = int(img.shape[0] * width / img.shape[1])
        tiles.append(cv2.resize(img, (width, h)))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)
