"""Camera calibration from chessboard images (Zhang's method via OpenCV).

Example:
    lanes-calibrate data/udacity/camera_cal --nx 9 --ny 6 -o models/udacity_calibration.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float
    image_size: tuple[int, int]

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs,
                 rms=self.rms, image_size=np.array(self.image_size))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        d = np.load(path)
        return cls(d["camera_matrix"], d["dist_coeffs"], float(d["rms"]),
                   tuple(int(v) for v in d["image_size"]))


def calibrate(images: list[np.ndarray], nx: int = 9, ny: int = 6) -> tuple[Calibration, int]:
    """Calibrate from chessboard photos with ``nx`` x ``ny`` inner corners.

    Returns the calibration and how many images had a detectable board.
    """
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    obj_points, img_points, size = [], [], None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
    if len(obj_points) < 3:
        raise ValueError(f"only {len(obj_points)} chessboards detected; need at least 3")

    rms, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return Calibration(mtx, dist, float(rms), size), len(obj_points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--nx", type=int, default=9)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("models/udacity_calibration.npz"))
    args = parser.parse_args()

    paths = sorted(args.folder.glob("*.jpg"))
    images = [cv2.imread(str(p)) for p in paths]
    calib, used = calibrate(images, args.nx, args.ny)
    calib.save(args.output)
    print(f"Calibrated from {used}/{len(paths)} images, RMS reprojection error "
          f"{calib.rms:.3f} px -> {args.output}")


if __name__ == "__main__":
    main()
