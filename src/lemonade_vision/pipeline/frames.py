"""
Frame extraction from rotation video using ffmpeg subprocess.
Sharpness scoring uses Laplacian variance on the grayscale image —
no OpenCV dependency required.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

SECTORS = 12          # 30° per sector across 360°
FPS_EXTRACT = 3.0     # frames per second to extract from video
BLUR_THRESHOLD = 50.0 # Laplacian variance below this → discard


def laplacian_variance(image_path: Path) -> float:
    try:
        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=float)
    except Exception:
        return 0.0
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    patches = sliding_window_view(arr, (3, 3)).reshape(-1, 9)
    k = kernel.flatten()
    lap = patches @ k
    return float(np.var(lap))


def extract_frames_from_video(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%04d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps={FPS_EXTRACT}",
            "-q:v", "2",
            pattern,
        ],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.jpg"))


def select_sharpest_frames(
    indexed_frames: list[tuple[int, str]],
) -> list[str]:
    """
    indexed_frames: list of (degree_angle, path_str)
    Returns at most SECTORS frames — one per 30° sector, sharpest wins.
    """
    sector_best: dict[int, tuple[float, str]] = {}
    for angle, path_str in indexed_frames:
        sector = int(angle // (360.0 / SECTORS)) % SECTORS
        score = laplacian_variance(Path(path_str))
        if score < BLUR_THRESHOLD:
            continue
        if sector not in sector_best or score > sector_best[sector][0]:
            sector_best[sector] = (score, path_str)
    return [v[1] for v in sector_best.values()]


def frames_from_video(video_path: Path, out_dir: Path) -> list[str]:
    all_frames = extract_frames_from_video(video_path, out_dir)
    total = len(all_frames)
    indexed = [
        (int(i / max(total, 1) * 360), str(p))
        for i, p in enumerate(all_frames)
    ]
    return select_sharpest_frames(indexed)
