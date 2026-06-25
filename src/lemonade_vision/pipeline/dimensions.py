"""
Converts ARKit LiDAR depth grid to physical W×H×D estimate.
Assumes iPhone 15 Pro Max horizontal FOV ≈ 69°.
The product extent is estimated as the fraction of the frame
occupied by pixels at the product depth plane.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

IPHONE_FOV_H_DEG = 69.0
IPHONE_ASPECT = 4.0 / 3.0


def depth_to_dimensions(
    depth_grid: np.ndarray,
    scan_distance_mm: float = 350.0,
) -> Optional[tuple[float, float, float]]:
    if depth_grid.size == 0:
        return None

    h_px, w_px = depth_grid.shape if depth_grid.ndim == 2 else (0, 0)
    if h_px == 0 or w_px == 0:
        return None

    fov_h_rad = math.radians(IPHONE_FOV_H_DEG)
    fov_v_rad = 2.0 * math.atan(math.tan(fov_h_rad / 2.0) / IPHONE_ASPECT)

    # Physical size of the full frame at the scan distance
    frame_w_mm = 2.0 * scan_distance_mm * math.tan(fov_h_rad / 2.0)
    frame_h_mm = 2.0 * scan_distance_mm * math.tan(fov_v_rad / 2.0)

    # Foreground pixels = depth < scan_distance * 0.95 (product is closer than bg)
    fg_mask = depth_grid < (scan_distance_mm * 0.95)
    rows_with_fg = np.any(fg_mask, axis=1)
    cols_with_fg = np.any(fg_mask, axis=0)

    if not np.any(rows_with_fg) or not np.any(cols_with_fg):
        # No distinguishable foreground — estimate from full grid range
        row_span = h_px
        col_span = w_px
    else:
        row_span = int(np.sum(rows_with_fg))
        col_span = int(np.sum(cols_with_fg))

    width_mm = (col_span / w_px) * frame_w_mm
    height_mm = (row_span / h_px) * frame_h_mm
    depth_mm_estimate = float(np.percentile(depth_grid, 5))

    return (round(width_mm, 1), round(height_mm, 1), round(depth_mm_estimate, 1))
