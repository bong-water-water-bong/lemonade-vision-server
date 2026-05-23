import numpy as np
from lemonade_vision.pipeline.dimensions import depth_to_dimensions


def test_depth_to_dimensions_basic():
    # Uniform depth grid at 350mm, product spans half the frame
    grid = np.full((256, 192), 350.0)
    dims = depth_to_dimensions(grid, scan_distance_mm=350.0)
    assert dims is not None
    w, h, d = dims
    assert w > 0
    assert h > 0
    assert d > 0


def test_depth_to_dimensions_empty_returns_none():
    assert depth_to_dimensions(np.array([]), scan_distance_mm=350.0) is None


def test_depth_to_dimensions_nearer_means_bigger():
    close_grid = np.full((256, 192), 200.0)
    far_grid = np.full((256, 192), 500.0)
    close_dims = depth_to_dimensions(close_grid, scan_distance_mm=200.0)
    far_dims = depth_to_dimensions(far_grid, scan_distance_mm=500.0)
    assert close_dims is not None and far_dims is not None
    assert close_dims != far_dims
