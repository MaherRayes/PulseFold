import numpy as np

from pulsefold.frameloader import FrameStack


GridCell = tuple[int, int, int, int]

def _factored_pixel_counts(height: int, width: int, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("spatial factor must be at least 1")

    row_counts = np.full((height + factor - 1) // factor, factor, dtype=np.int32)
    column_counts = np.full((width + factor - 1) // factor, factor, dtype=np.int32)

    row_counts[-1] = height % factor if height % factor > 0 else factor
    column_counts[-1] = width % factor if width % factor > 0 else factor

    return np.outer(row_counts, column_counts)

def build_quadtree(frames: FrameStack, leaves:int, base_cell_factor:int) -> np.ndarray:

    print(_factored_pixel_counts(frames.height, frames.width, base_cell_factor))

    return np.empty([1])