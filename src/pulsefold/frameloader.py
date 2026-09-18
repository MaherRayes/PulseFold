from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .config import ColorChannel

TIFF_IMAGE_EXTENSIONS = {
    ".tiff", ".tif"
}

@dataclass
class FrameStack:
    frames: np.ndarray
    frame_count: int
    original_width: int
    original_height: int
    reduced_width: int
    reduced_height: int
    counts: np.ndarray

def _spatial_average(frames: np.ndarray, factor: int, counts: np.ndarray, dtype=np.float64) -> np.ndarray:
    array = np.asarray(frames).astype(dtype)
    y_indicies = np.arange(0, frames.shape[-2], factor)
    x_indicies = np.arange(0, frames.shape[-1], factor)

    reduced_frames = np.add.reduceat(np.add.reduceat(array, y_indicies, axis=-2), x_indicies, axis=-1) / counts

    return reduced_frames

def _factored_pixel_counts(height: int, width: int, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("spatial factor must be at least 1")

    row_counts = np.full((height + factor - 1) // factor, factor, dtype=np.int32)
    column_counts = np.full((width + factor - 1) // factor, factor, dtype=np.int32)

    row_counts[-1] = height % factor if height % factor > 0 else factor
    column_counts[-1] = width % factor if width % factor > 0 else factor

    return np.outer(row_counts, column_counts)

def _read_frame(page, channel: ColorChannel) -> np.ndarray:
    frame = page.asarray()
    return frame[:, :, channel.value]

def load_frames(path: Path, channel: ColorChannel, base_cell_factor: int, output_path: Path) -> FrameStack:
    if not path.exists():
        raise OSError("Path is not valid.")
    
    if path.suffix.lower() in TIFF_IMAGE_EXTENSIONS:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) == 0:
                raise ValueError("Sequence must have at least 1 frame.")

            first_frame = _read_frame(tif.pages[0], channel)

            #Handle unsupported tiff files
            if first_frame is None:
                raise ValueError("Image/Sequence could not be loaded.")
            elif first_frame.ndim != 3 and first_frame.ndim != 2:
                raise ValueError(f"Unsupported TIFF frame shape: {first_frame.shape}")
            elif first_frame.dtype != np.uint8:
                raise ValueError(f"Unsupported TIFF dtype: {first_frame.dtype}. Only accepts uint8")

            counts = _factored_pixel_counts(first_frame.shape[0], first_frame.shape[1], base_cell_factor)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sequence = np.memmap(output_path, mode="w+", dtype=np.float32, shape=(len(tif.pages), *counts.shape))
            
            sequence[0] = _spatial_average(first_frame, base_cell_factor, counts, np.float32)
            for i, page in enumerate(tif.pages[1:], start=1):
                sequence[i] = _spatial_average(_read_frame(page, channel), base_cell_factor, counts, np.float32)

            sequence.flush()

        
    else:
        raise OSError("Path doesn't point to an image.")
    
    frame_stack = FrameStack(
        frames=sequence,
        frame_count=sequence.shape[0],
        original_width=first_frame.shape[1],
        original_height=first_frame.shape[0],
        reduced_width=sequence.shape[2],
        reduced_height=sequence.shape[1],
        counts=counts
        )
    
    return frame_stack