from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from pulsefold.config import ColorChannel

TIFF_IMAGE_EXTENSIONS = {
    ".tiff", ".tif"
}

@dataclass
class FrameStack:
    frames: np.ndarray
    count: int
    width: int
    height: int

def load_frames(path: Path, channel: ColorChannel) -> FrameStack:
    if not path.exists():
        raise OSError("Path is not valid.")
    
    if path.suffix.lower() in TIFF_IMAGE_EXTENSIONS:
        with tifffile.TiffFile(path) as tif:
            map_to_channel = lambda page : page.asarray() if channel is ColorChannel.ALL else page.asarray()[:, :, channel.value]
            sequence = np.stack([map_to_channel(page) for page in tif.pages], axis=0)

        #Handle unsupported tiff files
        if sequence is None:
            raise ValueError("Image/Sequence could not be loaded.")
        elif sequence.ndim != 4 and sequence.ndim != 3:
            raise ValueError(f"Unsupported TIFF shape: {sequence.shape}")
        elif sequence.dtype != np.uint8:
            raise ValueError(f"Unsupported TIFF dtype: {sequence.dtype}. Only accepts uint8")
    else:
        raise OSError("Path doesn't point to an image.")
    
    frame_stack = FrameStack(frames= sequence, count= sequence.shape[0], width= sequence.shape[2], height= sequence.shape[1])
    
    return frame_stack