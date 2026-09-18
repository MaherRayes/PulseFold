from dataclasses import dataclass
from enum import Enum
from pathlib import Path

FRAME_CHUNK_SIZE = 16

class ColorChannel(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

class DataType(Enum):
    UINT8 = 1
    INT16 = 2


@dataclass(frozen=True)
class QuadtreeConfig:
    source: Path
    channel: ColorChannel
    base_factor:int
    leaves: int
    unguided_nodes: float
    temporal_weight: float
    dtype: DataType
    output: Path
