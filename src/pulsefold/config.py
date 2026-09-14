from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ColorChannel(Enum):
    RED =0
    GREEN = 1
    BLUE = 2
    ALL = 3

class DataType(Enum):
    UINT8 = 1
    INT16 = 2
    FP16 = 3


@dataclass(frozen=True)
class QuadtreeConfig:
    source: Path
    timestamps: Path
    channel: ColorChannel
    leaves: int
    unguided_nodes: float
    temporal_weight: float
    dtype: DataType
    output: Path
