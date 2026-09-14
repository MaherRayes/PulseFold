

from .config import QuadtreeConfig
from .frameloader import load_frames
from .quadtree import build_quadtree
from .encoder import encode_residual


def run(config: QuadtreeConfig) -> None:

    frames = load_frames(config.source, config.channel)
    qt_data, gridcells = build_quadtree(frames, 2, config.leaves, config.unguided_nodes, config.temporal_weight, config.output / "raw.bin")
    res_values, baseline, res_scale, zero_point = encode_residual(qt_data, config.output / "encoded.bin", config.dtype) 