import tempfile
from pathlib import Path

from .config import QuadtreeConfig
from .encoder import encode_residual
from .exporter import export_quadtree
from .frameloader import load_frames
from .quadtree import build_quadtree


def run(config: QuadtreeConfig) -> None:
    temp_dir = (config.output / "temp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_dir) as temporary:
        frames = load_frames(config.source, config.channel, config.base_factor, Path(temporary + "/sequence.bin"))
        print("finished loading frames.")
        qt_data, gridcells, counts = build_quadtree(frames, config.base_factor, config.leaves, config.unguided_nodes, config.temporal_weight, Path(temporary + "/raw.bin"))
        print("finished building quadtree.")
        res_data, baseline, res_scale, zero_point = encode_residual(qt_data, Path(temporary + "/encoded.bin"), config.dtype)
        metadata = {
            "version": "1",
            "original_width": frames.original_width,
            "original_height": frames.original_height,
            "origianl_frame_count": frames.frame_count,
            "color_channel": config.channel.value,
            "base_factor": config.base_factor,
            "zero_point": zero_point,
            "data_dtype": str.lower(config.dtype.name)
        }

        export_quadtree(res_data, baseline, res_scale, counts, gridcells, metadata, config.output)

        frames.frames._mmap.close()
        qt_data._mmap.close()
        res_data._mmap.close()

