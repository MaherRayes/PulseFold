

from pulsefold.config import QuadtreeConfig
from pulsefold.frameloader import load_frames
from .quadtree import build_quadtree




def run(config: QuadtreeConfig) -> None:

    frames = load_frames(config.source, config.channel)
    build_quadtree(frames, config.leaves, 2)
