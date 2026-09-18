import argparse
from pathlib import Path

from pulsefold.config import ColorChannel, DataType, QuadtreeConfig
from pulsefold.pipeline import run


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    modes = parser.add_subparsers(dest="mode", required=True)

    quadtree_mode = modes.add_parser("quadtree", help="Compress a tiff sequence as a static quadtree")
    quadtree_mode.add_argument("--src", required=True, type=Path, help="Path to source sequence")
    quadtree_mode.add_argument("--channel", required=False, type=str, default="all", choices=[e.name.lower() for e in ColorChannel], help="Compress specific channel")
    quadtree_mode.add_argument("--base-factor", required=False, type=int, default=2, help="Factor for initial spatial compression of the sequence (factor 1 = original size)")
    quadtree_mode.add_argument("--leaf-budget", required=False, type=int, default=34000, help="Number of quadtree leaves (Directly corresponds to the size)")
    quadtree_mode.add_argument("--unguided-node-percentage", required=False, type=float, default=0.5, help="Percentage of the quadtree leaves that get allocated unguided at the start")
    quadtree_mode.add_argument("--temporal-weight", required=False, type=float, default=8.0, help="Weight of temporal error to static error when building the quadtree")
    quadtree_mode.add_argument("--dtype", required=False, type=str, default="uint8", choices=[e.name.lower() for e in DataType], help="Type of stored data")
    quadtree_mode.add_argument("--out", required=True, type=Path, help="Path to export the compressed file")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.src.exists():
        raise argparse.ArgumentTypeError("The source path does not exist.")
    if args.base_factor < 1:
        raise argparse.ArgumentTypeError("Base factor has to be 1 or higher.")
    if args.leaf_budget < 1:
        raise argparse.ArgumentTypeError("Leaf budget has to be 1 or higher.")
    if args.unguided_node_percentage < 0.0 or args.unguided_node_percentage > 1.0:
            raise argparse.ArgumentTypeError("Unguided node percentage has to be between 0.0 and 1.0.")
    if args.temporal_weight < 0.0:
            raise argparse.ArgumentTypeError("Temporal weight has to be higher than 0.0.")
    Path(args.out).mkdir(parents=True, exist_ok=True)

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    validate_args(args)

    config = QuadtreeConfig(
        source=args.src,
        channel=ColorChannel[args.channel.upper()],
        base_factor=args.base_factor,
        leaves=args.leaf_budget,
        unguided_nodes=args.unguided_node_percentage,
        temporal_weight=args.temporal_weight,
        dtype=DataType[args.dtype.upper()],
        output=args.out
    )

    run(config)

if __name__ == "__main__":
    main()