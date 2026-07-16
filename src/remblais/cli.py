import argparse
import sys
from pathlib import Path

from .io import load_image, mass_delta_rgb, save_video
from .morph import morph


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="remblais",
        description="Morph images by routing pixel mass with optimal transport.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("morph", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmd.add_argument("source", type=Path)
    cmd.add_argument("target", type=Path)
    cmd.add_argument("--out", type=Path, default=Path("morph.gif"))
    cmd.add_argument("--fps", type=int, default=12)
    cmd.add_argument("--frames", type=int, default=None)
    cmd.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=None)
    cmd.add_argument("--X", "--distance-weight", dest="X", type=float, default=1.0)
    cmd.add_argument("--Y", "--mass-penalty", dest="Y", type=float, default=1e10)
    cmd.add_argument("--epsilon", type=float, default=0.01)
    cmd.add_argument("--iters", type=int, default=300)
    cmd.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    cmd.add_argument("--normalize", choices=["clip", "global", "tonemap"], default="clip")
    return root


def run_morph(args: argparse.Namespace) -> None:
    size = tuple(args.size) if args.size else None
    A, B = load_image(args.source, size), load_image(args.target, size)
    if A.shape != B.shape:
        sys.exit(f"error: shape mismatch A={A.shape} B={B.shape}")
    print(f"mass delta per channel: {mass_delta_rgb(A, B)}")
    result = morph(
        A,
        B,
        X=args.X,
        Y=args.Y,
        epsilon=args.epsilon,
        n_frames=args.frames,
        normalize=args.normalize,
        num_iters=args.iters,
        dtype=args.dtype,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = result.frames_np()
    save_video(frames, args.out, args.fps)
    print(f"saved {len(frames)} frames -> {args.out}")


def main() -> None:
    args = parser().parse_args()
    if args.command == "morph":
        run_morph(args)


if __name__ == "__main__":
    main()
