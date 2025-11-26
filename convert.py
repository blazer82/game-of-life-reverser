#!/usr/bin/env python3
"""
PNG conversion utility for Game of Life grids.

Convert between text grid files and 32x32 PNG images.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from PIL import Image

from grid import Grid, GRID_SIZE


def grid_to_png(grid: Grid, output_path: str | Path) -> None:
    """
    Convert grid to 32x32 PNG image.

    - 1 pixel per cell
    - White (#FFFFFF) = alive
    - Black (#000000) = dead
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create image (mode 'L' = 8-bit grayscale)
    img = Image.new("L", (GRID_SIZE, GRID_SIZE), color=0)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x]:
                img.putpixel((x, y), 255)  # White for alive

    img.save(output_path)


def png_to_grid(input_path: str | Path) -> Grid:
    """
    Convert 32x32 PNG image to grid.

    - Brightness > 127 = alive
    - Brightness <= 127 = dead
    """
    input_path = Path(input_path)
    img = Image.open(input_path).convert("L")  # Convert to grayscale

    # Check dimensions
    if img.size != (GRID_SIZE, GRID_SIZE):
        print(
            f"Warning: Image is {img.size[0]}x{img.size[1]}, expected {GRID_SIZE}x{GRID_SIZE}",
            file=sys.stderr,
        )
        # Resize if needed
        img = img.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.NEAREST)

    grid = Grid()
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            brightness = img.getpixel((x, y))
            grid.cells[y][x] = brightness > 127

    return grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert between grid text files and PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert grid to PNG
    python convert.py grid2png pattern.txt -o pattern.png

    # Convert PNG to grid
    python convert.py png2grid pattern.png -o pattern.txt
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # grid2png command
    g2p = subparsers.add_parser("grid2png", help="Convert grid file to PNG")
    g2p.add_argument("input", type=str, help="Input grid file")
    g2p.add_argument("-o", "--output", type=str,
                     required=True, help="Output PNG file")

    # png2grid command
    p2g = subparsers.add_parser("png2grid", help="Convert PNG to grid file")
    p2g.add_argument("input", type=str, help="Input PNG file")
    p2g.add_argument("-o", "--output", type=str,
                     required=True, help="Output grid file")
    p2g.add_argument(
        "--format",
        choices=["binary", "dot", "compact"],
        default="binary",
        help="Output format (default: binary)",
    )

    args = parser.parse_args()

    if args.command == "grid2png":
        try:
            grid = Grid.from_file(args.input)
            grid_to_png(grid, args.output)
            print(f"Saved PNG to {args.output}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "png2grid":
        try:
            grid = png_to_grid(args.input)
            grid.to_file(args.output, args.format)
            print(f"Saved grid to {args.output}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
