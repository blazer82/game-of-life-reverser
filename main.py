#!/usr/bin/env python3
"""
Conway's Game of Life Reverser

Find predecessor states for Game of Life patterns.

Usage:
    python main.py input.txt -o output.txt
    python main.py input.txt -n 5 -o output_dir/
    python main.py pattern.txt -o output.txt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from grid import Grid
from solver import find_predecessor, find_predecessors_n_steps
from garden_of_eden import find_closest_reversible_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse Conway's Game of Life",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Find single predecessor
    python main.py pattern.txt -o predecessor.txt

    # Go back 5 steps
    python main.py pattern.txt -n 5 -o predecessors/

    # Verbose mode
    python main.py pattern.txt -n 3 -o output/ -v

Input format:
    Text file with 0/1 or ./X grid representation.
    Pattern will be centered on 30x30 grid.

    Example patterns:
    - Glider: .X.
              ..X
              XXX

    - Blinker: .X.
               .X.
               .X.
        """,
    )

    parser.add_argument("input", type=str, help="Input pattern file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file or directory (for -n > 1)",
    )
    parser.add_argument(
        "-n",
        "--steps",
        type=int,
        default=1,
        help="Number of steps to go back (default: 1)",
    )
    parser.add_argument(
        "--max-edit",
        type=int,
        default=10,
        help="Max edit distance for reversible search (default: 10)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="Candidates per edit distance for reversible search (default: 100)",
    )
    parser.add_argument(
        "--format",
        choices=["binary", "dot", "compact"],
        default="binary",
        help="Output format (default: binary 0/1)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--early-exit",
        action="store_true",
        help="Stop searching after finding first reversible candidate",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load input
    try:
        target = Grid.from_file(args.input)
    except Exception as e:
        print(f"Error loading input: {e}", file=sys.stderr)
        sys.exit(1)

    alive_count = target.count_alive()
    if args.verbose:
        print(f"Loaded pattern with {alive_count} alive cells")
        print(target)
        print()

    if alive_count == 0:
        print("Warning: Empty pattern loaded", file=sys.stderr)

    # Find predecessors
    if args.steps == 1:
        if args.verbose:
            print("Finding predecessor...")

        result = find_predecessor(target)

        if result is None:
            print("No predecessor exists (Garden of Eden)")
            print("Searching for closest reversible state...")

            closest = find_closest_reversible_state(
                target,
                max_edit_distance=args.max_edit,
                candidates_per_distance=args.candidates,
                verbose=args.verbose,
                output_path=args.output,
                early_exit=args.early_exit,
            )

            if closest:
                print(
                    f"Found reversible state (similarity distance: {closest.similarity:.1f})"
                )

                # Save both modified state and its predecessor
                output_path = Path(args.output)

                # Save predecessor
                closest.predecessor.to_file(output_path, args.format)
                print(f"Saved predecessor to {args.output}")

                # Save modified state
                modified_path = output_path.with_stem(
                    output_path.stem + "_modified")
                closest.modified_grid.to_file(modified_path, args.format)
                print(f"Saved modified (reversible) state to {modified_path}")

                if args.verbose:
                    print("\nModified state:")
                    print(closest.modified_grid)
                    print("\nPredecessor:")
                    print(closest.predecessor)
            else:
                print("Could not find nearby reversible state")
                sys.exit(1)
        else:
            result.to_file(args.output, args.format)
            print(f"Saved predecessor to {args.output}")

            if args.verbose:
                print("\nPredecessor:")
                print(result)
    else:
        # Multiple steps
        if args.verbose:
            print(f"Finding {args.steps} predecessors...")

        # Try to go back all steps
        chain = find_predecessors_n_steps(
            target, args.steps, verbose=args.verbose)

        if chain is None:
            print(f"Could not go back {args.steps} steps (hit Garden of Eden)")
            print("Attempting recovery with closest reversible state...")

            # Try to find how far we can go, then recover
            current = target
            successful_steps = 0
            predecessors = []

            for i in range(args.steps):
                pred = find_predecessor(current)
                if pred is None:
                    print(f"  Hit Garden of Eden at step {i + 1}")
                    # Try to find closest reversible
                    closest = find_closest_reversible_state(
                        current,
                        max_edit_distance=args.max_edit,
                        candidates_per_distance=args.candidates,
                        verbose=args.verbose,
                        output_path=None,  # Don't save intermediate for multi-step
                        early_exit=args.early_exit,
                    )
                    if closest:
                        print(
                            f"  Found reversible alternative (similarity: {closest.similarity:.1f})"
                        )
                        predecessors.append(closest.predecessor)
                        current = closest.predecessor
                        successful_steps += 1
                    else:
                        print("  Could not find reversible alternative")
                        break
                else:
                    predecessors.append(pred)
                    current = pred
                    successful_steps += 1

            if successful_steps == 0:
                print("Could not find any predecessors")
                sys.exit(1)

            chain = predecessors
            print(
                f"Successfully went back {successful_steps} steps (with recovery)")

        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, pred in enumerate(chain):
            step_num = args.steps - i
            step_file = output_dir / f"step_{step_num:03d}.txt"
            pred.to_file(step_file, args.format)

            if args.verbose:
                print(f"\nStep {step_num}:")
                print(pred)

        print(f"Saved {len(chain)} predecessors to {args.output}/")


if __name__ == "__main__":
    main()
