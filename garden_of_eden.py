"""
Garden of Eden handling for Game of Life.

When no predecessor exists (Garden of Eden), find the closest
pattern that IS reversible by modifying cells.
"""

from __future__ import annotations
import random
import time
import sys
from dataclasses import dataclass

from grid import Grid, GRID_SIZE
from solver import find_predecessor
from similarity import visual_similarity


@dataclass
class CellInfo:
    """Information about a cell for modification priority."""

    x: int
    y: int
    priority: int
    is_alive: bool


def get_modifiable_cells(grid: Grid) -> list[CellInfo]:
    """
    Get cells sorted by modification priority (lower = modify first).

    Priority logic (matching JS):
    - Alive + (neighbors <= 1 or >= 4): priority 1 (unstable, modify first)
    - Dead + neighbors > 0: priority 2 (nearby dead cells)
    - Alive + (neighbors == 2 or 3): priority 3 (stable alive)
    - Dead + neighbors == 0: priority 4 (far from pattern)
    """
    cells = []

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            is_alive = grid[y][x]
            neighbors = grid.count_neighbors(x, y)

            if is_alive:
                # Unstable cells (will die) are good targets
                priority = 1 if (neighbors <= 1 or neighbors >= 4) else 3
            else:
                # Dead cells near pattern are better targets
                priority = 2 if neighbors > 0 else 4

            cells.append(
                CellInfo(x=x, y=y, priority=priority, is_alive=is_alive))

    # Sort by priority
    cells.sort(key=lambda c: c.priority)
    return cells


@dataclass
class ReversibleResult:
    """Result of finding closest reversible state."""

    modified_grid: Grid
    predecessor: Grid
    similarity: float


def find_closest_reversible_state(
    original: Grid,
    max_edit_distance: int = 10,
    candidates_per_distance: int = 100,
    verbose: bool = False,
    output_path: str | None = None,
    early_exit: bool = False,
) -> ReversibleResult | None:
    """
    Find the closest reversible state to a Garden of Eden.

    Algorithm (iterative deepening by edit distance):
    1. Try original grid first (might already be reversible)
    2. For edit_distance in 1..max_edit_distance:
       a. Generate candidate grids by flipping cells (prioritized)
       b. For each candidate:
          - Check similarity score (skip if worse than best)
          - Try to find predecessor using SAT solver
          - If successful, update best result and save to output_path
       c. Early exit if similarity < edit_distance * 0.5

    Args:
        output_path: If provided, saves best result whenever improved (for Ctrl+C safety)
        early_exit: If True, return immediately after finding first reversible candidate

    Returns ReversibleResult or None if no reversible state found.
    """
    # Check if original is already reversible
    original_pred = find_predecessor(original)
    if original_pred:
        return ReversibleResult(
            modified_grid=original.copy(),
            predecessor=original_pred,
            similarity=0.0,
        )

    # Handle empty grid
    if original.is_empty():
        return ReversibleResult(
            modified_grid=original.copy(),
            predecessor=Grid(),
            similarity=0.0,
        )

    modifiable_cells = get_modifiable_cells(original)
    high_priority_cells = [c for c in modifiable_cells if c.priority <= 2]

    best_result: ReversibleResult | None = None
    best_similarity = float("inf")

    start_time = time.time()

    # Iterative deepening by edit distance
    for edit_distance in range(1, max_edit_distance + 1):
        dist_start = time.time()
        if verbose:
            print(
                f"  Trying edit distance {edit_distance}/{max_edit_distance}...")

        seen: set[str] = set()
        candidates_checked = 0
        sat_checks = 0

        while candidates_checked < candidates_per_distance:
            candidate = original.copy()
            cells_to_flip: list[CellInfo] = []
            used_indices: set[int] = set()

            # Sample cells with priority weighting
            # First half of attempts use high priority cells
            source_cells = (
                high_priority_cells
                if candidates_checked < candidates_per_distance // 2
                else modifiable_cells
            )

            # Make sure we have enough cells to flip
            if len(source_cells) < edit_distance:
                source_cells = modifiable_cells

            # Select random cells to flip
            attempts = 0
            while len(cells_to_flip) < edit_distance and attempts < 100:
                idx = random.randint(0, len(source_cells) - 1)
                if idx not in used_indices:
                    used_indices.add(idx)
                    cells_to_flip.append(source_cells[idx])
                attempts += 1

            # Flip the selected cells
            for cell in cells_to_flip:
                candidate.cells[cell.y][cell.x] = not candidate.cells[cell.y][cell.x]

            # Skip duplicates
            key = candidate.to_key()
            if key in seen:
                candidates_checked += 1
                continue
            seen.add(key)

            # Quick similarity check
            similarity = visual_similarity(original, candidate)
            if similarity >= best_similarity:
                candidates_checked += 1
                continue

            # Progress indicator
            if verbose:
                sat_checks += 1
                elapsed = time.time() - start_time
                print(
                    f"\r    Checking candidate {sat_checks} (elapsed: {elapsed:.1f}s)...", end="")
                sys.stdout.flush()

            # Try to find predecessor
            predecessor = find_predecessor(candidate)

            if predecessor is not None:
                best_result = ReversibleResult(
                    modified_grid=candidate.copy(),
                    predecessor=predecessor,
                    similarity=similarity,
                )
                best_similarity = similarity

                if verbose:
                    print(
                        f"\n    Found reversible candidate! (similarity: {similarity:.1f})")

                # Save intermediate result if output path provided
                if output_path:
                    from pathlib import Path
                    out = Path(output_path)
                    best_result.predecessor.to_file(out, "binary")
                    modified_path = out.with_stem(out.stem + "_modified")
                    best_result.modified_grid.to_file(modified_path, "binary")
                    if verbose:
                        print(f"    Saved intermediate result to {out}")

                # Early exit if requested or very good match
                if early_exit:
                    if verbose:
                        print(f"    Early exit: returning first reversible candidate")
                    return best_result
                if similarity < edit_distance * 0.5:
                    return best_result

            candidates_checked += 1

        # End of distance level
        dist_elapsed = time.time() - dist_start
        if verbose:
            print(
                f"\n    Distance {edit_distance} done: {sat_checks} SAT checks in {dist_elapsed:.1f}s")
            if best_result is not None:
                print(f"    Best so far: similarity {best_similarity:.1f}")

    return best_result
