"""Visual similarity metrics for comparing Game of Life grids."""

from __future__ import annotations
import math

from grid import Grid, GRID_SIZE


def hamming_distance(grid_a: Grid, grid_b: Grid) -> int:
    """Count of cells that differ between two grids."""
    diff = 0
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid_a[y][x] != grid_b[y][x]:
                diff += 1
    return diff


def get_centroid(grid: Grid) -> tuple[float, float]:
    """
    Calculate center of mass of alive cells.
    Returns (x, y) coordinates.
    """
    sum_x = 0.0
    sum_y = 0.0
    count = 0

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x]:
                sum_x += x
                sum_y += y
                count += 1

    if count > 0:
        return (sum_x / count, sum_y / count)
    else:
        return (GRID_SIZE / 2, GRID_SIZE / 2)


def count_connected_components(grid: Grid) -> int:
    """
    Count connected components using flood fill.
    8-connectivity (includes diagonals).
    """
    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    components = 0

    def dfs(x: int, y: int) -> None:
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return
        if visited[y][x] or not grid[y][x]:
            return
        visited[y][x] = True
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx != 0 or dy != 0:
                    dfs(x + dx, y + dy)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] and not visited[y][x]:
                dfs(x, y)
                components += 1

    return components


def get_bounding_box(grid: Grid) -> tuple[int, int, int, int] | None:
    """
    Get bounding box of alive cells.
    Returns (min_x, max_x, min_y, max_y) or None if empty.
    """
    min_x = GRID_SIZE
    max_x = 0
    min_y = GRID_SIZE
    max_y = 0
    has_alive = False

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x]:
                has_alive = True
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    return (min_x, max_x, min_y, max_y) if has_alive else None


def visual_similarity(original: Grid, candidate: Grid) -> float:
    """
    Composite visual similarity score (lower = more similar).

    Weights (matching JS implementation):
    - Hamming distance: 1.0
    - Centroid shift: 2.0
    - Connectivity difference: 5.0
    - Bounding box difference: 1.5
    """
    # Hamming distance
    hamming = hamming_distance(original, candidate)

    # Centroid shift
    c_a = get_centroid(original)
    c_b = get_centroid(candidate)
    centroid_shift = math.sqrt((c_a[0] - c_b[0]) ** 2 + (c_a[1] - c_b[1]) ** 2)

    # Connectivity difference
    connectivity = abs(
        count_connected_components(original) -
        count_connected_components(candidate)
    )

    # Bounding box difference
    bb_a = get_bounding_box(original)
    bb_b = get_bounding_box(candidate)

    if bb_a and bb_b:
        bounding_box_diff = (
            abs(bb_a[0] - bb_b[0])
            + abs(bb_a[1] - bb_b[1])
            + abs(bb_a[2] - bb_b[2])
            + abs(bb_a[3] - bb_b[3])
        )
    elif bb_a or bb_b:
        bounding_box_diff = 100
    else:
        bounding_box_diff = 0

    # Weighted composite
    return (
        hamming * 1.0
        + centroid_shift * 2.0
        + connectivity * 5.0
        + bounding_box_diff * 1.5
    )
