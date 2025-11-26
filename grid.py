"""Grid operations and file I/O for Game of Life."""

from __future__ import annotations
from pathlib import Path

GRID_SIZE = 32


class Grid:
    """Represents a Game of Life grid as a 2D boolean array."""

    def __init__(self, cells: list[list[bool]] | None = None):
        """Initialize grid, defaulting to empty 30x30."""
        if cells is None:
            self.cells = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        else:
            self.cells = cells

    def __getitem__(self, y: int) -> list[bool]:
        """Allow grid[y][x] access."""
        return self.cells[y]

    def __eq__(self, other: object) -> bool:
        """Compare grids cell by cell."""
        if not isinstance(other, Grid):
            return False
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.cells[y][x] != other.cells[y][x]:
                    return False
        return True

    def __hash__(self) -> int:
        """Hash for use in sets."""
        return hash(tuple(tuple(row) for row in self.cells))

    def __str__(self) -> str:
        """Pretty print grid using dots and X."""
        lines = []
        for row in self.cells:
            lines.append("".join("X" if cell else "." for cell in row))
        return "\n".join(lines)

    @classmethod
    def from_file(cls, filepath: str | Path) -> Grid:
        """
        Load grid from text file.

        Supports formats:
        - '0 1 0\\n1 1 1' (space-separated 0/1)
        - '010\\n111' (compact binary)
        - '.X.\\nXXX' (dot/X format)
        - '.O.\\nOOO' (dot/O format)

        Auto-centers pattern on 30x30 grid.
        """
        filepath = Path(filepath)
        content = filepath.read_text().strip()
        lines = content.split("\n")

        # Parse pattern
        pattern: list[list[bool]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            row: list[bool] = []
            # Try space-separated format first
            if " " in line:
                parts = line.split()
                for part in parts:
                    row.append(part == "1" or part.upper()
                               == "X" or part.upper() == "O")
            else:
                # Compact format
                for char in line:
                    if char in "1XxOo":
                        row.append(True)
                    elif char in "0.":
                        row.append(False)
                    # Skip unknown characters

            if row:
                pattern.append(row)

        if not pattern:
            return cls()

        # Center pattern on grid
        return cls._center_pattern(pattern)

    @classmethod
    def _center_pattern(cls, pattern: list[list[bool]]) -> Grid:
        """Center a smaller pattern in a 30x30 grid."""
        h = len(pattern)
        w = max(len(row) for row in pattern) if pattern else 0

        offset_y = (GRID_SIZE - h) // 2
        offset_x = (GRID_SIZE - w) // 2

        grid = cls()
        for y, row in enumerate(pattern):
            for x, cell in enumerate(row):
                ny = offset_y + y
                nx = offset_x + x
                if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                    grid.cells[ny][nx] = cell

        return grid

    def to_file(self, filepath: str | Path, fmt: str = "binary") -> None:
        """
        Save grid to file in specified format.

        Formats:
        - 'binary': '0 1 0\\n1 1 1'
        - 'dot': '.X.\\nXXX'
        - 'compact': '010\\n111'
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for row in self.cells:
            if fmt == "binary":
                lines.append(" ".join("1" if cell else "0" for cell in row))
            elif fmt == "dot":
                lines.append("".join("X" if cell else "." for cell in row))
            else:  # compact
                lines.append("".join("1" if cell else "0" for cell in row))

        filepath.write_text("\n".join(lines) + "\n")

    def copy(self) -> Grid:
        """Return deep copy of grid."""
        return Grid([row[:] for row in self.cells])

    def count_neighbors(self, x: int, y: int) -> int:
        """Count alive neighbors for cell at (x, y)."""
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.cells[ny][nx]:
                        count += 1
        return count

    def step_forward(self) -> Grid:
        """Return next generation using standard Game of Life rules."""
        new_grid = Grid()
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                neighbors = self.count_neighbors(x, y)
                if self.cells[y][x]:
                    # Alive: survives with 2 or 3 neighbors
                    new_grid.cells[y][x] = neighbors in (2, 3)
                else:
                    # Dead: born with exactly 3 neighbors
                    new_grid.cells[y][x] = neighbors == 3
        return new_grid

    def produces_target(self, target: Grid) -> bool:
        """Check if this grid produces target after one step."""
        return self.step_forward() == target

    def get_alive_cells(self) -> set[tuple[int, int]]:
        """Return set of (x, y) coordinates of alive cells."""
        alive = set()
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.cells[y][x]:
                    alive.add((x, y))
        return alive

    def get_relevant_region(self, radius: int = 2) -> set[tuple[int, int]]:
        """
        Return cells within radius of any alive cell.
        Used to reduce SAT problem size.
        """
        relevant = set()
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.cells[y][x]:
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                                relevant.add((nx, ny))
        return relevant

    def count_alive(self) -> int:
        """Count total alive cells."""
        return sum(sum(row) for row in self.cells)

    def is_empty(self) -> bool:
        """Check if grid has no alive cells."""
        return self.count_alive() == 0

    def to_key(self) -> str:
        """Convert to string key for deduplication."""
        return "".join("1" if cell else "0" for row in self.cells for cell in row)
