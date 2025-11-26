"""SAT-based predecessor finding for Game of Life."""

from __future__ import annotations
from pysat.solvers import Glucose4
from pysat.card import CardEnc, EncType

from grid import Grid, GRID_SIZE


class PredecessorSolver:
    """
    SAT-based solver for finding Game of Life predecessors.

    Constraint Encoding (matching JS implementation):

    For target cell that should be ALIVE:
      (cell AND 2 <= neighbors <= 3) OR (NOT cell AND neighbors = 3)

    For target cell that should be DEAD:
      (cell AND (neighbors <= 1 OR neighbors >= 4)) OR (NOT cell AND neighbors != 3)
    """

    def __init__(self, target: Grid):
        self.target = target
        self.var_map: dict[tuple[int, int], int] = {}
        self.next_var = 1
        self.clauses: list[list[int]] = []

    def _get_var(self, x: int, y: int) -> int:
        """Get or create SAT variable for cell (x, y)."""
        if (x, y) not in self.var_map:
            self.var_map[(x, y)] = self.next_var
            self.next_var += 1
        return self.var_map[(x, y)]

    def _get_neighbor_vars(
        self, x: int, y: int, relevant: set[tuple[int, int]]
    ) -> list[int]:
        """
        Get SAT variables for neighbors of (x, y).
        Only includes cells in the relevant region.
        """
        neighbors = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if (nx, ny) in relevant:
                        neighbors.append(self._get_var(nx, ny))
        return neighbors

    def _add_clause(self, clause: list[int]) -> None:
        """Add a clause to the formula."""
        self.clauses.append(clause)

    def _add_clauses(self, clauses: list[list[int]]) -> None:
        """Add multiple clauses."""
        self.clauses.extend(clauses)

    def _encode_at_least(self, lits: list[int], k: int) -> list[list[int]]:
        """Encode: at least k of the literals are true."""
        if k <= 0:
            return []  # Always satisfied
        if k > len(lits):
            return [[]]  # Unsatisfiable - return empty clause
        if k == len(lits):
            # All must be true
            return [[lit] for lit in lits]

        cnf = CardEnc.atleast(
            lits=lits, bound=k, top_id=self.next_var, encoding=EncType.seqcounter
        )
        self.next_var = cnf.nv + 1
        return list(cnf.clauses)

    def _encode_at_most(self, lits: list[int], k: int) -> list[list[int]]:
        """Encode: at most k of the literals are true."""
        if k >= len(lits):
            return []  # Always satisfied
        if k < 0:
            return [[]]  # Unsatisfiable

        cnf = CardEnc.atmost(
            lits=lits, bound=k, top_id=self.next_var, encoding=EncType.seqcounter
        )
        self.next_var = cnf.nv + 1
        return list(cnf.clauses)

    def _encode_exactly(self, lits: list[int], k: int) -> list[list[int]]:
        """Encode: exactly k of the literals are true."""
        if k < 0 or k > len(lits):
            return [[]]  # Unsatisfiable

        cnf = CardEnc.equals(
            lits=lits, bound=k, top_id=self.next_var, encoding=EncType.seqcounter
        )
        self.next_var = cnf.nv + 1
        return list(cnf.clauses)

    def _new_aux_var(self) -> int:
        """Create a new auxiliary variable."""
        var = self.next_var
        self.next_var += 1
        return var

    def _encode_implies(self, condition: int, consequence: list[list[int]]) -> None:
        """
        Encode: if condition then consequence clauses must hold.
        Uses Tseitin transformation.
        """
        # condition => clause is equivalent to: NOT condition OR clause
        for clause in consequence:
            self._add_clause([-condition] + clause)

    def _encode_alive_constraint(
        self, cell_var: int, neighbor_vars: list[int]
    ) -> None:
        """
        Encode: target cell should be ALIVE after evolution.

        (cell AND 2 <= neighbors <= 3) OR (NOT cell AND neighbors = 3)

        Strategy: Create auxiliary variables for each disjunct, then OR them.
        """
        if len(neighbor_vars) == 0:
            # No neighbors means this cell can't survive or be born
            # This is UNSAT for an alive target
            self._add_clause([])  # Empty clause = UNSAT
            return

        # Auxiliary variable for survival condition
        survival = self._new_aux_var()

        # Auxiliary variable for birth condition
        birth = self._new_aux_var()

        # Either survival or birth must be true
        self._add_clause([survival, birth])

        # Encode survival = (cell AND 2 <= neighbors <= 3)
        # survival => cell
        self._add_clause([-survival, cell_var])
        # survival => at_least_2(neighbors)
        at_least_2 = self._encode_at_least(neighbor_vars, 2)
        self._encode_implies(survival, at_least_2)
        # survival => at_most_3(neighbors)
        at_most_3 = self._encode_at_most(neighbor_vars, 3)
        self._encode_implies(survival, at_most_3)

        # Encode birth = (NOT cell AND neighbors = 3)
        # birth => NOT cell
        self._add_clause([-birth, -cell_var])
        # birth => exactly_3(neighbors)
        exactly_3 = self._encode_exactly(neighbor_vars, 3)
        self._encode_implies(birth, exactly_3)

    def _encode_dead_constraint(self, cell_var: int, neighbor_vars: list[int]) -> None:
        """
        Encode: target cell should be DEAD after evolution.

        (cell AND (neighbors <= 1 OR neighbors >= 4)) OR (NOT cell AND neighbors != 3)

        Equivalent to:
        - death_under = cell AND neighbors <= 1
        - death_over = cell AND neighbors >= 4
        - stay_dead = NOT cell AND neighbors != 3

        At least one of these must be true.
        """
        if len(neighbor_vars) == 0:
            # No neighbors - cell stays dead (no birth possible)
            # Any state is fine, no constraint needed
            return

        # Create auxiliary variables for each case
        death_under = self._new_aux_var()
        death_over = self._new_aux_var()
        stay_dead = self._new_aux_var()

        # At least one must be true
        self._add_clause([death_under, death_over, stay_dead])

        # death_under = cell AND neighbors <= 1
        self._add_clause([-death_under, cell_var])
        at_most_1 = self._encode_at_most(neighbor_vars, 1)
        self._encode_implies(death_under, at_most_1)

        # death_over = cell AND neighbors >= 4
        self._add_clause([-death_over, cell_var])
        at_least_4 = self._encode_at_least(neighbor_vars, 4)
        self._encode_implies(death_over, at_least_4)

        # stay_dead = NOT cell AND neighbors != 3
        # neighbors != 3 is: neighbors <= 2 OR neighbors >= 4
        self._add_clause([-stay_dead, -cell_var])

        # For stay_dead => (neighbors <= 2 OR neighbors >= 4)
        # We create two more aux vars
        not_3_low = self._new_aux_var()
        not_3_high = self._new_aux_var()
        self._add_clause([-stay_dead, not_3_low, not_3_high])

        at_most_2 = self._encode_at_most(neighbor_vars, 2)
        self._encode_implies(not_3_low, at_most_2)

        at_least_4_v2 = self._encode_at_least(neighbor_vars, 4)
        self._encode_implies(not_3_high, at_least_4_v2)

    def solve(self) -> Grid | None:
        """
        Build and solve SAT problem.
        Returns predecessor Grid or None if no solution (Garden of Eden).
        """
        relevant = self.target.get_relevant_region(radius=2)

        if not relevant:
            # Empty target is its own predecessor
            return Grid()

        # Add constraints for each cell in relevant region
        for x, y in relevant:
            cell_var = self._get_var(x, y)
            neighbor_vars = self._get_neighbor_vars(x, y, relevant)
            target_alive = self.target[y][x]

            if target_alive:
                self._encode_alive_constraint(cell_var, neighbor_vars)
            else:
                self._encode_dead_constraint(cell_var, neighbor_vars)

        # Solve
        with Glucose4(bootstrap_with=self.clauses) as solver:
            if solver.solve():
                model = solver.get_model()
                return self._extract_grid(model, relevant)
            return None

    def _extract_grid(
        self, model: list[int], relevant: set[tuple[int, int]]
    ) -> Grid:
        """Convert SAT solution to Grid."""
        grid = Grid()
        model_set = set(model)

        for (x, y), var in self.var_map.items():
            if var in model_set:  # Positive literal means True
                grid.cells[y][x] = True

        return grid


def find_predecessor(target: Grid) -> Grid | None:
    """
    Find a predecessor state that evolves into target.
    Returns None if target is a Garden of Eden.
    """
    solver = PredecessorSolver(target)
    result = solver.solve()

    # Verify the result if found
    if result is not None:
        if not result.produces_target(target):
            # This shouldn't happen, but let's be safe
            return None

    return result


def find_predecessors_n_steps(
    target: Grid, n: int, verbose: bool = False
) -> list[Grid] | None:
    """
    Find chain of n predecessors.
    Returns list [pred_n, pred_n-1, ..., pred_1] or None if any step fails.
    """
    chain = []
    current = target

    for i in range(n):
        if verbose:
            print(f"  Finding predecessor {i + 1}/{n}...")

        pred = find_predecessor(current)
        if pred is None:
            return None  # Hit a Garden of Eden

        chain.append(pred)
        current = pred

    return chain
