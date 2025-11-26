# Game of Life Reverser

Python CLI tool that reverses Conway's Game of Life by finding predecessor states using SAT-based constraint solving.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
# Find single predecessor
python main.py pattern.txt -o output.txt

# Go back 5 steps
python main.py pattern.txt -n 5 -o results/

# Verbose mode (recommended)
python main.py pattern.txt -o output.txt -v
```

### Speed Options

```bash
# Fast: stop at first reversible solution found
python main.py pattern.txt -o output.txt -v --early-exit

# Balanced: fewer candidates, lower max edit distance
python main.py pattern.txt -o output.txt -v --max-edit 3 --candidates 50

# Default search saves intermediate results - safe to Ctrl+C anytime
python main.py pattern.txt -o output.txt -v
```

### All Options

| Option          | Description                                  | Default  |
| --------------- | -------------------------------------------- | -------- |
| `-n, --steps`   | Number of steps to go back                   | 1        |
| `-o, --output`  | Output file or directory                     | required |
| `-v, --verbose` | Show progress and timing                     | off      |
| `--early-exit`  | Stop after first solution                    | off      |
| `--max-edit`    | Max cell changes for Garden of Eden recovery | 10       |
| `--candidates`  | Candidates per edit distance                 | 100      |
| `--format`      | Output format: binary, dot, compact          | binary   |

## PNG Conversion

```bash
# Convert grid to 32x32 PNG (white=alive, black=dead)
python convert.py grid2png pattern.txt -o pattern.png

# Convert PNG back to grid
python convert.py png2grid pattern.png -o pattern.txt
```

## Input Format

Plain text with `.`/`X` or `0`/`1`:

```
.X.
..X
XXX
```

Small patterns are auto-centered on the 32x32 grid. Full 32x32 grids (space-separated `0 1`) are used as-is.

## Garden of Eden Handling

When a pattern has no predecessor (Garden of Eden), the tool automatically searches for the closest reversible pattern by making small modifications. Progress is shown in verbose mode, and intermediate results are saved as better candidates are found.
