# mutools.io

Data loading and parsing utilities.

## Modules

### `spine`

Utilities for loading and aggregating SPINE training/validation log files (CSV format).

| Function | Description |
|---|---|
| `load_logs(log_dir, ...)` | Load and aggregate CSV log files from a directory. |

**Parameters for `load_logs`:**

| Parameter | Type | Description |
|---|---|---|
| `log_dir` | `Path` | Directory containing the log CSV files. |
| `pattern` | `str` | Glob pattern to match files (default: `"*.csv"`). |
| `method` | `str` | Aggregation method — `"concat"` or `"mean"` (default: `"concat"`). |
| `bpe` | `int`, optional | Iterations per epoch. Required when `method="mean"`. |

**Aggregation methods:**

- `"concat"` — Concatenates all logs in chronological order (sorted by the iteration number in the filename). Duplicate iterations are averaged. Returns a DataFrame sorted by epoch.
- `"mean"` — Computes per-field mean and standard deviation across all logs, indexed by checkpoint. Requires `bpe` to convert iteration numbers to epochs.

## Usage

```python
from pathlib import Path
from mutools.io import load_logs

# Concatenate all training logs
df = load_logs(Path("logs/"), method="concat")

# Summarize logs by checkpoint
summary = load_logs(Path("logs/"), method="mean", bpe=1000)
```
