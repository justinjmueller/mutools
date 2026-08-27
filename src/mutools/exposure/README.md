# mutools.exposure

Infrastructure for defining, composing, and applying quality cuts to beam spill DataFrames.

## Modules

### `cut`

| Symbol | Description |
|---|---|
| `QualityCut` | A callable, composable cut backed by an expression tree. |
| `CutSummary` | Hierarchical pass-rate breakdown returned by `QualityCut.summary()`. |
| `NodeSummary` | One node in the `CutSummary` evaluation tree. |

### `correlations`

| Symbol | Description |
|---|---|
| `CutCorrelations` | Pairwise and individual statistics returned by `QualityCut.correlations()`. |

### `registry`

| Symbol | Description |
|---|---|
| `CutRegistry` | Registry of cut factories for a beam context. Supports TOML-driven instantiation. |

---

## `CutRegistry`

**Constructor:**

```python
CutRegistry(name: str = "")
```

`name` is used only in error messages.

**`@registry.quality_cut` decorator:**

Registers a function as a cut factory. The decorated function receives a `pd.DataFrame` as its first argument and must return a boolean `pd.Series`. All additional parameters must be keyword-only; they are bound at construction time when the factory is called.

```python
bnb = CutRegistry("bnb")

@bnb.quality_cut(columns=["E1DCBOB"])
def good_pot(df, *, threshold: float) -> pd.Series:
    return df["E1DCBOB"] > threshold
```

The `columns` argument declares which DataFrame columns the predicate reads. `QualityCut.__call__` checks for them before evaluation and raises `KeyError` with a clear message if any are missing. `columns` may be omitted when no column check is needed.

The decorator may also be used without parentheses when no `columns` argument is required:

```python
@bnb.quality_cut
def always_pass(df) -> pd.Series:
    return pd.Series(True, index=df.index)
```

**`registry[name]`:**

Returns the factory registered under `name`. Raises `KeyError` with a list of available names if not found.

**`registry.load_cuts(path)`:**

Instantiates cuts from a TOML file. Entries under `[cuts]` are resolved in definition order, so composite cuts may reference earlier cuts by name.

Leaf entry:

```toml
[cuts.good_pot]
function  = "good_pot"
threshold = 0.0
```

Composite entry referencing named cuts:

```toml
[cuts.standard_bnb]
operator = "and"
operands = ["good_pot", "no_inhibit"]
```

Composite entry with inline definitions:

```toml
[cuts.strict_bnb]
operator = "and"
operands = [
    {function = "good_pot", threshold = 0.0},
    {function = "no_inhibit"},
]
```

Supported operators: `"and"`, `"or"`, `"not"`. Returns `dict[str, QualityCut]`.

---

## `QualityCut`

Instances are created by calling a factory registered with `CutRegistry`:

```python
cut = good_pot(threshold=0.0)
```

**Composition operators:**

| Expression | Result |
|---|---|
| `a & b` | Passes rows where both `a` and `b` pass. |
| `a \| b` | Passes rows where either `a` or `b` passes. |
| `~a` | Passes rows where `a` does not pass. |

Operators can be chained arbitrarily:

```python
cut = (good_pot(threshold=0.0) & no_inhibit() & ~veto()).named("standard_bnb")
```

**`.named(label)`:**

Returns a copy of the cut with a user-assigned display label. Labeled nodes are never absorbed by the summary flattener, so the label always appears as its own level in the hierarchy.

**`cut(df)`:**

Applies the cut to `df` and returns a boolean `pd.Series`. Raises `KeyError` if any declared columns are absent.

**`cut.summary(df)` → `CutSummary`:**

Evaluates the cut and every sub-cut against `df`. Sub-cuts shared across branches are evaluated once. See `CutSummary` below.

**`cut.correlations(df)` → `CutCorrelations`:**

Computes pairwise and individual statistics for all leaf predicates. See `CutCorrelations` below.

**`cut.to_dict()`:**

Serializes the expression tree to a TOML-compatible dictionary.

---

## `CutSummary`

Returned by `QualityCut.summary()`.

| Attribute | Type | Description |
|---|---|---|
| `total` | `int` | Total rows in the evaluated DataFrame. |
| `passed` | `int` | Rows passing the composed cut. |
| `pass_rate` | `float` | `passed / total`; `nan` when `total` is zero. |
| `tree` | `NodeSummary` | Root of the hierarchical evaluation tree. |
| `per_predicate` | `dict[str, tuple[int, int]]` | Flat mapping of leaf name → `(n_passed, n_total)`. |

`repr(summary)` prints a human-readable hierarchy with pass counts and rates at every level. Unlabeled nodes at the same operator level are flattened into a single line to reduce clutter:

```
standard_bnb: 1800/2000 (90.0%)
  good_pot: 1900/2000 (95.0%)
  no_inhibit: 1850/2000 (92.5%)
```

---

## `CutCorrelations`

Returned by `QualityCut.correlations()`. Leaf cuts are taken in left-to-right expression-tree order with duplicates removed by name.

| Attribute | Type | Description |
|---|---|---|
| `phi` | `pd.DataFrame` | Symmetric NxN phi (Pearson) coefficients between the failure masks of each leaf-cut pair. |
| `conditional` | `pd.DataFrame` | Asymmetric NxN matrix; entry (i, j) = P(fail j \| fail i). |
| `unique_rejection` | `pd.Series` | Per-cut fraction of total spills rejected exclusively by that cut and no other. |
| `total_rejection` | `pd.Series` | Per-cut fraction of total spills rejected, regardless of other cuts. |
| `waterfall` | `pd.DataFrame` | Cumulative pass counts and rates as cuts are applied in order. Indexed by cut name; columns: `passed`, `total`, `pass_rate`, `incremental_loss`. |

**Plot methods** — all accept `ax`, `title`, and `path` keyword arguments:

| Method | Description |
|---|---|
| `plot_phi(...)` | Phi coefficient matrix as a diverging heatmap (RdBu_r, range −1 to 1). |
| `plot_conditional(...)` | Conditional failure probability matrix as a sequential heatmap (YlOrRd, range 0 to 1). |
| `plot_unique_rejection(...)` | Grouped horizontal bar chart comparing total vs. unique rejection per cut. |
| `plot_waterfall(...)` | Horizontal bar chart of cumulative pass rate with incremental loss annotations. |
| `plot(...)` | All four analyses in a single 2×2 figure. Accepts `title` and `path`. |

Each plot method returns the `matplotlib.figure.Figure` it drew into.

---

## Usage

```python
import mutools.exposure as exp
import pandas as pd

bnb = exp.CutRegistry("bnb")

@bnb.quality_cut(columns=["E1DCBOB"])
def good_pot(df, *, threshold: float) -> pd.Series:
    return df["E1DCBOB"] > threshold

@bnb.quality_cut(columns=["inhibit"])
def no_inhibit(df) -> pd.Series:
    return df["inhibit"] == 0

# Compose and label
cut = (good_pot(threshold=0.0) & no_inhibit()).named("standard_bnb")

# Apply
df: pd.DataFrame = ...   # your spill DataFrame
mask = cut(df)
good_spills = df[mask]

# Per-predicate pass rates
summary = cut.summary(df)
print(summary)
print(summary.per_predicate)

# Correlation analysis
corr = cut.correlations(df)
print(corr.waterfall)
corr.plot(title="BNB quality cuts", path="figures/bnb_correlations.pdf")

# Load from TOML
cuts = bnb.load_cuts("cuts.toml")
standard = cuts["standard_bnb"]
```
