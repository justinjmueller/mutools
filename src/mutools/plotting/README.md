# mutools.plotting

Plotting utilities for common visualization tasks.

## Modules

### `helpers`

General-purpose plotting helpers.

| Function | Description |
|---|---|
| `mark_axis(ax, label, ...)` | Place a contextual label above an axis (e.g. "SBN Internal"). Supports left/right alignment and fine-tuned position adjustments via `vadj` and `hadj`. |

### `profit`

Tools for reading and visualizing data from [PROfit](https://github.com/ubneutrinos/PELEE) ROOT output files.

| Symbol | Description |
|---|---|
| `ProfitPlotData` | Reads a PROfit ROOT file and organizes histogram contents, error bands, and fractional systematics into named traces. |
| `ProfitPlotData.get_counts(variable, detector, n_subchannels)` | Returns a list of total event counts per subchannel for the given variable and detector. |
| `TraceType` | Enum identifying the three trace types: `HIST_CONTENTS`, `HIST_ERROR_BAND`, `FRAC_SYST`. |
| `histogram(data, ...)` | High-level function that produces a stacked histogram with an error band and an optional ratio panel. |
| `uncertainty(data, ...)` | Step plot of fractional systematic uncertainties for a set of named systematic tags. |
| `add_error_band(ax, ...)` | Add a hatched uncertainty band to an existing axes. |
| `add_outline(ax, ...)` | Overlay black step-line outlines on a stacked histogram. |
| `construct_proxy_stack(subchannels)` | Build legend `Patch` handles matching the default color cycle. |
| `construct_meta_handle(...)` | Build a legend `Patch` carrying version metadata. |

### `dispatch`

TOML-driven dispatcher that reads a configuration and executes all defined plots without manual looping.

| Function | Description |
|---|---|
| `run(config)` | Accept a TOML file path, a TOML string, or a pre-parsed dict and execute every `[[plot]]` entry. |

**TOML keys:**

| Section | Key | Description |
|---|---|---|
| `[general]` | `input` | Path to the PROfit ROOT file. |
| | `output` | Directory for saved figures. |
| | `savefig` | Boolean — whether to save figures to disk. |
| | `code_version` | Version string included in legend metadata. |
| | `selection_version` | Version string included in legend metadata. |
| | `subchannels` | Ordered list of subchannel names. |
| | `detectors` | List of detector label strings (indexed by detector number). |
| | `counter_index` | Optional variable index for the raw event counter. When set, candidate counts are appended to subchannel legend labels. |
| `[[plot]]` | `counter_fmt` | Python format spec for count display. Float spec (e.g. `".0f"`, `".1f"`) shows raw counts; percent spec (e.g. `".1%"`) shows fraction of total. Default: `".0f"`. Histogram only. |
| `[[plot]]` | `type` | Plot type — `"histogram"` or `"uncertainty"`. |
| | `variable` | Variable index within the PROfit configuration. Histogram only. |
| | `detectors` | List of detector indices to plot. |
| | `xlabel` | x-axis label. |
| | `ylabel` | y-axis label. Histogram only. |
| | `xlim` | Optional `[min, max]` x-axis range. |
| | `ylim` | Optional `[min, max]` y-axis range. |
| | `scale-by-width` | Optional bin-width scaling (`"forward"`, `"backward"`, or `"null"`). Histogram only. |
| | `ratio` | Optional ratio panel mode (`"data"` or `"null"`). Histogram only. |
| | `rlim` | Optional `[min, max]` ratio panel y-axis range. Histogram only. |
| | `disable_systematics` | Boolean — if `true`, the error band is hidden. Default: `false`. Histogram only. |
| | `tags` | List of systematic tag names to plot. Uncertainty only. |
| | `watermark` | Optional watermark string (default: `"$\bf{SBN}$ Internal"`). |

### `style`

Style sheet management for consistent figure appearance.

| Function | Description |
|---|---|
| `use_style(name)` | Apply a bundled `.mplstyle` sheet by name. |
| `list_styles()` | Return a sorted list of all available style names. |

Style sheets live in [`styles/`](styles/) and can be added by dropping a new `.mplstyle` file into that directory.

**Available styles:**

| Name | Description |
|---|---|
| `rootlike` | Serif font (DejaVu Serif / Times New Roman), inward-facing ticks on all four sides with minor ticks, colorblind-friendly color cycle, no legend frame. Mimics the classic ROOT aesthetic. |

## Usage

### Dispatcher (recommended)

```python
from mutools.plotting import run, use_style

use_style("rootlike")
run("plots.toml")            # from a file
run(cfg_string)              # from a TOML string
run(toml.loads(cfg_string))  # from a pre-parsed dict
```

### Manual

```python
from mutools.plotting import use_style
from mutools.plotting.profit import ProfitPlotData, histogram

use_style("rootlike")

data = ProfitPlotData("profit_output.root")
histogram(data, variable=0, detector=0, ...)
```
