# Introduction
The `mutools` package is a consolidated collection of often-reused functions that I have developed over time for various projects. It serves as a personal utility library to enhance organization and efficiency in coding tasks.

# Installation

## Requirements

- Python >= 3.9

## Using `uv` (recommended)

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd mutools
uv sync
```

To include development dependencies (e.g. `ruff`):

```bash
uv sync --group dev
```

## Using `pip`

Clone the repository, create a virtual environment, and install in editable mode:

```bash
git clone <repo-url>
cd mutools
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

# Plotting (`mutools.plotting`)

## PROfit-style histograms

`run(config)` dispatches a TOML-driven batch of plots. The config can be
a pre-parsed `dict`, a raw TOML string, or a path to a `.toml` file.

```python
import mutools.plotting as mp

mp.run("plots.toml")
```

A minimal TOML configuration looks like:

```toml
[general]
input            = "output.root"
code_version     = "v1.0"
selection_version = "v2.3"
subchannels      = ["CC QE", "CC Res", "NC"]
savefig          = true
output           = "figures/"

# Integer keys map detector/channel indices to display labels.
# channel_label is shown on a new line beneath detector_label in the legend.
[general.detectors]
1 = "SBND"
2 = "ICARUS"

[general.channels]
0 = "CC Inclusive"
1 = "NC"

[[plot]]
type      = "histogram"
variable  = 0
channel   = 0
detectors = [1, 2]
xlabel    = "Reconstructed energy [GeV]"
ylabel    = "Events / bin"

[[plot]]
type      = "uncertainty"
channel   = 0
detectors = [1, 2]
tags      = ["flux", "xsec", "detector"]
xlabel    = "Reconstructed energy [GeV]"
```

The `[general.channels]` block is optional. When present, the label for
`plot.channel` is looked up automatically and passed as `channel_label` to
the plotting function.

Individual plot functions are also available directly:

```python
from mutools.plotting.profit import histogram, uncertainty, overlay, ProfitPlotData

data = ProfitPlotData("output.root")

histogram(
    data,
    variable=0, detector=1, channel=0,
    xlabel="Reconstructed energy [GeV]",
    ylabel="Events / bin",
    code_version="v1.0", selection_version="v2.3",
    subchannels=["CC QE", "CC Res", "NC"],
    detector_label="SBND",
    channel_label="CC Inclusive",   # optional: shown beneath detector label
    colors=["steelblue", "tomato", "C2"],   # optional: one per subchannel
)

uncertainty(
    data,
    detector=1, channel=0,
    tags=["flux", "xsec", "detector"],
    xlabel="Reconstructed energy [GeV]",
    code_version="v1.0", selection_version="v2.3",
    detector_label="SBND",
    colors=["C3", "C5", "#aabbcc"],   # optional: one per tag
)

overlay(
    data,
    variable=0, detectors=[1, 2], channel=0,
    xlabel="Reconstructed energy [GeV]",
    ylabel="Events / bin",
    code_version="v1.0", selection_version="v2.3",
    detector_labels=["SBND", "ICARUS"],
    channel_label="CC Inclusive",   # optional: shown as legend title
    colors=["C1", "C4"],            # optional: one per detector
)
```

All three functions accept an optional `colors` list. Each entry can be
any matplotlib color spec — color cycle indices (``"C0"``, ``"C3"``, ...),
named colors (``"steelblue"``), hex strings, or RGB tuples. When omitted,
the active style's color cycle is used in order.

`overlay` draws the total CV spectrum for each detector as a step-filled
histogram with a semi-transparent fill and solid edge, making it easy to
compare shapes. A different variable can be assigned to each detector by
passing a list to `variable`:

```python
overlay(
    data,
    variable=[0, 1], detectors=[1, 2], channel=0,
    ...
)
```

In TOML, `overlay` produces a single figure covering all listed detectors
rather than one figure per detector:

```toml
[[plot]]
type      = "overlay"
variable  = 0          # or variable = [0, 1] for per-detector variables
channel   = 0
detectors = [1, 2]
xlabel    = "Reconstructed energy [GeV]"
ylabel    = "Events / bin"
colors    = ["C1", "C4"]   # optional: one per detector
```

## Detector ratio plots

`ratio` overlays the pre- and post-fit systematic uncertainty bands for a
numerator/denominator detector pair. The main panel shows the ratio central
value as step lines (black = pre-fit, red = post-fit) with hatched
uncertainty bands. When data ratio points are present in the file they are
overlaid as errorbars, including statistical uncertainties when a
`bin_error` column is available. A sub-panel shows the error improvement
(post-fit error / pre-fit error) as a step plot.

```python
from mutools.plotting.profit import ratio

ratio(
    data,
    detector_num=0, detector_den=1, channel=0,
    xlabel="Reconstructed neutrino energy [GeV]",
    code_version="v1.0", selection_version="v2.3",
    detector_num_label="SBND",
    detector_den_label="ICARUS",
    channel_label="CC Inclusive",   # optional: shown beneath pair label
    show_data=True,                 # optional: overlay RATIO_DATA points
    ylim=(3.0, 10.0),               # optional
    rlim=(0.3, 1.1),                # optional: sub-panel y range
)
```

In TOML, `ratio` takes a detector pair rather than a list:

```toml
[[plot]]
type         = "ratio"
detector_num = 0
detector_den = 1
channel      = 0
xlabel       = "Reconstructed neutrino energy [GeV]"
show_data    = true      # optional, default true
xlim         = [0.0, 3.0]
ylim         = [3.0, 10.0]
rlim         = [0.3, 1.1]
```

Detector labels are looked up from `[general.detectors]` automatically.
The channel label is picked up from `[general.channels]` when present.

## SPINE training performance

`plot_train_performance` plots training metrics from SPINE log files. Multiple
metrics can be overlaid on a single figure, with optional smoothing and
validation checkpoints.

```python
from pathlib import Path
from mutools.plotting import plot_train_performance

plot_train_performance(
    Path("logs/"),
    metrics={"loss": "Total loss"},
    ylabel="Loss",
    smooth=10,
)
```

Pass multiple metrics to overlay them on the same axes:

```python
plot_train_performance(
    Path("logs/"),
    metrics={"loss": "Total loss", "loss_seg": "Segmentation loss"},
    ylabel="Loss",
    smooth=10,
)
```

Validation checkpoints are overlaid when `bpe` (batches per epoch) is provided.
`val_checkpoint_stride` limits the density of plotted checkpoints to avoid
crowding:

```python
plot_train_performance(
    Path("logs/"),
    metrics={"loss": "Total loss"},
    ylabel="Loss",
    bpe=500,
    val_checkpoint_stride=5.0,   # minimum epoch spacing between val. points
    ullabel="SBN Internal",
    urlabel="SBND",
    output=Path("figures/"),
    name="training_loss",
)
```

The `output` and `name` parameters follow the same convention as all other
mutools plotting functions — `output` is a directory and `name` is the filename
stem. The active `saver` settings (DPI, format, etc.) apply.

## PRISM schematic

`prism_schematic` produces a two-panel figure (SBND | ICARUS) showing
the transverse cross-section of each detector overlaid with concentric
rings for the PRISM off-axis angle (OAA) bins.

```python
from mutools.plotting import prism_schematic

fig = prism_schematic()                                       # display in notebook
fig = prism_schematic(n_bins=6, oaa_max=2.0)                  # uniform binning
fig = prism_schematic(bin_edges=[0.0, 0.4, 0.9, 1.4, 1.7])    # explicit edges
fig = prism_schematic(show_cathode=False)                     # hide cathode ring
fig = prism_schematic(output="figures/")                      # save to figures/prism_schematic.pdf
```

## Figure saving

All plotting functions accept an `output` keyword argument. When provided,
the figure is saved to that directory via the module-level `saver` object,
which holds persistent settings (DPI, format, bounding-box trimming).

```python
import mutools.plotting as mp

# Configure once — applies everywhere
mp.saver.configure(fmt="png", dpi=300)

# Temporary override for a single block
with mp.saver.settings(fmt="svg", dpi=600):
    histogram(data, ..., output="figures/")

# Rasterize histogram bars to eliminate inter-bin seam artefacts in PDF/SVG
mp.saver.configure(rasterized=True)

# Fix y-axis tick label precision for consistent width across frames
mp.saver.configure(ytick_precision=1)   # e.g. 1.0 × 10³
```

Supported formats: `eps`, `pdf`, `png`, `ps`, `svg`.

The `rasterized` flag embeds histogram bars as a raster image inside the
vector file, which prevents the thin vertical streaks that PDF/SVG viewers
render between adjacent bins of a stacked histogram.

## Animated GIFs

`create_gif` stitches an ordered sequence of PNG files into an animated GIF
using `imageio`.

```python
import mutools.plotting as mp

mp.create_gif(
    ["frame_0.png", "frame_1.png", "frame_2.png"],
    output="figures/animation.gif",
    fps=2.0,   # frames per second (default: 2.0)
    loop=0,    # 0 = loop forever (default)
)
```

Only PNG inputs are supported. The output directory is created automatically
if it does not exist.

## Styles

```python
import mutools.plotting as mp

mp.list_styles()          # list available style sheets
mp.use_style("rootlike")  # apply a style
```