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
from mutools.plotting.profit import histogram, uncertainty, ProfitPlotData

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
)

uncertainty(
    data,
    detector=1, channel=0,
    tags=["flux", "xsec", "detector"],
    xlabel="Reconstructed energy [GeV]",
    code_version="v1.0", selection_version="v2.3",
    detector_label="SBND",
)
```

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
```

Supported formats: `eps`, `pdf`, `png`, `ps`, `svg`.

The `rasterized` flag embeds histogram bars as a raster image inside the
vector file, which prevents the thin vertical streaks that PDF/SVG viewers
render between adjacent bins of a stacked histogram.

## Styles

```python
import mutools.plotting as mp

mp.list_styles()          # list available style sheets
mp.use_style("rootlike")  # apply a style
```