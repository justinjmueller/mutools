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

fig = prism_schematic()                          # display in notebook
fig = prism_schematic(n_bins=6, oaa_max=2.0)    # custom binning
fig = prism_schematic(output="figures/")        # save to figures/prism_schematic.pdf
```

## Styles

```python
import mutools.plotting as mp

mp.list_styles()          # list available style sheets
mp.use_style("rootlike")  # apply a style
```