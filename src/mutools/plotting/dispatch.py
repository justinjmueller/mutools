"""
Dispatcher for PROfit-style plots driven by a TOML configuration.
"""

from pathlib import Path
from typing import Union

import toml

from .profit import ProfitPlotData, histogram, uncertainty

# Map of plot type strings to their handler functions. New plot types
# can be registered here as the module grows.
_HANDLERS = {
    "histogram": histogram,
    "uncertainty": uncertainty,
}


def run(config: Union[dict, str, Path]) -> None:
    """
    Execute all plots defined in a PROfit TOML configuration.

    The configuration can be supplied as a pre-parsed dict, a raw TOML
    string, or a path to a TOML file. Each ``[[plot]]`` entry is
    dispatched to the appropriate plotting function based on its
    ``type`` key, and is repeated for every detector listed in
    ``plot.detectors``.

    Parameters
    ----------
    config : dict, str, or Path
        The configuration as a pre-parsed dict, a TOML string, or a
        path to a TOML file.

    Raises
    ------
    ValueError
        If a ``[[plot]]`` entry specifies an unsupported ``type``.
    """
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.exists():
            with open(path) as f:
                plots = toml.load(f)
        else:
            plots = toml.loads(str(config))
    else:
        plots = config

    general = plots["general"]
    source = Path(general["input"])
    if general.get("savefig", False) and not general.get("output"):
        raise ValueError("Output path must be specified if savefig is True.")
    output = Path(general["output"]) if general.get("savefig", False) else None

    # Shared keyword arguments that apply to every plot.
    base = {
        "code_version": general["code_version"],
        "selection_version": general["selection_version"],
        "subchannels": general["subchannels"],
        "output": output,
        "counter_index": general.get("counter_index"),
    }

    for plot in plots["plot"]:
        plot_type = plot["type"]
        if plot_type not in _HANDLERS:
            raise ValueError(
                f"Unsupported plot type: {plot_type!r}. Available types: {sorted(_HANDLERS)}"
            )

        handler = _HANDLERS[plot_type]

        # ProfitPlotData is instantiated once per plot entry since
        # scale-by-width is a per-plot setting shared across detectors.
        data = ProfitPlotData(source, plot.get("scale-by-width", "null"))

        for detector in plot["detectors"]:
            # Common kwargs shared across all plot types.
            kwargs = {
                **base,
                "detector": detector,
                "detector_label": general["detectors"][detector],
                "xlabel": plot["xlabel"],
                "xlim": plot.get("xlim"),
                "ylim": plot.get("ylim"),
            }
            if "watermark" in plot:
                kwargs["watermark"] = plot["watermark"]

            # Type-specific kwargs.
            if plot_type == "histogram":
                kwargs.update(
                    {
                        "variable": plot["variable"],
                        "ylabel": plot["ylabel"],
                        "ratio": plot.get("ratio"),
                        "rlim": plot.get("rlim"),
                    }
                )
            elif plot_type == "uncertainty":
                kwargs["tags"] = plot["tags"]

            handler(data, **kwargs)
