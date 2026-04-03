"""
Module for reading PROfit plot data and providing a consolidated set of
methods for data access, manipulation, and visualization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import uproot
from typing import Optional, Tuple, Union
from enum import Enum
from .helpers import mark_axis
from .save import saver, FixedPrecisionScalarFormatter
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class TraceType(Enum):
    """
    Enumeration for the different types of traces that can be present in
    the PROfit plot data. This can be used to identify and access the
    different types of data (e.g., histogram contents, error bands,
    etc.) in a structured way.
    """

    HIST_CONTENTS = "hist1d"
    HIST_ERROR_BAND = "errorband"
    FRAC_SYST = "frac_syst"


class ProfitPlotData:
    """
    Class for reading PROfit plot data serialized into a ROOT file and
    providing access to the underlying data for plotting.

    Attributes
    ---------
    _file_path : str
        The path to the ROOT file containing the PROfit plot data.
    """

    def __init__(self, file_path, scale_by_width: bool = False):
        """
        Initializes the ProfitPlotData object by reading the data from
        the specified ROOT file and organizing it into traces and
        bands.

        Parameters
        ----------
        file_path : str
            The path to the ROOT file containing the PROfit plot data.
        scale_by_width : bool, optional
            Whether the upstream data has been scaled by bin width
            (i.e. stored as density). Default is False. If True, the
            file data is stored as-is for display and raw event counts
            are recovered by multiplying by the bin width.
        """
        self._file_path = file_path
        self._scale_by_width = scale_by_width
        self._raw_data = dict()
        self._data = dict()

        # Read the full set of data into a single DataFrame. We can
        # then filter this down by the "tag" column to get the
        # different types of data (histogram contents, error bands,
        # etc.).
        self._rf = uproot.open(self._file_path)

        # Columns for grouping the data. The "tag" column will be used
        # to filter the data into different types, and the remaining
        # columns will be used to group the data into different traces
        # and bands.
        cols = ["variable", "mode", "detector", "channel", "subchannel", "prefix"]

        # Filter to the "hist1d" tag, which contains the raw histogram
        # contents. This content will have multiple subchannels in
        # addition to the "total" histogram.
        data = pd.DataFrame(self._rf["hist1d"].arrays(library="np"))
        data = data.drop_duplicates()
        self._raw_data[TraceType.HIST_CONTENTS] = dict()
        self._data[TraceType.HIST_CONTENTS] = dict()
        deserialize = ["bin_center", "bin_low_edge", "bin_high_edge", "bin_content"]
        if "bin_error" in data.columns:
            #TODO: Revisit this when the bin_error column is fully implemented
            deserialize.append("bin_error")
        for k, group in data.groupby(cols):
            name = f"{k[0]}:{k[1]}:{k[2]}:{k[3]}:{k[4]}:{k[5]}"
            raw = group[deserialize].to_numpy().copy()
            self._data[TraceType.HIST_CONTENTS][name] = raw

            if self._scale_by_width:
                counts = raw.copy()
                widths = counts[:, 2] - counts[:, 1]
                counts[:, 3] *= widths
                if counts.shape[1] > 4:
                    counts[:, 4] *= widths
                self._raw_data[TraceType.HIST_CONTENTS][name] = counts
            else:
                self._raw_data[TraceType.HIST_CONTENTS][name] = raw

        # Filter to the "errorband" tag, which contains the error band
        # information. This content is only available for the "total"
        # histogram, so we can just grab this directly.
        data = pd.DataFrame(self._rf["errorband"].arrays(library="np"))
        self._raw_data[TraceType.HIST_ERROR_BAND] = dict()
        self._data[TraceType.HIST_ERROR_BAND] = dict()
        deserialize = ["x_value", "y_value", "error_y_low", "error_y_high"]
        for k, group in data.groupby(cols):
            name = f"{k[0]}:{k[1]}:{k[2]}:{k[3]}:{k[4]}:{k[5]}"
            raw = group[deserialize].to_numpy()
            self._data[TraceType.HIST_ERROR_BAND][name] = raw

            if self._scale_by_width:
                counts = raw.copy()
                tmptrace = self._data[TraceType.HIST_CONTENTS][name]
                widths = tmptrace[:, 2] - tmptrace[:, 1]
                counts[:, 1] *= widths
                counts[:, 2] *= widths
                counts[:, 3] *= widths
                self._raw_data[TraceType.HIST_ERROR_BAND][name] = counts
            else:
                self._raw_data[TraceType.HIST_ERROR_BAND][name] = raw

        # Infer grid dimensions from non-ratio hist1d entries.  Ratio
        # entries carry RATIO_* prefixes; regular entries use CV, DATA,
        # etc.  These counts are needed for decoding the flat detector
        # index stored in the subchannel field of ratio traces.
        _reg = [k for k in self._data[TraceType.HIST_CONTENTS]
                if not k.split(":")[5].startswith("RATIO_")]
        self._n_detectors = (max(int(k.split(":")[2]) for k in _reg) + 1) if _reg else 1
        self._n_channels  = (max(int(k.split(":")[3]) for k in _reg) + 1) if _reg else 1

        # Filter to the "frac_syst" tag, which contains the fractional
        # systematic uncertainties.
        data = pd.DataFrame(self._rf["frac_syst"].arrays(library="np"))
        self._raw_data[TraceType.FRAC_SYST] = dict()
        self._data[TraceType.FRAC_SYST] = dict()
        deserialize = ["bin_center", "bin_low_edge", "bin_high_edge", "bin_content"]
        # Keys are normalised to mode:detector:channel:detector2:tag:systname
        # regardless of whether the file contains the detector2 branch.  For
        # older files that lack the branch, detector2 is synthesised as equal
        # to detector (the standard non-ratio case).
        if "detector2" in data.columns:
            cols = ["mode", "detector", "channel", "detector2", "tag", "systname"]
            for k, group in data.groupby(cols):
                name = f"{k[0]}:{k[1]}:{k[2]}:{k[3]}:{k[4]}:{k[5]}"
                raw = group[deserialize].to_numpy()
                self._raw_data[TraceType.FRAC_SYST][name] = raw
                self._data[TraceType.FRAC_SYST][name] = raw
        else:
            cols = ["mode", "detector", "channel", "tag", "systname"]
            for k, group in data.groupby(cols):
                name = f"{k[0]}:{k[1]}:{k[2]}:{k[1]}:{k[3]}:{k[4]}"
                raw = group[deserialize].to_numpy()
                self._raw_data[TraceType.FRAC_SYST][name] = raw
                self._data[TraceType.FRAC_SYST][name] = raw

    def get_ratio_trace(
        self,
        detector_num: int,
        detector_den: int,
        channel: int,
        prefix: str,
        trace_type: TraceType = TraceType.HIST_CONTENTS,
    ) -> np.ndarray:
        """
        Retrieve a ratio trace for a numerator/denominator detector pair.

        The ratio entries in hist1d and errorband encode the numerator
        histogram's flat index in the (mode, detector, channel) fields
        and store the denominator's flat index in the subchannel field.
        This method resolves that encoding using the inferred grid
        dimensions and returns the matching array.

        Parameters
        ----------
        detector_num : int
            Index of the numerator detector.
        detector_den : int
            Index of the denominator detector.
        channel : int
            Channel index (only same-channel ratios are stored).
        prefix : str
            One of ``"RATIO_CV"``, ``"RATIO_BF"``, ``"RATIO_DATA"``,
            or ``"RATIO_ERR"``.
        trace_type : TraceType, optional
            ``TraceType.HIST_CONTENTS`` (default) or
            ``TraceType.HIST_ERROR_BAND``.

        Returns
        -------
        np.ndarray
            The requested trace array.  For HIST_CONTENTS the columns
            are ``[bin_center, bin_low_edge, bin_high_edge,
            bin_content]``; for HIST_ERROR_BAND they are
            ``[x_value, y_value, error_y_low, error_y_high]``.

        Raises
        ------
        KeyError
            If no matching trace is found.
        """
        nd, nc = self._n_detectors, self._n_channels
        store = self._data[trace_type]
        for key, arr in store.items():
            parts = key.split(":")
            if parts[5] != prefix:
                continue
            if int(parts[2]) != detector_num:
                continue
            if int(parts[3]) != channel:
                continue
            idx_den = int(parts[4])
            # Same-channel constraint: idx_den % nc == channel.
            # Denominator detector:  (idx_den // nc) % nd == detector_den.
            if idx_den % nc == channel and (idx_den // nc) % nd == detector_den:
                return arr
        raise KeyError(
            f"No ratio trace found: detector_num={detector_num}, "
            f"detector_den={detector_den}, channel={channel}, prefix={prefix!r}"
        )

    def get_counts(self, variable: int, detector: int, channel: int, n_subchannels: int) -> list:
        """
        Compute the total event count for each subchannel by summing bin
        contents from the histogram trace of the specified variable.

        Parameters
        ----------
        variable : int
            The variable index to use for the count (typically a raw
            event counter variable from the PROfit configuration).
        detector : int
            The detector index.
        channel : int
            The channel index.
        n_subchannels : int
            The number of subchannels.

        Returns
        -------
        list[float]
            Total counts for each subchannel, in subchannel order.
        """
        return [
            self.get_trace(
                f"{variable}:0:{detector}:{channel}:{si}:CV", TraceType.HIST_CONTENTS, scaled=False
            )[:, 3].sum()
            for si in range(n_subchannels)
        ]

    def get_trace(self, name: str, trace_type: TraceType, scaled: bool = True):
        """
        Retrieves a specific trace from the data based on the provided
        name and trace type.

        Parameters
        ----------
        name : str
            The name of the trace to retrieve.
        trace_type : TraceType
            The type of trace to retrieve.
        scaled : bool, optional
            If True (default), return the scaled data. If False, return
            the raw unscaled data. When scale_by_width is disabled on
            the ProfitPlotData object, both return the same arrays.

        Returns
        -------
        np.ndarray
            The requested trace data.
        """
        name = name.replace("total", "4294967295")
        store = self._data if scaled else self._raw_data
        return store[trace_type][name]


def add_error_band(
    ax: "matplotlib.axes.Axes",
    edges: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | list[np.ndarray],
    label: str = "Flux+Interaction Uncertainty",
    color: str = "black",
    hatch: str = "////",
) -> Patch:
    """
    Add an error band to a plot by filling the area between the upper
    and lower bounds defined by the central values `y` and the
    uncertainties `yerr`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    edges : np.ndarray
        The edges of the histogram bins.
    y : np.ndarray
        The central values of the histogram (e.g., the "total"
        histogram).
    yerr : np.ndarray or list of np.ndarray
        The uncertainties for the error band. This can be either a
        single array of symmetric uncertainties (in which case the
        band will be symmetric around `y`), or a list of two arrays
        containing the lower and upper uncertainties separately (in
        which case the band can be asymmetric).
    label : str, optional
        The label for the error band to be used in the legend, by
        default 'Total Error Band'.

    Returns
    -------
    Patch
        A matplotlib Patch object representing the error band, which
        can be used for the legend.
    """
    ymin = y - yerr[0]
    ymax = y + yerr[1]

    # Make arrays length n+1 for step='post' filling
    ymin_step = np.r_[ymin, ymin[-1]]
    ymax_step = np.r_[ymax, ymax[-1]]

    ax.fill_between(
        edges,
        ymin_step,
        ymax_step,
        step="post",
        facecolor=color,
        alpha=0.15,
        edgecolor=color,
        hatch=hatch,
        linewidth=0.0,
        zorder=1,
    )

    patch = Patch(
        facecolor=color, alpha=0.15, edgecolor=color, hatch=hatch, linewidth=0.0, label=label
    )

    return patch


def add_outline(
    ax: "matplotlib.axes.Axes",
    edges: np.ndarray,
    stack: list[np.ndarray],
) -> None:
    """
    Add an outline to a stacked histogram by plotting the cumulative
    values of the stack as step lines on top of the histogram bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    edges : np.ndarray
        The edges of the histogram bins.
    stack : list[np.ndarray]
        A list of arrays containing the histogram contents for each
        stack component. Each array should have the same shape and
        correspond to the same bins defined by `edges`.

    Returns
    -------
    None.
    """
    cum = np.cumsum(np.vstack(stack), axis=0)
    for k in range(cum.shape[0]):
        ax.stairs(cum[k], edges, fill=False, color="black", linewidth=1.0)


def construct_proxy_stack(
    subchannels: list[str],
    colors: Optional[list] = None,
) -> list[Patch]:
    """
    Construct Patches for the legend that match the colors of the
    stacked histogram bars.

    Parameters
    ----------
    subchannels : list[str]
        List of subchannel names.
    colors : list, optional
        Color specs for each subchannel, in subchannel order. Any
        matplotlib color spec is accepted. Defaults to ``C0``, ``C1``,
        ... from the active color cycle.

    Returns
    -------
    list[Patch]
        List of matplotlib Patch objects for the legend.
    """
    n = len(subchannels)
    resolved = colors if colors is not None else [f"C{i}" for i in range(n)]
    proxy_stack = [
        Patch(facecolor=resolved[i], edgecolor="black", linewidth=1.5, alpha=0.7, label=subchannel)
        for i, subchannel in enumerate(subchannels[::-1])
    ]
    return proxy_stack


def construct_meta_handle(
    profit_version: str,
    selection_version: str,
) -> Patch:
    """
    Construct a Patch for the legend that represents the metadata about
    the plot, such as the PROfit version and selection version.

    Parameters
    ----------
    profit_version : str
        The version of PROfit used to generate the plot.
    selection_version : str
        The version of the selection used to generate the plot.

    Returns
    -------
    Patch
        A matplotlib Patch object representing the metadata, which can
        be used for the legend.
    """
    label = f"PROfit {profit_version}\nmedulla {selection_version}"
    patch = Patch(
        facecolor="none",
        edgecolor="none",
        label=label,
    )
    return patch


def histogram(
    data: ProfitPlotData,
    *,
    variable: int,
    detector: int,
    channel: int,
    xlabel: str,
    ylabel: str,
    code_version: str,
    selection_version: str,
    subchannels: list[str],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    rlim: Optional[Tuple[float, float]] = None,
    ratio: Optional[str] = None,
    disable_systematics: bool = False,
    counter_index: Optional[int] = None,
    counter_fmt: str = ".0f",
    scale_by_width: bool = False,
    detector_label: Optional[str] = None,
    channel_label: Optional[str] = None,
    watermark: Optional[str] = r"$\bf{SBN}$ Internal",
    colors: Optional[list] = None,
    output: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    Create a plot of information from a ProfitPlotData object,
    including the stacked histogram for the subchannels and the error
    band for the total histogram.

    Parameters
    ----------
    data : ProfitPlotData
        The data object containing the traces and bands for the plot.
    variable : int
        The variable number corresponding to the internal numbering
        within the PROfit configuration.
    detector : int
        The detector number corresponding to the internal numbering
        within the PROfit configuration.
    channel : int
        The channel number corresponding to the internal numbering
        within the PROfit configuration.
    xlabel : str
        The label for the x-axis of the plot.
    ylabel : str
        The label for the y-axis of the plot.
    code_version : str
        The version of the code used to generate the plot, to be
        included in the legend for metadata purposes.
    selection_version : str
        The version of the selection used to generate the plot, to be
        included in the legend for metadata purposes.
    subchannels : list[str]
        List of subchannel names corresponding to the traces in the
        data object.
    xlim : Optional[tuple[float, float]]
        The limits for the x-axis of the plot.
    ylim : Optional[tuple[float, float]]
        The limits for the y-axis of the plot.
    rlim : Optional[tuple[float, float]]
        The limits for the y-axis of the ratio panel, if displayed.
    ratio : Optional[str]
        Toggles the optional displaying of a ratio panel below the main
        plot. Some options:
        - 'data' to display the ratio of the total histogram to data
        - 'null' to display the ratio of the total histogram to itself
    disable_systematics : bool
        If True, the error band for the total histogram will not be
        displayed, which can be useful for debugging or when the
        systematic uncertainties are not relevant for the plot.
    counter_index : Optional[int]
        The variable index of the raw event counter. When provided,
        per-subchannel candidate counts are appended to the legend
        labels (e.g. "CC QE (123)").
    counter_fmt : str
        Python format spec applied to the count value. Use a standard
        float spec (e.g. ".0f", ".1f") for raw counts, or a percent
        spec (e.g. ".1%", ".0%") to display the fraction of the total
        instead. Default is ".0f".
    scale_by_width : bool
        If True, retrieve scaled traces (divided by bin width) for
        display. Raw unscaled traces are always used for statistical
        uncertainty calculations. Default is False.
    detector_label : Optional[str]
        An optional label for the detector, displayed as the legend title.
    channel_label : Optional[str]
        An optional channel name displayed on a new line beneath
        ``detector_label`` in the legend title.  Ignored if
        ``detector_label`` is ``None``.
    colors : list, optional
        Color specs for each subchannel, in subchannel order. Any
        matplotlib color spec is accepted. Defaults to the active color
        cycle.
    output : Optional[Path]
        An optional path for saving the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    # Create the figure and axes for the plot
    figure = plt.figure(figsize=(8, 6))
    gspec = figure.add_gridspec(
        nrows=2 if ratio else 1,
        ncols=1,
        height_ratios=[4, 1] if ratio else [1],
        hspace=0.02 if ratio else 0.05,
    )
    ax = figure.add_subplot(gspec[0, 0])

    # Optional ratio axis: share x with the main axis and keep the
    # separation minimal.
    ax_ratio = None
    if ratio:
        ax_ratio = figure.add_subplot(gspec[1, 0], sharex=ax)
        ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.set_xlabel("")

    # Get the traces for each subchannel from the data object using the
    # appropriate naming convention for the subchannel traces. We also 
    # calculate the bin edges from the first trace, assuming all traces 
    # share the same binning (they should).
    traces = [
        data.get_trace(f"{variable}:0:{detector}:{channel}:{si}:CV", TraceType.HIST_CONTENTS, scaled=scale_by_width)
        for si in range(len(subchannels))
    ]
    edges = np.concatenate([np.array([traces[0][0, 1]]), traces[0][:, 2]])

    # Plot the stacked histogram for the subchannels using the bin
    # centers and contents from the traces, and the edges calculated
    # from the first trace. We also create a proxy stack for the legend
    # based on the provided subchannel names.
    if colors is not None:
        ax.set_prop_cycle(color=[mcolors.to_rgba(c) for c in colors])
    ax.hist(
        [trace[:, 0] for trace in traces],
        bins=edges,
        weights=[trace[:, 3] for trace in traces],
        histtype="barstacked",
        edgecolor="none",
        alpha=0.7,
        rasterized=saver.rasterized,
    )
    proxy_stack = construct_proxy_stack(subchannels, colors=colors)[::-1]

    # If a counter variable is provided, append the per-subchannel
    # candidate counts to the legend labels. construct_proxy_stack
    # assigns colors in reversed subchannel order, so proxy_stack[0]
    # visually corresponds to the last subchannel in the stack;
    # reversing counts aligns each count with the correct visual entry.
    if counter_index is not None:
        counts = data.get_counts(counter_index, detector, channel, len(subchannels))
        if counter_fmt.endswith("%"):
            total = sum(counts)
            values = [c / total for c in counts]
        else:
            values = counts
        for patch, value in zip(proxy_stack, values[::-1]):
            patch.set_label(f"{patch.get_label()} ({value:{counter_fmt}})")

    # Add an outline to the stacked histogram by plotting the
    # cumulative values of the stack as step lines on top of the
    # histogram bars.
    add_outline(ax, edges, [trace[:, 3] for trace in traces])

    # Add the error band for the total histogram using the central
    # values and uncertainties from the band trace. We also capture
    # the Patch object returned by the add_error_band function so that
    # we can include it in the legend.
    if not disable_systematics:
        band = data.get_trace(f"{variable}:0:{detector}:{channel}:total:CV", TraceType.HIST_ERROR_BAND, scaled=scale_by_width)
        band_patch = add_error_band(ax, edges, band[:, 1], yerr=[band[:, 2], band[:, 3]])
    # Construct a Patch for the legend that represents the metadata
    # about the plot, such as the PROfit version and selection version,
    # and include this in the legend along with the proxy stack and the
    # band patch. The metadata is important to include in the legend to
    # provide context about how the plot was generated.
    meta_patch = construct_meta_handle(code_version, selection_version)

    # Add the legend for the stacked histogram and the error band using
    # the proxy stack and the band patch.
    ext = [band_patch] if not disable_systematics else []
    legend = ax.legend(handles=proxy_stack + ext + [meta_patch])

    if detector_label is not None:
        title = detector_label if channel_label is None else f"{detector_label}\n{channel_label}"
        legend.set_title(title)
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(14)
        legend.get_title().set_color("#d67a11")
    texts = legend.get_texts()
    texts[-1].set_fontsize(8)
    texts[-1].set_alpha(0.6)

    # The y-axis uses scientific notation. When ytick_precision is set on the
    # global saver, a fixed-precision formatter is applied so tick labels do
    # not change width between frames.
    if saver.ytick_precision is not None:
        ax.yaxis.set_major_formatter(FixedPrecisionScalarFormatter(saver.ytick_precision))
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Set the y-axis label and range for the main axis.
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)

    if not ratio:
        # If the ratio panel is not displayed, we can set the x-axis
        # label and range on the main axis. Otherwise, the x-axis label
        # and range will be set on the ratio axis below.
        ax.set_xlabel(xlabel)
        if xlim is not None:
            ax.set_xlim(xlim)

    else:
        # Get the ratio axis and plot the requested ratio (see the
        # docstring for the `ratio` parameter for options) using the
        # appropriate traces from the data object. We also add a
        # horizontal line at y=1 to guide the eye.
        x = traces[0][:, 0]
        xerr = 0.5 * (traces[0][:, 2] - traces[0][:, 1])

        add_error_band(
            ax_ratio,
            edges,
            np.ones_like(band[:, 1]),
            yerr=[band[:, 2] / band[:, 1], band[:, 3] / band[:, 1]],
            label="",
        )

        if ratio == "data":
            data_trace = data.get_trace(
                f"{variable}:0:{detector}:{channel}:total:DATA", TraceType.HIST_CONTENTS, scaled=scale_by_width
            )
            data_band = data.get_trace(
                f"{variable}:0:{detector}:{channel}:total:DATA", TraceType.HIST_ERROR_BAND, scaled=scale_by_width
            )
            y = data_band[:, 1] / data_trace[:, 3]
            ylo = data_band[:, 2] / data_trace[:, 3]
            yhi = data_band[:, 3] / data_trace[:, 3]

        elif ratio == "null":
            y = np.ones_like(band[:, 1])
            # Get the raw total histogram trace to calculate the statistical
            # uncertainties. Raw (unscaled) counts are required so that
            # sqrt(N)/N correctly represents the Poisson relative error.
            total_trace = data.get_trace(
                f"{variable}:0:{detector}:{channel}:total:CV", TraceType.HIST_CONTENTS, scaled=False
            )
            ylo = np.sqrt(total_trace[:, 3]) / total_trace[:, 3]
            yhi = np.sqrt(total_trace[:, 3]) / total_trace[:, 3]

        else:
            raise ValueError(f"Unsupported ratio option: {ratio!r}")

        # Draw the central ratio as an errorbar-style plot.
        ax_ratio.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=[ylo, yhi],
            fmt="o",
            color="black",
            markersize=4,
            linewidth=1.0,
            capsize=0,
            zorder=3,
        )

        ax_ratio.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_xlabel(xlabel)

        # Set the y-axis range for the ratio panel if specified, and
        # the x-axis range if specified.
        if rlim is not None:
            ax_ratio.set_ylim(rlim)
        if xlim is not None:
            ax_ratio.set_xlim(xlim)

    # Mark the axis with the appropriate label
    mark_axis(ax, watermark, hadj=0.035)

    if output is not None:
        saver.save(figure, output, f"hist_{detector}_{variable}_{code_version}")

    return figure


def ratio(
    data: ProfitPlotData,
    *,
    detector_num: int,
    detector_den: int,
    channel: int,
    xlabel: str,
    code_version: str,
    selection_version: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    rlim: Optional[Tuple[float, float]] = None,
    detector_num_label: Optional[str] = None,
    detector_den_label: Optional[str] = None,
    channel_label: Optional[str] = None,
    show_data: bool = True,
    watermark: Optional[str] = r"$\bf{SBN}$ Internal",
    output: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot the pre- and post-fit uncertainty bands for a detector ratio.

    The main panel overlays the pre-fit systematic band (``RATIO_CV``
    from the errorband tree, drawn in black) and the post-fit systematic
    band (``RATIO_BF`` from the errorband tree, drawn in red).  A step
    line at the central ratio value is drawn for each band.  Optionally,
    data ratio points (``RATIO_DATA`` from hist1d) are overlaid as
    errorbars.

    The sub-panel shows the error improvement ratio (``RATIO_ERR`` from
    hist1d), i.e. post-fit error divided by pre-fit error, as a step
    plot with a dashed reference line at unity.

    Parameters
    ----------
    data : ProfitPlotData
        The data object containing the ratio traces.
    detector_num : int
        Index of the numerator detector.
    detector_den : int
        Index of the denominator detector.
    channel : int
        Channel index.
    xlabel : str
        Label for the x-axis (shown on the sub-panel).
    code_version : str
        PROfit version string included in the legend metadata.
    selection_version : str
        Selection version string included in the legend metadata.
    xlim : tuple[float, float], optional
        x-axis limits.
    ylim : tuple[float, float], optional
        y-axis limits for the main ratio panel.
    rlim : tuple[float, float], optional
        y-axis limits for the error-improvement sub-panel.
    detector_num_label : str, optional
        Display name for the numerator detector.
    detector_den_label : str, optional
        Display name for the denominator detector.
    channel_label : str, optional
        Optional channel name shown beneath the detector pair label in
        the legend title.
    show_data : bool, optional
        If ``True`` (default), overlay data ratio points from
        ``RATIO_DATA`` when they are present in the file.
    watermark : str, optional
        Watermark label placed above the axis.
    output : Path, optional
        Directory in which to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    # --- Fetch traces ------------------------------------------------
    # Sort each array by bin position (column 0) so that the ordering
    # from hist1d and errorband is always consistent — the two trees
    # are filled independently and may arrive in different row orders.
    cv_hist = data.get_ratio_trace(
        detector_num, detector_den, channel, "RATIO_CV", TraceType.HIST_CONTENTS
    )
    cv_hist = cv_hist[cv_hist[:, 0].argsort()]

    pre_band = data.get_ratio_trace(
        detector_num, detector_den, channel, "RATIO_CV", TraceType.HIST_ERROR_BAND
    )
    pre_band = pre_band[pre_band[:, 0].argsort()]

    post_band = data.get_ratio_trace(
        detector_num, detector_den, channel, "RATIO_BF", TraceType.HIST_ERROR_BAND
    )
    post_band = post_band[post_band[:, 0].argsort()]

    err_hist = data.get_ratio_trace(
        detector_num, detector_den, channel, "RATIO_ERR", TraceType.HIST_CONTENTS
    )
    err_hist = err_hist[err_hist[:, 0].argsort()]

    # Bin edges from the CV hist1d trace (bin_low_edge + last bin_high_edge).
    edges = np.r_[cv_hist[:, 1], cv_hist[-1, 2]]

    # --- Layout ------------------------------------------------------
    figure = plt.figure(figsize=(8, 8))
    gspec = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.02)
    ax_main = figure.add_subplot(gspec[0, 0])
    ax_sub  = figure.add_subplot(gspec[1, 0], sharex=ax_main)
    ax_main.tick_params(axis="x", which="both", labelbottom=False)

    # --- Main panel --------------------------------------------------
    pre_patch = add_error_band(
        ax_main, edges, pre_band[:, 1],
        yerr=[pre_band[:, 2], pre_band[:, 3]],
        label="Pre-fit unc.",
        color="black",
        hatch="////",
    )
    ax_main.step(
        edges, np.r_[pre_band[:, 1], pre_band[-1, 1]],
        where="post", color="black", linewidth=1.5, zorder=2,
    )

    post_patch = add_error_band(
        ax_main, edges, post_band[:, 1],
        yerr=[post_band[:, 2], post_band[:, 3]],
        label="Post-fit unc.",
        color="red",
        hatch=r"\\\\",
    )
    ax_main.step(
        edges, np.r_[post_band[:, 1], post_band[-1, 1]],
        where="post", color="red", linewidth=1.5, zorder=2,
    )

    if show_data:
        try:
            dat = data.get_ratio_trace(
                detector_num, detector_den, channel, "RATIO_DATA",
                TraceType.HIST_CONTENTS,
            )
            ax_main.errorbar(
                dat[:, 0],
                dat[:, 3],
                xerr=0.5 * (dat[:, 2] - dat[:, 1]),
                yerr=dat[:, 4] if dat.shape[1] > 4 else None,
                fmt="ko",
                markersize=4,
                linewidth=1.0,
                capsize=2 if dat.shape[1] > 4 else 0,
                zorder=3,
                label="Data",
            )
        except KeyError:
            pass

    # Legend
    num_lbl = detector_num_label if detector_num_label is not None else str(detector_num)
    den_lbl = detector_den_label if detector_den_label is not None else str(detector_den)
    title = f"{num_lbl} / {den_lbl}"
    if channel_label is not None:
        title = f"{title}\n{channel_label}"
    meta_patch = construct_meta_handle(code_version, selection_version)
    h, _ = ax_main.get_legend_handles_labels()
    legend = ax_main.legend(handles=[pre_patch, post_patch] + h + [meta_patch])
    legend.set_title(title)
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_fontsize(14)
    legend.get_title().set_color("#d67a11")
    texts = legend.get_texts()
    texts[-1].set_fontsize(8)
    texts[-1].set_alpha(0.6)

    ax_main.set_ylabel(f"{num_lbl} / {den_lbl}")
    if ylim is not None:
        ax_main.set_ylim(ylim)
    if xlim is not None:
        ax_main.set_xlim(xlim)

    # --- Sub-panel (error improvement) -------------------------------
    err_edges = np.r_[err_hist[:, 1], err_hist[-1, 2]]
    ax_sub.step(
        err_edges, np.r_[err_hist[:, 3], err_hist[-1, 3]],
        where="post", color="C0", linewidth=1.5,
    )
    ax_sub.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_sub.set_ylabel("Post / Pre err.")
    ax_sub.set_xlabel(xlabel)
    if xlim is not None:
        ax_sub.set_xlim(xlim)
    if rlim is not None:
        ax_sub.set_ylim(rlim)

    mark_axis(ax_main, watermark, hadj=0.035)

    if output is not None:
        saver.save(figure, output, f"ratio_{detector_num}_{detector_den}_{channel}_{code_version}")

    return figure


def uncertainty(
    data: ProfitPlotData,
    *,
    detector: int,
    channel: int,
    tags: list,
    xlabel: str,
    code_version: str,
    selection_version: str,
    detector2: Optional[int] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    detector_label: Optional[str] = None,
    watermark: Optional[str] = r"$\bf{SBN}$ Internal",
    colors: Optional[list] = None,
    output: Optional[Path] = None,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """
    Create a step plot of the fractional systematic uncertainties for
    a set of tags (groups of systematics) from a ProfitPlotData object.

    Parameters
    ----------
    data : ProfitPlotData
        The data object containing the fractional systematic traces.
    detector : int
        The detector number corresponding to the internal numbering
        within the PROfit configuration.
    channel : int
        The channel number corresponding to the internal numbering
        within the PROfit configuration.
    tags : list[str]
        The systematic tags to plot, each corresponding to a named
        group of systematics in the data object.
    xlabel : str
        The label for the x-axis of the plot.
    code_version : str
        The version of the code used to generate the plot, to be
        included in the legend for metadata purposes.
    selection_version : str
        The version of the selection used to generate the plot, to be
        included in the legend for metadata purposes.
    detector2 : Optional[int]
        The denominator detector index for ratio-based fractional
        systematics.  When ``None`` (default), ``detector`` is used,
        which corresponds to the standard non-ratio case.
    xlim : Optional[tuple[float, float]]
        The limits for the x-axis of the plot.
    ylim : Optional[tuple[float, float]]
        The limits for the y-axis of the plot.
    detector_label : Optional[str]
        An optional label for the detector, placed as a bold title
        above the legend.
    watermark : Optional[str]
        The watermark label placed above the axis.
    colors : list, optional
        Color specs for each tag, in tag order. Any matplotlib color
        spec is accepted. Defaults to the active color cycle.
    output : Optional[Path]
        Directory in which to save the figure. The filename is
        constructed from ``detector`` and ``code_version``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    _detector2 = detector2 if detector2 is not None else detector

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()

    if colors is not None:
        ax.set_prop_cycle(color=[mcolors.to_rgba(c) for c in colors])
    for tag in tags:
        name = f"0:{detector}:{channel}:{_detector2}:{tag}:SUM"
        trace = data.get_trace(name, TraceType.FRAC_SYST)

        # The trace shape is (N, 4): bin_center, bin_low_edge,
        # bin_high_edge, value. We append the last bin's right edge and
        # repeat the last value to close the step correctly.
        ax.step(
            np.concatenate([trace[:, 1], trace[-1:, 2]]),
            np.concatenate([trace[:, 3], trace[:, 3][-1:]]),
            where="post",
            label=tag,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fractional Systematic Uncertainty")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    h, _ = ax.get_legend_handles_labels()
    meta_patch = construct_meta_handle(code_version, selection_version)
    legend = ax.legend(handles=h + [meta_patch])

    if detector_label is not None:
        legend.set_title(detector_label)
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(14)
        legend.get_title().set_color("#d67a11")
    texts = legend.get_texts()
    texts[-1].set_fontsize(8)
    texts[-1].set_alpha(0.6)

    mark_axis(ax, watermark, hadj=0.035)

    if output is not None:
        saver.save(figure, output, f"uncertainty_{detector}_{code_version}")

    return figure


def overlay(
    data: ProfitPlotData,
    *,
    variable: Union[int, list[int]],
    detectors: list[int],
    channel: int,
    xlabel: str,
    ylabel: str,
    code_version: str,
    selection_version: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    detector_labels: Optional[list[str]] = None,
    channel_label: Optional[str] = None,
    watermark: Optional[str] = r"$\bf{SBN}$ Internal",
    colors: Optional[list] = None,
    scale_by_width: bool = False,
    output: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    Overlay the total CV spectrum for multiple detectors on a single plot.

    Each detector is drawn as a step-filled histogram with a semi-transparent
    fill and a fully opaque edge, making it easy to compare shapes across
    detectors.

    Parameters
    ----------
    data : ProfitPlotData
        The data object containing the histogram traces.
    variable : int or list[int]
        The variable index within the PROfit configuration. A single int
        is used for all detectors; a list must match the length of
        *detectors* and maps each entry to the corresponding detector.
    detectors : list[int]
        Detector indices to overlay, in the order they will be drawn and
        labelled.
    channel : int
        The channel index within the PROfit configuration.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    code_version : str
        PROfit version string included in the legend metadata.
    selection_version : str
        Selection version string included in the legend metadata.
    xlim : tuple[float, float], optional
        x-axis limits.
    ylim : tuple[float, float], optional
        y-axis limits.
    detector_labels : list[str], optional
        Display names for each detector, in the same order as *detectors*.
        Defaults to the string representation of each detector index.
    channel_label : optional[str]
        Optional legend title shown above the detector entries.
    watermark : str, optional
        Watermark label placed above the axis.
    colors : list, optional
        Color specs for each detector, in the same order as *detectors*.
        Any matplotlib color spec is accepted. Defaults to ``C0``,
        ``C1``, ... from the active color cycle.
    scale_by_width : bool, optional
        If ``True``, retrieve bin-width-scaled traces. Default is ``False``.
    output : Path, optional
        Directory in which to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    variables = [variable] * len(detectors) if isinstance(variable, int) else variable

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()

    for i, (detector, var) in enumerate(zip(detectors, variables)):
        color = colors[i] if colors is not None else f"C{i}"
        trace = data.get_trace(
            f"{var}:0:{detector}:{channel}:total:CV",
            TraceType.HIST_CONTENTS,
            scaled=scale_by_width,
        )
        edges = np.concatenate([np.array([trace[0, 1]]), trace[:, 2]])
        label = detector_labels[i] if detector_labels is not None else str(detector)
        ax.hist(
            trace[:, 0],
            bins=edges,
            weights=trace[:, 3],
            histtype="stepfilled",
            fc=mcolors.to_rgba(color, alpha=0.5),
            ec=mcolors.to_rgba(color, alpha=1.0),
            label=label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    meta_patch = construct_meta_handle(code_version, selection_version)
    h, _ = ax.get_legend_handles_labels()
    legend = ax.legend(handles=h + [meta_patch])
    if channel_label is not None:
        legend.set_title(channel_label)
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(14)
        legend.get_title().set_color("#d67a11")
    texts = legend.get_texts()
    texts[-1].set_fontsize(8)
    texts[-1].set_alpha(0.6)

    if saver.ytick_precision is not None:
        ax.yaxis.set_major_formatter(FixedPrecisionScalarFormatter(saver.ytick_precision))
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    mark_axis(ax, watermark, hadj=0.035)

    if output is not None:
        var_str = "-".join(str(v) for v in variables)
        saver.save(figure, output, f"overlay_{channel}_{var_str}_{code_version}")

    return figure
