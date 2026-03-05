"""
Module for reading PROfit plot data and providing a consolidated set of
methods for data access, manipulation, and visualization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import uproot
from typing import Optional, Tuple
from enum import Enum
from .helpers import mark_axis
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

    def __init__(self, file_path, scale_by_width: str = "null"):
        """
        Initializes the ProfitPlotData object by reading the data from
        the specified ROOT file and organizing it into traces and
        bands.

        Parameters
        ----------
        file_path : str
            The path to the ROOT file containing the PROfit plot data.
        scale_by_width : str, optional
            Whether to scale the histogram contents by the bin width.
            Default is 'null', which means apply no scaling. If set to
            'forward', the histogram contents will be scaled by the bin
            width, whereas 'backward' will apply the inverse scaling.
        """
        self._file_path = file_path
        self._scale_by_width = scale_by_width
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
        self._data[TraceType.HIST_CONTENTS] = dict()
        deserialize = ["bin_center", "bin_low_edge", "bin_high_edge", "bin_content"]
        for k, group in data.groupby(cols):
            name = f"{k[0]}:{k[1]}:{k[2]}:{k[3]}:{k[4]}:{k[5]}"
            raw = group[deserialize].to_numpy().copy()

            # Handle scaling of the histogram contents by the bin width
            if self._scale_by_width != "null":
                widths = raw[:, 2] - raw[:, 1]
                if self._scale_by_width == "forward":
                    raw[:, 3] /= widths
                elif self._scale_by_width == "backward":
                    raw[:, 3] *= widths

            self._data[TraceType.HIST_CONTENTS][name] = raw

        # Filter to the "errorband" tag, which contains the error band
        # information. This content is only available for the "total"
        # histogram, so we can just grab this directly.
        data = pd.DataFrame(self._rf["errorband"].arrays(library="np"))
        self._data[TraceType.HIST_ERROR_BAND] = dict()
        deserialize = ["x_value", "y_value", "error_y_low", "error_y_high"]
        for k, group in data.groupby(cols):
            name = f"{k[0]}:{k[1]}:{k[2]}:{k[3]}:{k[4]}:{k[5]}"
            raw = group[deserialize].to_numpy()

            # Handle scaling of the error band by the bin width, which
            # should be consistent with the scaling applied to the
            # histogram contents. This is a bit more complicated than
            # the scaling of the histogram contents because the error
            # band data does not have the bin edges, so we need to
            # retrieve the corresponding histogram contents trace to
            # calculate the bin widths from the bin centers.
            if self._scale_by_width != "null":
                raw = raw.copy()
                tmptrace = self.get_trace(name, TraceType.HIST_CONTENTS)
                widths = tmptrace[:, 2] - tmptrace[:, 1]
                if self._scale_by_width == "forward":
                    raw[:, 1] /= widths
                    raw[:, 2] /= widths
                    raw[:, 3] /= widths
                elif self._scale_by_width == "backward":
                    raw[:, 1] *= widths
                    raw[:, 2] *= widths
                    raw[:, 3] *= widths

            self._data[TraceType.HIST_ERROR_BAND][name] = raw

        # mode, detector, channel, tag, systname, bin_index, bin_center, bin_low_edge, bin_content
        # Filter to the "frac_syst" tag, which contains the fractional
        # systematic uncertainties.
        data = pd.DataFrame(self._rf["frac_syst"].arrays(library="np"))
        self._data[TraceType.FRAC_SYST] = dict()
        deserialize = ["bin_center", "bin_low_edge", "bin_high_edge", "bin_content"]
        cols = ["mode", "detector", "channel", "tag", "systname"]
        for k, group in data.groupby(cols):
            name = f"{k[0]}:{k[1]}:{k[2]}:{k[3]}:{k[4]}"
            raw = group[deserialize].to_numpy()
            self._data[TraceType.FRAC_SYST][name] = raw

    def get_trace(self, name: str, trace_type: TraceType):
        """
        Retrieves a specific trace from the data based on the provided
        name and trace type.

        Parameters
        ----------
        name : str
            The name of the trace to retrieve.
        trace_type : TraceType
            The type of trace to retrieve.

        Returns
        -------
        np.ndarray
            The requested trace data.
        """
        name = name.replace("total", "4294967295")
        return self._data[trace_type][name]


def add_error_band(
    ax: "matplotlib.axes.Axes",
    edges: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | list[np.ndarray],
    label: str = "Total Error Band",
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
        facecolor="black",
        alpha=0.15,
        edgecolor="black",
        hatch="////",
        linewidth=0.0,
        zorder=1,
    )

    patch = Patch(
        facecolor="black", alpha=0.15, edgecolor="black", hatch="////", linewidth=0.0, label=label
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
) -> list[Patch]:
    """
    Construct Patches for the legend that match the colors of the
    stacked histogram bars.

    Parameters
    ----------
    subchannels : list[str]
        List of subchannel names.

    Returns
    -------
    list[Patch]
        List of matplotlib Patch objects for the legend.
    """
    # Each patch corresponds to a subchannel, and the color is
    # determined by the default color cycle in matplotlib. The legend
    # labels are taken from the provided subchannels list.
    proxy_stack = [
        Patch(facecolor=f"C{i}", edgecolor="black", linewidth=1.5, alpha=0.7, label=subchannel)
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
    xlabel: str,
    ylabel: str,
    code_version: str,
    selection_version: str,
    subchannels: list[str],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    rlim: Optional[Tuple[float, float]] = None,
    ratio: Optional[str] = None,
    detector_label: Optional[str] = None,
    watermark: Optional[str] = r"$\bf{SBN}$ Internal",
    output: Optional[Path] = None,
) -> None:
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
    detector_label : Optional[str]
        An optional label for the detector, which can be used in the
        plot title or annotations.
    output : Optional[Path]
        An optional path for saving the figure.

    Returns
    -------
    None.
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

    # Get the traces for each subchannel and the total band from the
    # data object using the appropriate naming convention for the
    # subchannel traces and the total histogram band. We also calculate
    # the bin edges from the first trace, assuming all traces share the
    # same binning.
    traces = [
        data.get_trace(f"{variable}:0:{detector}:0:{si}:CV", TraceType.HIST_CONTENTS)
        for si in range(len(subchannels))
    ]
    band = data.get_trace(f"{variable}:0:{detector}:0:total:CV", TraceType.HIST_ERROR_BAND)
    edges = np.concatenate([np.array([traces[0][0, 1]]), traces[0][:, 2]])

    # Plot the stacked histogram for the subchannels using the bin
    # centers and contents from the traces, and the edges calculated
    # from the first trace. We also create a proxy stack for the legend
    # based on the provided subchannel names.
    ax.hist(
        [trace[:, 0] for trace in traces],
        bins=edges,
        weights=[trace[:, 3] for trace in traces],
        histtype="barstacked",
        edgecolor="none",
        alpha=0.7,
    )
    proxy_stack = construct_proxy_stack(subchannels)[::-1]

    # Add an outline to the stacked histogram by plotting the
    # cumulative values of the stack as step lines on top of the
    # histogram bars.
    add_outline(ax, edges, [trace[:, 3] for trace in traces])

    # Add the error band for the total histogram using the central
    # values and uncertainties from the band trace. We also capture
    # the Patch object returned by the add_error_band function so that
    # we can include it in the legend.
    band_patch = add_error_band(ax, edges, band[:, 1], yerr=[band[:, 2], band[:, 3]])

    # Construct a Patch for the legend that represents the metadata
    # about the plot, such as the PROfit version and selection version,
    # and include this in the legend along with the proxy stack and the
    # band patch. The metadata is important to include in the legend to
    # provide context about how the plot was generated.
    meta_patch = construct_meta_handle(code_version, selection_version)

    # Add the legend for the stacked histogram and the error band using
    # the proxy stack and the band patch.
    legend = ax.legend(handles=proxy_stack + [band_patch, meta_patch])

    if detector_label is not None:
        legend.set_title(detector_label)
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(14)
        legend.get_title().set_color("#d67a11")
    texts = legend.get_texts()
    texts[-1].set_fontsize(8)
    texts[-1].set_alpha(0.6)

    # The y-axis should use scientific notation to force consistent
    # formatting across different plots, especially when the range of
    # values can vary significantly.
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
                f"{variable}:0:{detector}:0:total:DATA", TraceType.HIST_CONTENTS
            )
            data_band = data.get_trace(
                f"{variable}:0:{detector}:0:total:DATA", TraceType.HIST_ERROR_BAND
            )
            y = data_band[:, 1] / data_trace[:, 3]
            ylo = data_band[:, 2] / data_trace[:, 3]
            yhi = data_band[:, 3] / data_trace[:, 3]

        elif ratio == "null":
            y = np.ones_like(band[:, 1])
            ylo = np.zeros_like(band[:, 1])
            yhi = np.zeros_like(band[:, 1])

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

    # Save the figure if requested. We use the detector name, variable
    # index, and code version to construct the filename, which provides
    # useful information about the content and provenance of the plot.
    if output is not None:
        figname = f"hist_{detector}_{variable}_{code_version}.png"
        figure.savefig(output / figname)


def uncertainty(
    data: ProfitPlotData,
    *,
    detector: int,
    tags: list,
    xlabel: str,
    code_version: str,
    selection_version: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    watermark: Optional[str] = r"$\bf{SBN}$ Internal",
    **kwargs,
) -> None:
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
    xlim : Optional[tuple[float, float]]
        The limits for the x-axis of the plot.
    ylim : Optional[tuple[float, float]]
        The limits for the y-axis of the plot.
    watermark : Optional[str]
        The watermark label placed above the axis.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()

    for tag in tags:
        name = f"0:{detector}:0:{tag}:SUM"
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
    texts = legend.get_texts()
    texts[-1].set_fontsize(8)
    texts[-1].set_alpha(0.6)

    mark_axis(ax, watermark, hadj=0.035)
