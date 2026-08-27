"""
Pairwise and individual cut-failure statistics, with built-in visualisations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Private plot helpers
# ---------------------------------------------------------------------------

def _get_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """
    Return (fig, ax), creating a new figure when ax is None.

    New figures are created with constrained_layout=True so that
    colorbars are handled without requiring a separate tight_layout() call.

    Parameters
    -----------
    ax: Axes | None
       Existing axes to reuse, or None to create a new figure.

    Returns
    --------
    fig: Figure
       The figure that owns the axes.
    ax: Axes
       The axes to draw into.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, new_ax = plt.subplots(constrained_layout=True)
        return fig, new_ax
    return ax.get_figure(), ax


def _annotate_imshow(
    ax: Axes,
    data: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    fmt: str = ".2f",
) -> None:
    """
    Overlay text values on each cell of an imshow plot.

    Text colour is chosen automatically from the cell's luminance so it
    remains legible against both light and dark backgrounds.

    Parameters
    -----------
    ax: Axes
       Axes containing the imshow image.
    data: np.ndarray
       2-D array of values displayed in the image.
    cmap_name: str
       Name of the matplotlib colormap used for the image.
    vmin: float
       Lower bound of the colormap normalisation.
    vmax: float
       Upper bound of the colormap normalisation.
    fmt: str
       Format string applied to each cell value.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            r, g, b, _ = cmap(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j, i, format(val, fmt),
                ha="center", va="center", fontsize=8,
                color="white" if lum < 0.5 else "black",
            )


def _configure_matrix_axes(ax: Axes, xlabels: list[str], ylabels: list[str]) -> None:
    """
    Set tick positions and rotated labels for a square matrix plot.

    Parameters
    -----------
    ax: Axes
       Axes to configure.
    xlabels: list[str]
       Labels for the x-axis ticks.
    ylabels: list[str]
       Labels for the y-axis ticks.
    """
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)


def _save_figure(fig: Figure, path: str | Path | None) -> None:
    """
    Save a figure to disk if a path is provided.

    Parameters
    -----------
    fig: Figure
       The figure to save.
    path: str | Path | None
       Destination path; no-op when None.
    """
    if path is not None:
        fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class CutCorrelations:
    """
    Pairwise and individual statistics for a QualityCut's leaf predicates.

    Produced by QualityCut.correlations(). Leaf cuts are taken in
    left-to-right expression-tree order with duplicates removed.

    Attributes
    -----------
    phi: pd.DataFrame
       Symmetric NxN phi (Pearson) coefficients between the failure
       masks of each leaf-cut pair.
    conditional: pd.DataFrame
       Asymmetric NxN matrix; entry (i, j) = P(fail j | fail i).
    unique_rejection: pd.Series
       Per-cut fraction of total spills rejected exclusively by that
       cut and no other.
    total_rejection: pd.Series
       Per-cut fraction of total spills rejected, regardless of other cuts.
    waterfall: pd.DataFrame
       Cumulative pass counts and rates as cuts are applied in order.
       Indexed by cut name; columns: passed, total, pass_rate,
       incremental_loss.
    """

    phi: pd.DataFrame
    conditional: pd.DataFrame
    unique_rejection: pd.Series
    total_rejection: pd.Series
    waterfall: pd.DataFrame

    def _resolve_names(self, names: list[str] | None) -> list[str]:
        """
        Return the list of cut names to display.

        When *names* is None, all cuts in the result are returned.
        Otherwise each entry is validated against the phi matrix index.

        Raises
        -------
        KeyError
           If any name is not present in this CutCorrelations result.
        """
        all_names = list(self.phi.index)
        if names is None:
            return all_names
        missing = [n for n in names if n not in self.phi.index]
        if missing:
            raise KeyError(
                f"Name(s) not found in CutCorrelations: {missing}. "
                f"Available: {all_names}"
            )
        return names

    def plot_phi(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        names: list[str] | None = None,
        path: str | Path | None = None,
    ) -> Figure:
        """
        Plot the phi coefficient matrix as a heatmap.

        Parameters
        -----------
        ax: Axes | None
           Existing axes to draw into; a new figure is created if None.
        title: str | None
           Axes title; defaults to 'Phi Coefficient (Failure Correlation)'.
        names: list[str] | None
           Subset of cut names to display, in the given order. When
           None (default), all cuts are shown.
        path: str | Path | None
           If provided, save the figure to this path.

        Returns
        --------
        fig: Figure
           The matplotlib Figure containing the plot.
        """
        fig, ax = _get_axes(ax)
        sel = self._resolve_names(names)
        data = self.phi.loc[sel, sel].to_numpy(dtype=float)

        im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        _annotate_imshow(ax, data, "RdBu_r", -1, 1)
        _configure_matrix_axes(ax, sel, sel)
        ax.set_title(title or "Phi Coefficient (Failure Correlation)")
        fig.colorbar(im, ax=ax, label="phi")
        _save_figure(fig, path)
        return fig

    def plot_conditional(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        names: list[str] | None = None,
        path: str | Path | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> Figure:
        """
        Plot the conditional failure probability matrix as a heatmap.

        Entry (i, j) shows P(fail j | fail i).

        Parameters
        -----------
        ax: Axes | None
           Existing axes to draw into; a new figure is created if None.
        title: str | None
           Axes title; defaults to 'Conditional Failure P(col | row)'.
        names: list[str] | None
           Subset of cut names to display, in the given order. When
           None (default), all cuts are shown.
        path: str | Path | None
           If provided, save the figure to this path.
        xlabel: str | None
           Label for the x-axis; defaults to 'Fails →'.
        ylabel: str | None
           Label for the y-axis; defaults to 'Given fails →'.

        Returns
        --------
        fig: Figure
           The matplotlib Figure containing the plot.
        """
        fig, ax = _get_axes(ax)
        sel = self._resolve_names(names)
        data = self.conditional.loc[sel, sel].to_numpy(dtype=float)

        im = ax.imshow(data, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        _annotate_imshow(ax, data, "YlOrRd", 0, 1)
        _configure_matrix_axes(ax, sel, sel)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        fig.colorbar(im, ax=ax, label="P(fail col | fail row)")
        _save_figure(fig, path)
        return fig

    def plot_unique_rejection(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        names: list[str] | None = None,
        path: str | Path | None = None,
    ) -> Figure:
        """
        Plot total and unique rejection fractions as a grouped bar chart.

        Parameters
        -----------
        ax: Axes | None
           Existing axes to draw into; a new figure is created if None.
        title: str | None
           Axes title; defaults to 'Total vs. Unique Rejection per Cut'.
        names: list[str] | None
           Subset of cut names to display, in the given order. When
           None (default), all cuts are shown.
        path: str | Path | None
           If provided, save the figure to this path.

        Returns
        --------
        fig: Figure
           The matplotlib Figure containing the plot.
        """
        import numpy as np

        fig, ax = _get_axes(ax)
        sel = self._resolve_names(names)
        total = self.total_rejection.loc[sel]
        unique = self.unique_rejection.loc[sel]
        n = len(sel)
        y = np.arange(n)
        h = 0.35

        ax.barh(y + h / 2, total.values * 100, h, label="Total")
        ax.barh(y - h / 2, unique.values * 100, h, label="Unique")
        ax.set_yticks(y)
        ax.set_yticklabels(sel, fontsize=8)
        ax.set_xlabel("Rejection (%)")
        ax.set_title(title or "Total vs. Unique Rejection per Cut")
        ax.legend()
        _save_figure(fig, path)
        return fig

    def plot_waterfall(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        names: list[str] | None = None,
        path: str | Path | None = None,
    ) -> Figure:
        """
        Plot the cumulative pass rate as cuts are applied sequentially.

        Parameters
        -----------
        ax: Axes | None
           Existing axes to draw into; a new figure is created if None.
        title: str | None
           Axes title; defaults to 'Waterfall: Cumulative Pass Rate'.
        names: list[str] | None
           Subset of cut names to display, in the given order. When
           None (default), all cuts are shown.
        path: str | Path | None
           If provided, save the figure to this path.

        Returns
        --------
        fig: Figure
           The matplotlib Figure containing the plot.
        """
        import numpy as np

        fig, ax = _get_axes(ax)
        sel = self._resolve_names(names)
        wf = self.waterfall.loc[sel]
        pass_rates = wf["pass_rate"].values * 100
        losses = wf["incremental_loss"].values * 100
        y = np.arange(len(sel))

        bars = ax.barh(y, pass_rates)
        for bar, rate, loss in zip(bars, pass_rates, losses):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.2f}%  (↓{loss:.3f}%)",
                va="center", fontsize=8,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(sel, fontsize=8)
        ax.set_xlabel("Cumulative Pass Rate (%)")
        ax.set_xlim(0, 108)
        ax.set_title(title or "Waterfall: Cumulative Pass Rate")
        _save_figure(fig, path)
        return fig

    def plot(
        self,
        *,
        title: str | None = None,
        names: list[str] | None = None,
        path: str | Path | None = None,
    ) -> Figure:
        """
        Plot all four analyses in a single 2×2 figure.

        Parameters
        -----------
        title: str | None
           Figure-level suptitle placed above all panels.
        names: list[str] | None
           Subset of cut names to display across all panels, in the
           given order. When None (default), all cuts are shown.
        path: str | Path | None
           If provided, save the combined figure to this path.

        Returns
        --------
        fig: Figure
           The matplotlib Figure containing all four plots.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
        self.plot_phi(ax=axes[0, 0], names=names)
        self.plot_conditional(ax=axes[0, 1], names=names)
        self.plot_unique_rejection(ax=axes[1, 0], names=names)
        self.plot_waterfall(ax=axes[1, 1], names=names)
        if title:
            fig.suptitle(title)
        _save_figure(fig, path)
        return fig
