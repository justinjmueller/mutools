"""
SPINE training performance visualization.

Provides a semi-internal axis-level helper (_plot_metric) and a
figure-level wrapper (plot_train_performance) for plotting metrics
from SPINE training log files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d

from ..io.spine import load_logs
from .helpers import mark_axis
from .save import saver


def _plot_metric(
    log_dir: Path,
    pattern: str,
    metric: Union[List[str], str],
    label: Union[List[str], str],
    ax: plt.Axes,
    smooth: Optional[int] = None,
    colors: Optional[List[str]] = None,
    bpe: Optional[int] = None,
    val_stride: Optional[int] = None,
    val_checkpoint_stride: Optional[float] = None,
) -> None:
    """
    Plot training metric(s) onto an existing Axes. Handles log loading,
    aggregation, smoothing, and optional validation overlay.

    Parameters
    ----------
    log_dir : Path
        Directory containing log CSV files.
    pattern : str
        Glob pattern to match training log files.
    metric : str or list of str
        Column name(s) to plot (e.g. ``'loss'`` or ``['loss', 'acc']``).
    label : str or list of str
        Legend label(s) corresponding to each metric.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    smooth : int, optional
        Rolling-window size for smoothing the training curve. No
        smoothing is applied when ``None`` or ``<= 1``.
    colors : list of str, optional
        One colour per metric. Defaults to ``'C0'``, ``'C1'``, … when
        ``None``.
    bpe : int, optional
        Batches per epoch. When provided, validation checkpoints are
        loaded and plotted alongside the training curve. When ``None``
        (default), validation plotting is skipped.
    val_stride : int, optional
        Keep every *n*-th validation checkpoint. No effect when ``None``.
    val_checkpoint_stride : float, optional
        Minimum epoch spacing between consecutive plotted validation
        checkpoints. Points are greedily selected so that consecutive
        plotted checkpoints differ by at least this value. No effect
        when ``None``.
    """
    # Normalise to lists first so that the colors default uses the
    # correct length.
    if isinstance(metric, str):
        metric = [metric]
    if isinstance(label, str):
        label = [label]

    if colors is None:
        colors = [f'C{i}' for i in range(len(metric))]

    train_data = load_logs(log_dir, pattern=pattern, method='concat')

    values = train_data[metric].values

    if smooth is not None and smooth > 1:
        kernel = np.ones(smooth) / smooth
        values = np.apply_along_axis(
            lambda m: convolve1d(m, kernel, mode='nearest'),
            axis=0,
            arr=values,
        )

    for i in range(values.shape[1]):
        ax.plot(
            train_data['epoch'],
            values[:, i],
            label=label[i] + ' (Train)',
            alpha=0.5,
            color=colors[i],
        )

    if bpe is None:
        return

    validation_data = load_logs(
        log_dir,
        pattern='inference*.csv',
        method='mean',
        bpe=bpe,
    )

    if validation_data.empty:
        return

    mean_cols = [v + '_mean' for v in metric]
    std_cols = [v + '_std' for v in metric]
    validation_values = validation_data[mean_cols].values
    validation_stds = validation_data[std_cols].values

    if val_stride is not None and val_stride > 1:
        validation_data = validation_data.iloc[::val_stride]
        validation_values = validation_values[::val_stride]
        validation_stds = validation_stds[::val_stride]

    if val_checkpoint_stride is not None:
        checkpoints = validation_data['checkpoint'].values
        mask = np.zeros(len(checkpoints), dtype=bool)
        mask[0] = True
        last = checkpoints[0]
        for i in range(1, len(checkpoints)):
            if checkpoints[i] - last >= val_checkpoint_stride:
                mask[i] = True
                last = checkpoints[i]
        validation_data = validation_data[mask]
        validation_values = validation_values[mask]
        validation_stds = validation_stds[mask]

    for i in range(validation_values.shape[1]):
        ax.errorbar(
            validation_data['checkpoint'],
            validation_values[:, i],
            yerr=validation_stds[:, i],
            fmt='o',
            label=label[i] + ' (Val.)',
            color=colors[i],
        )


def plot_train_performance(
    path: Path,
    metrics: Dict[str, str],
    *,
    pattern: Optional[str] = 'train_*.csv',
    smooth: Optional[int] = 10,
    colors: Optional[List[str]] = None,
    ullabel: Optional[str] = None,
    urlabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ylabel: Optional[str] = None,
    bpe: Optional[int] = None,
    val_stride: Optional[int] = None,
    val_checkpoint_stride: Optional[float] = None,
    output: Optional[Path] = None,
    name: str = 'train_performance',
) -> None:
    """
    Plot training performance metrics from SPINE log files.

    Parameters
    ----------
    path : Path
        Directory containing the log files.
    metrics : dict of str to str
        Metrics to plot. Keys are column names in the log CSV; values
        are the corresponding legend labels.
    pattern : str, optional
        Glob pattern for training log files. Default is
        ``'train_*.csv'``.
    smooth : int, optional
        Smoothing window size passed to :func:`_plot_metric`. Default
        is ``10``.
    colors : list of str, optional
        One colour per metric. Defaults to ``'C0'``, ``'C1'``, … when
        ``None``.
    ullabel : str, optional
        Label placed above the upper-left corner of the plot.
    urlabel : str, optional
        Label placed above the upper-right corner of the plot.
    xlim : tuple of float, optional
        X-axis limits as ``(min, max)``.
    ylim : tuple of float, optional
        Y-axis limits as ``(min, max)``.
    ylabel : str, optional
        Y-axis label. No label is set when ``None``.
    bpe : int, optional
        Batches per epoch. When provided, validation checkpoints are
        overlaid on the plot.
    val_stride : int, optional
        Keep every *n*-th validation checkpoint.
    val_checkpoint_stride : float, optional
        Minimum epoch spacing between consecutive plotted validation
        checkpoints.
    output : Path, optional
        Output directory for saving the figure. The figure is not
        saved when ``None``.
    name : str, optional
        Filename stem (without extension) used when saving. Default is
        ``'train_performance'``.
    """
    figure, ax = plt.subplots(figsize=(8, 6))

    _plot_metric(
        path,
        pattern,
        list(metrics.keys()),
        list(metrics.values()),
        ax=ax,
        smooth=smooth,
        colors=colors,
        bpe=bpe,
        val_stride=val_stride,
        val_checkpoint_stride=val_checkpoint_stride,
    )

    ax.set_xlabel('Epoch')

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if ullabel is not None:
        mark_axis(ax, ullabel, alignment='left')

    if urlabel is not None:
        mark_axis(ax, urlabel, alignment='right')

    ax.legend()

    if output is not None:
        saver.save(figure, output, name)
