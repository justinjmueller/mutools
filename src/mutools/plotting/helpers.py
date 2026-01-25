import matplotlib.pyplot as plt
from typing import Optional

def mark_axis(
    ax,
    label : str,
    vadj : Optional[float] = 0,
    hadj : Optional[float] = 0,
    alignment : Optional[str] = 'left'
) -> None:
    """
    Mark the axis with a contextual label. The adjustment parameters
    allow for fine-tuning the label position relative to the axis. The
    default position places the label above the top-left corner of the
    axis, which is suitable for most plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to mark.
    label : str
        The label to place on the axis.
    vadj : float, optional
        Vertical adjustment for the label position.
    hadj : float, optional
        Horizontal adjustment for the label position.
    """
    yrange = ax.get_ylim()
    usey = yrange[1] + 0.01*(yrange[1] - yrange[0]) + vadj*(yrange[1] - yrange[0])
    xrange = ax.get_xlim()
    usex = xrange[0] + 0.025*(xrange[1] - xrange[0]) + hadj*(xrange[1] - xrange[0])
    if alignment == 'left':
        ax.text(x=usex, y=usey, s=label, fontsize=14, color='#d67a11')
    elif alignment == 'right':
        usex = xrange[1] - 0.025*(xrange[1] - xrange[0]) + hadj*(xrange[1] - xrange[0])
        ax.text(x=usex, y=usey, s=label, fontsize=14, color='#d67a11', horizontalalignment='right')