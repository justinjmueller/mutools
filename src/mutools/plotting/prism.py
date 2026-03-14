"""
PRISM schematic plots.

Produces a two-panel schematic (SBND | ICARUS) showing the transverse
cross-section of each detector with concentric annular rings that
represent the PRISM off-axis angle (OAA) bins.

The OAA is defined using the *projection* method: the angle is computed
from the beam axis to the interaction vertex using the transverse
displacement projected onto the front face of the detector, i.e.

    theta_OAA = arctan( sqrt((x - x0)^2 + (y - y0)^2) / z_det )

where (x0, y0) is the beam-axis offset at the detector and z_det is
the distance from the neutrino production target to the front face of
the detector.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.cm import ScalarMappable


# ── Detector geometry ─────────────────────────────────────────────────────────
# All lengths in centimetres.  TPC positions are the centres of each TPC
# volume.  Half-extents are derived from the YAML geometry configs as
# half = full_dimension / 2.

# SBND — two TPC modules side-by-side in x
SBND_TPC_DX = 201.4 / 2        # cm, half-width  in x per TPC
SBND_TPC_DY = 400.015 / 2      # cm, half-height in y per TPC
SBND_TPC_CX = [-100.9, 100.9]  # cm, x-centres of the two TPCs
SBND_TPC_CY = [0.0, 0.0]       # cm, y-centres of the two TPCs
SBND_BEAM_X = -74.0            # cm, BNB beam-axis x-offset at the SBND front face
SBND_BEAM_Y = 0.0              # cm, BNB beam-axis y-offset at the SBND front face
SBND_Z      = 11000.0          # cm, distance from neutrino target to SBND front face

# ICARUS — four TPC modules in two cryostats (two TPCs per cryostat)
ICAR_TPC_DX = 148.2 / 2                              # cm, half-width  in x per TPC
ICAR_TPC_DY = 316.82 / 2                             # cm, half-height in y per TPC
ICAR_TPC_CX = [-284.39, -136.04, 136.04, 284.39]     # cm, x-centres
ICAR_TPC_CY = [-23.45, -23.45, -23.45, -23.45]       # cm, y-centres
ICAR_BEAM_X = 0.0                                    # cm, BNB beam-axis x-offset at ICARUS front face
ICAR_BEAM_Y = 0.0                                    # cm, BNB beam-axis y-offset at ICARUS front face
ICAR_Z      = 60000.0 - 895.0                        # cm, distance from target to ICARUS front face

# ── Visual style ──────────────────────────────────────────────────────────────
_CMAP       = plt.cm.plasma
_TPC_FACE   = "#d6e8f5"  # light blue fill for active TPC volumes
_TPC_EDGE   = "#2a5f8a"  # dark blue outline
_CATHODE_COL = "crimson" # cathode ring colour


def oaa_to_radius(theta_deg: float, z: float) -> float:
    """
    Convert an off-axis angle to a transverse radius at the detector
    front face using the projection OAA formula.

    Parameters
    ----------
    theta_deg : float
        Off-axis angle in degrees.
    z : float
        Distance from the neutrino production target to the detector
        front face, in centimetres.

    Returns
    -------
    float
        Transverse radius in centimetres, i.e. ``z * tan(theta)``.
    """
    return z * np.tan(np.deg2rad(theta_deg))


def _detector_extent(
    tpc_cx: list,
    tpc_cy: list,
    tpc_dx: float,
    tpc_dy: float,
    pad: float = 50.0,
) -> tuple:
    """
    Compute the axis-aligned bounding box that contains all TPC volumes,
    with an optional uniform padding.

    Returns x_lo, x_hi, y_lo, y_hi, x_centre, y_centre.
    """
    x_lo = min(cx - tpc_dx for cx in tpc_cx) - pad
    x_hi = max(cx + tpc_dx for cx in tpc_cx) + pad
    y_lo = min(cy - tpc_dy for cy in tpc_cy) - pad
    y_hi = max(cy + tpc_dy for cy in tpc_cy) + pad
    x_centre = 0.5 * (
        min(cx - tpc_dx for cx in tpc_cx) + max(cx + tpc_dx for cx in tpc_cx)
    )
    y_centre = 0.5 * (
        min(cy - tpc_dy for cy in tpc_cy) + max(cy + tpc_dy for cy in tpc_cy)
    )
    return x_lo, x_hi, y_lo, y_hi, x_centre, y_centre


def _make_lim(centre: float, half: float) -> tuple:
    return (centre - half, centre + half)


def _draw_detector(
    ax,
    title: str,
    tpc_cx: list,
    tpc_cy: list,
    tpc_dx: float,
    tpc_dy: float,
    beam_x: float,
    beam_y: float,
    radii: list,
    bin_edges: np.ndarray,
    colors: list,
    cathode_r: float,
    cathode_oaa: float,
    xlim: tuple,
    ylim: tuple,
    show_cathode: bool = True,
) -> None:
    """
    Draw a single detector panel showing TPC active volumes, PRISM OAA
    bin rings, cathode ring, and beam-axis marker.

    Drawing order (back to front):

    1. Filled ``matplotlib.patches.Annulus`` for each OAA bin, drawn
       largest-to-smallest so inner rings sit on top.
    2. TPC active-volume rectangles.
    3. Dashed OAA bin boundary circles drawn over TPCs so they remain
       visible above the TPC fill colour.
    4. Dotted cathode ring.
    5. Beam-axis cross-hair marker.
    6. Text labels at the midpoint radius of each bin.
    """
    ax.set_facecolor("white")
    theta_arr = np.linspace(0, 2 * np.pi, 600)

    # 1. Filled PRISM annuli — drawn back-to-front
    for i in reversed(range(len(radii) - 1)):
        r_outer = radii[i + 1]
        r_inner = radii[i]
        ring = patches.Annulus(
            (beam_x, beam_y),
            r=r_outer,
            width=r_outer - r_inner,
            color=to_rgba(colors[i], 0.22),
            zorder=2,
        )
        ax.add_patch(ring)

    # 2. TPC active-volume rectangles
    for cx, cy in zip(tpc_cx, tpc_cy):
        rect = patches.FancyBboxPatch(
            (cx - tpc_dx, cy - tpc_dy),
            2 * tpc_dx,
            2 * tpc_dy,
            boxstyle="square,pad=0",
            linewidth=1.6,
            edgecolor=_TPC_EDGE,
            facecolor=_TPC_FACE,
            zorder=3,
        )
        ax.add_patch(rect)

    # 3. OAA bin boundary circles drawn over TPCs
    for i, r in enumerate(radii):
        if i == 0:
            continue  # zero-radius innermost edge — nothing to draw
        ax.plot(
            beam_x + r * np.cos(theta_arr),
            beam_y + r * np.sin(theta_arr),
            color=colors[i - 1],
            lw=1.2,
            ls="--",
            zorder=5,
            alpha=0.85,
        )

    # 4. Cathode ring (optional)
    if show_cathode:
        ax.plot(
            beam_x + cathode_r * np.cos(theta_arr),
            beam_y + cathode_r * np.sin(theta_arr),
            color=_CATHODE_COL,
            lw=1.5,
            ls=":",
            zorder=6,
            alpha=0.9,
            label=f"Cathode ({cathode_oaa:.2f}°)",
        )

    # 5. Beam-axis marker
    ax.plot(beam_x, beam_y, "k+", ms=10, mew=2.0, zorder=8, label="Beam axis")

    # 6. Bin labels placed at the midpoint radius along a fixed angle
    label_angle = np.deg2rad(52)
    for i in range(len(radii) - 1):
        r_mid = 0.5 * (radii[i] + radii[i + 1])
        lx = beam_x + r_mid * np.cos(label_angle)
        ly = beam_y + r_mid * np.sin(label_angle)
        if (xlim[0] < lx < xlim[1]) and (ylim[0] < ly < ylim[1]):
            ax.text(
                lx, ly,
                f"Bin {i + 1}\n{bin_edges[i]:.2f}°–{bin_edges[i + 1]:.2f}°",
                ha="center", va="center",
                fontsize=6.5, color="k", zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1),
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]", fontsize=11)
    ax.set_ylabel("y [cm]", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    ax.tick_params(labelsize=9)
    ax.grid(True, lw=0.4, alpha=0.4)


def prism_schematic(
    n_bins: int = 5,
    oaa_min: float = 0.0,
    oaa_max: float = 1.7,
    bin_edges: Optional[list] = None,
    show_cathode: bool = True,
    output: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    Produce a two-panel PRISM schematic showing the transverse
    cross-section of SBND and ICARUS with concentric OAA bin rings.

    Both panels share the same physical (cm) scale, driven by the full
    active volume of ICARUS (the larger detector) plus a 50 cm margin.
    OAA ring arcs that extend beyond the window are clipped naturally by
    the axis limits.

    Parameters
    ----------
    n_bins : int
        Number of OAA bins. Ignored when ``bin_edges`` is provided.
        Default is 5.
    oaa_min : float
        Lower edge of the first OAA bin in degrees. Ignored when
        ``bin_edges`` is provided. Default is 0.0.
    oaa_max : float
        Upper edge of the last OAA bin in degrees. Ignored when
        ``bin_edges`` is provided. Default is 1.7.
    bin_edges : list of float, optional
        Explicit OAA bin edges in degrees. When provided, overrides
        ``n_bins``, ``oaa_min``, and ``oaa_max``.
    show_cathode : bool
        Whether to draw the cathode OAA ring on each panel.
        Default is ``True``.
    output : Optional[Path]
        Directory in which to save the figure as
        ``prism_schematic.pdf``.  If ``None``, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    if bin_edges is not None:
        bin_edges = np.asarray(bin_edges, dtype=float)
        n_bins = len(bin_edges) - 1
        oaa_min, oaa_max = float(bin_edges[0]), float(bin_edges[-1])
    else:
        bin_edges = np.linspace(oaa_min, oaa_max, n_bins + 1)
    bin_colors = [_CMAP(0.15 + 0.7 * i / max(n_bins - 1, 1)) for i in range(n_bins)]

    sbnd_radii = [oaa_to_radius(e, SBND_Z) for e in bin_edges]
    icar_radii = [oaa_to_radius(e, ICAR_Z) for e in bin_edges]

    sbnd_cathode_oaa = np.degrees(np.arctan(
        np.sqrt((0.0 - SBND_BEAM_X) ** 2 + (0.0 - SBND_BEAM_Y) ** 2) / SBND_Z
    ))
    icar_cathode_oaa = np.degrees(np.arctan(
        np.sqrt((210.12 - ICAR_BEAM_X) ** 2 + (0.0 - ICAR_BEAM_Y) ** 2) / ICAR_Z
    ))
    sbnd_cathode_r = oaa_to_radius(sbnd_cathode_oaa, SBND_Z)
    icar_cathode_r = oaa_to_radius(icar_cathode_oaa, ICAR_Z)

    # Shared axis window driven by the ICARUS (larger) detector extent
    pad = 50.0  # cm
    _, _, _, _, sbnd_xc, sbnd_yc = _detector_extent(
        SBND_TPC_CX, SBND_TPC_CY, SBND_TPC_DX, SBND_TPC_DY, pad=0
    )
    icar_x_lo, icar_x_hi, icar_y_lo, icar_y_hi, icar_xc, icar_yc = _detector_extent(
        ICAR_TPC_CX, ICAR_TPC_CY, ICAR_TPC_DX, ICAR_TPC_DY, pad=0
    )
    half_x = 0.5 * (icar_x_hi - icar_x_lo) + pad
    half_y = 0.5 * (icar_y_hi - icar_y_lo) + pad
    half_span = max(half_x, half_y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.patch.set_facecolor("white")

    _draw_detector(
        axes[0], "SBND",
        SBND_TPC_CX, SBND_TPC_CY, SBND_TPC_DX, SBND_TPC_DY,
        SBND_BEAM_X, SBND_BEAM_Y,
        sbnd_radii, bin_edges, bin_colors,
        sbnd_cathode_r, sbnd_cathode_oaa,
        xlim=_make_lim(SBND_BEAM_X, half_span),
        ylim=_make_lim(sbnd_yc, half_span),
        show_cathode=show_cathode,
    )
    _draw_detector(
        axes[1], "ICARUS",
        ICAR_TPC_CX, ICAR_TPC_CY, ICAR_TPC_DX, ICAR_TPC_DY,
        ICAR_BEAM_X, ICAR_BEAM_Y,
        icar_radii, bin_edges, bin_colors,
        icar_cathode_r, icar_cathode_oaa,
        xlim=_make_lim(ICAR_BEAM_X, half_span),
        ylim=_make_lim(icar_yc, half_span),
        show_cathode=show_cathode,
    )

    sm = ScalarMappable(cmap=_CMAP, norm=plt.Normalize(oaa_min, oaa_max))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes,
        orientation="horizontal",
        fraction=0.03, pad=0.12, aspect=40,
    )
    cbar.set_label(r"$\theta_\mathrm{OAA}$ [degrees]", fontsize=11)
    cbar.set_ticks(bin_edges)
    cbar.set_ticklabels([f"{v:.2f}°" for v in bin_edges], fontsize=8)

    fig.suptitle(
        "PRISM Bins: Off-Axis Angle Rings on Detector Front Faces",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "prism_schematic.pdf", bbox_inches="tight", dpi=200)

    return fig
