"""
Utilities for applying matplotlib style sheets bundled with mutools.
"""

from pathlib import Path

import matplotlib.pyplot as plt

_STYLES_DIR = Path(__file__).parent / "styles"


def use_style(name: str) -> None:
    """
    Apply a bundled mutools matplotlib style sheet.

    Parameters
    ----------
    name : str
        Name of the style to apply, without the .mplstyle extension.
        Use list_styles() to see all available options.
    """
    path = _STYLES_DIR / f"{name}.mplstyle"
    if not path.exists():
        available = list_styles()
        raise ValueError(f"Style {name!r} not found. Available styles: {available}")
    plt.style.use(path)


def list_styles() -> list:
    """
    List all available mutools style names.

    Returns
    -------
    list[str]
        Names of available styles, without the .mplstyle extension.
    """
    return sorted(p.stem for p in _STYLES_DIR.glob("*.mplstyle"))
