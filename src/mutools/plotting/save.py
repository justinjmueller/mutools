"""
Centralised figure saving for mutools.plotting.

A module-level ``saver`` instance holds persistent settings (DPI, format,
``bbox_inches``). All plotting functions in the package route their output
through this object, so a single ``saver.configure(...)`` call changes
behaviour everywhere at once.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union


_VALID_FORMATS = {"eps", "pdf", "png", "ps", "svg"}


class FigureSaver:
    """
    Stateful helper for saving matplotlib figures.

    Settings are changed via :meth:`configure` and persist for the lifetime
    of the process.  Use the :meth:`settings` context manager for temporary
    per-call overrides without mutating the global state.

    Attributes
    ----------
    dpi : int
        Resolution in dots per inch. Default is 150.
    fmt : str
        Output format passed to ``Figure.savefig``. Default is ``"pdf"``.
    bbox_inches : str
        Whitespace trimming mode. Default is ``"tight"``.
    """

    def __init__(self) -> None:
        self.dpi: int = 150
        self.fmt: str = "pdf"
        self.bbox_inches: str = "tight"

    def configure(
        self,
        *,
        dpi: Optional[int] = None,
        fmt: Optional[str] = None,
        bbox_inches: Optional[str] = None,
    ) -> None:
        """
        Update one or more persistent save settings.

        Parameters
        ----------
        dpi : int, optional
            Resolution in dots per inch.
        fmt : str, optional
            Output format. Must be one of ``"eps"``, ``"pdf"``, ``"png"``,
            ``"ps"``, or ``"svg"``.
        bbox_inches : str, optional
            Passed directly to ``Figure.savefig``.  Use ``"tight"`` to trim
            excess whitespace, or ``None`` to use the figure bounding box.
        """
        if dpi is not None:
            self.dpi = int(dpi)
        if fmt is not None:
            if fmt not in _VALID_FORMATS:
                raise ValueError(
                    f"Unsupported format {fmt!r}. Choose from {sorted(_VALID_FORMATS)}."
                )
            self.fmt = fmt
        if bbox_inches is not None:
            self.bbox_inches = bbox_inches

    @contextmanager
    def settings(self, **kwargs):
        """
        Context manager for temporary save settings.

        Accepts the same keyword arguments as :meth:`configure`. The previous
        settings are restored on exit regardless of exceptions.

        Example
        -------
        >>> with saver.settings(dpi=600, fmt="svg"):
        ...     histogram(data, ..., output="figures/")
        """
        saved = {k: getattr(self, k) for k in ("dpi", "fmt", "bbox_inches")}
        self.configure(**kwargs)
        try:
            yield
        finally:
            for k, v in saved.items():
                setattr(self, k, v)

    def save(
        self,
        fig: "matplotlib.figure.Figure",
        directory: Union[str, Path],
        stem: str,
    ) -> Path:
        """
        Save *fig* to ``<directory>/<stem>.<fmt>`` using the current settings.

        The output directory is created if it does not already exist.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        directory : str or Path
            Output directory.
        stem : str
            Filename stem, without extension.

        Returns
        -------
        Path
            The path that was written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches=self.bbox_inches)
        return path


#: Module-level :class:`FigureSaver` instance used by all plotting functions.
saver = FigureSaver()
