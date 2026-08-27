"""
Factory and registry for quality-cut predicates.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable

import toml

from mutools.exposure._node import CutNode
from mutools.exposure.cut import QualityCut


class _QualityCutFactory:
    """
    Produced by @CutRegistry.quality_cut. Calling it with keyword
    parameters returns a fixed QualityCut instance.
    """

    def __init__(self, func: Callable, columns: frozenset[str]) -> None:
        self._func = func
        self._columns = columns
        functools.update_wrapper(self, func)

    def __call__(self, **params: Any) -> QualityCut:
        """
        Instantiate a fixed QualityCut with the given parameters.

        Parameters
        -----------
        **params: Any
           Keyword-only parameters bound into the predicate at
           construction time.

        Returns
        --------
        cut: QualityCut
           A fixed cut with params bound into its predicate.
        """
        node = CutNode.make_leaf(self._func.__name__, params, self._columns)
        return QualityCut(functools.partial(self._func, **params), node)

    def __repr__(self) -> str:
        return f"QualityCutFactory({self._func.__name__!r})"


class CutRegistry:
    """
    Registry of quality-cut factories for a specific beam context
    (e.g., BNB or NuMI).

    Examples
    --------
    >>> bnb = CutRegistry("bnb")
    >>>
    >>> @bnb.quality_cut(columns=["E1DCBOB"])
    ... def good_pot(df: pd.DataFrame, *, threshold: float) -> pd.Series:
    ...     return df["E1DCBOB"] > threshold
    >>>
    >>> cut = good_pot(threshold=0.0)
    >>> mask = cut(df)
    """

    def __init__(self, name: str = "") -> None:
        """
        Construct a CutRegistry.

        Parameters
        -----------
        name: str
           Identifier for this registry, used in error messages.
        """
        self._name = name
        self._factories: dict[str, _QualityCutFactory] = {}

    def quality_cut(
        self,
        func: Callable | None = None,
        *,
        columns: list[str] | None = None,
    ) -> _QualityCutFactory | Callable[[Callable], _QualityCutFactory]:
        """
        Register a function as a quality-cut factory.

        Can be used with or without arguments::

            @registry.quality_cut
            def my_cut(df, *, threshold): ...

            @registry.quality_cut(columns=["pot"])
            def my_cut(df, *, threshold): ...

        The decorated function must accept a DataFrame as its first
        argument and return a boolean Series. Additional parameters must
        be keyword-only and are bound at construction time.

        Parameters
        -----------
        func: Callable | None
           The function to decorate when used without parentheses.
        columns: list[str] | None
           DataFrame columns required by the predicate.

        Returns
        --------
        factory: _QualityCutFactory
           A factory that produces fixed QualityCut instances.
        """
        def decorator(f: Callable) -> _QualityCutFactory:
            factory = _QualityCutFactory(f, frozenset(columns or []))
            self._factories[f.__name__] = factory
            return factory

        if func is not None:
            return decorator(func)
        return decorator

    def __getitem__(self, name: str) -> _QualityCutFactory:
        """
        Look up a registered factory by name.

        Parameters
        -----------
        name: str
           The name of the registered cut function.

        Returns
        --------
        factory: _QualityCutFactory
           The factory registered under that name.

        Raises
        -------
        KeyError
           If no cut with that name exists in this registry.
        """
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(
                f"No cut named {name!r} in registry {self._name!r}. "
                f"Registered: {list(self._factories)}"
            ) from None

    def load_cuts(self, path: str | Path) -> dict[str, QualityCut]:
        """
        Instantiate named QualityCut instances from a TOML file.

        Entries under [cuts] are resolved in order, so composite cuts
        may reference earlier-defined cuts by name string.

        Leaf entry::

            [cuts.good_pot]
            function  = "good_pot"
            threshold = 0.0

        Composite entry referencing named cuts::

            [cuts.standard_bnb]
            operator = "and"
            operands = ["good_pot", "no_inhibit"]

        Composite entry with inline definitions::

            [cuts.strict_bnb]
            operator = "and"
            operands = [
                {function = "good_pot", threshold = 0.0},
                {function = "no_inhibit"},
            ]

        Parameters
        -----------
        path: str | Path
           Path to the TOML configuration file.

        Returns
        --------
        cuts: dict[str, QualityCut]
           Mapping of cut name to instantiated QualityCut.
        """
        with open(path) as fh:
            config = toml.load(fh)

        resolved: dict[str, QualityCut] = {}
        for cut_name, cut_cfg in config.get("cuts", {}).items():
            resolved[cut_name] = self._parse_entry(cut_cfg, resolved)
        return resolved

    def _parse_entry(
        self,
        cfg: dict[str, Any],
        resolved: dict[str, QualityCut],
    ) -> QualityCut:
        """
        Recursively parse a single TOML cut entry into a QualityCut.

        Parameters
        -----------
        cfg: dict[str, Any]
           Parsed TOML dictionary for one cut entry.
        resolved: dict[str, QualityCut]
           Already-instantiated cuts available for reference by name.

        Returns
        --------
        cut: QualityCut
           The instantiated cut described by cfg.
        """
        if "function" in cfg:
            func_name = cfg["function"]
            params = {k: v for k, v in cfg.items() if k != "function"}
            return self[func_name](**params)

        op = str(cfg.get("operator", "")).lower()
        if op not in ("and", "or", "not"):
            raise ValueError(
                f"Unknown operator {op!r}; expected 'and', 'or', or 'not'."
            )

        operands: list[QualityCut] = []
        for o in cfg.get("operands", []):
            if isinstance(o, str):
                if o not in resolved:
                    raise KeyError(
                        f"Cut {o!r} is referenced before it is defined in the TOML."
                    )
                operands.append(resolved[o])
            elif isinstance(o, dict):
                operands.append(self._parse_entry(o, resolved))
            else:
                raise TypeError(f"Unexpected operand type {type(o).__name__!r}.")

        if op == "not":
            if len(operands) != 1:
                raise ValueError("'not' requires exactly one operand.")
            return ~operands[0]

        if len(operands) < 2:
            raise ValueError(f"'{op}' requires at least two operands.")

        result = operands[0]
        for o in operands[1:]:
            result = (result & o) if op == "and" else (result | o)
        return result

    def __repr__(self) -> str:
        return f"CutRegistry({self._name!r}, cuts={list(self._factories)})"
