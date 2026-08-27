"""
QualityCut: a callable, composable quality cut backed by an expression tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from mutools.exposure._node import CutNode
from mutools.exposure.correlations import CutCorrelations

# ---------------------------------------------------------------------------
# Summary types
# ---------------------------------------------------------------------------

@dataclass
class NodeSummary:
    """One node in the hierarchical evaluation tree produced by summary().

    Attributes
    -----------
    name: str
       Display name of this node, derived from the expression tree.
    passed: int | float
       Number of rows passing this node's cut, or (when summary() is
       called with pot_column) the summed POT of passing rows.
    total: int | float
       Total number of rows evaluated, or (when summary() is called
       with pot_column) the summed POT of all rows.
    op: str
       Operator for this node ('leaf', 'and', 'or', 'not').
    labeled: bool
       True when the node was explicitly named via QualityCut.named().
    children: list[NodeSummary]
       Summaries of child nodes; empty for leaf cuts.
    """

    name: str
    passed: int | float
    total: int | float
    op: str = ""
    labeled: bool = False
    children: list[NodeSummary] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """
        Fraction of rows passing this node's cut.

        Returns
        --------
        rate: float
           passed / total, or nan if total is zero.
        """
        return self.passed / self.total if self.total else float("nan")


@dataclass
class CutSummary:
    """Result of evaluating a QualityCut against a DataFrame.

    Attributes
    -----------
    total: int | float
       Total number of rows in the evaluated DataFrame, or (when
       summary() is called with pot_column) the summed POT of all rows.
    passed: int | float
       Number of rows passing the composed cut, or (when summary() is
       called with pot_column) the summed POT of passing rows.
    tree: NodeSummary
       Root of the hierarchical evaluation tree.
    """

    total: int | float
    passed: int | float
    tree: NodeSummary

    @property
    def pass_rate(self) -> float:
        """
        Fraction of rows passing the cut.

        Returns
        --------
        rate: float
           passed / total, or nan if total is zero.
        """
        return self.passed / self.total if self.total else float("nan")

    @property
    def per_predicate(self) -> dict[str, tuple[int | float, int | float]]:
        """
        Flat mapping of leaf predicate name to (n_passed, n_total).

        Returns
        --------
        breakdown: dict[str, tuple[int | float, int | float]]
           Collected from all leaves of the evaluation tree.
        """
        result: dict[str, tuple[int | float, int | float]] = {}

        def collect(node: NodeSummary) -> None:
            if not node.children:
                result[node.name] = (node.passed, node.total)
            for child in node.children:
                collect(child)

        collect(self.tree)
        return result

    def __repr__(self) -> str:
        def fmt(value: int | float) -> str:
            return str(value) if isinstance(value, int) else f"{value:.6g}"

        lines = [
            f"{self.tree.name}: {fmt(self.passed)}/{fmt(self.total)} ({self.pass_rate:.1%})"
        ]

        def flat_children(node: NodeSummary) -> list[NodeSummary]:
            """Collect children, absorbing unlabeled same-op descendants."""
            result = []
            for child in node.children:
                if child.op == node.op and node.op in ("and", "or") and not child.labeled:
                    result.extend(flat_children(child))
                else:
                    result.append(child)
            return result

        def render(node: NodeSummary, depth: int) -> None:
            rate = node.passed / node.total if node.total else float("nan")
            indent = "  " * depth
            children = flat_children(node)
            if not node.labeled and children != node.children:
                sep = f" {node.op} "
                display = f"({sep.join(c.name for c in children)})"
            else:
                display = node.name
            lines.append(f"{indent}{display}: {fmt(node.passed)}/{fmt(node.total)} ({rate:.1%})")
            for child in children:
                render(child, depth + 1)

        for child in flat_children(self.tree):
            render(child, depth=1)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QualityCut
# ---------------------------------------------------------------------------

class QualityCut:
    """
    A fixed, callable quality cut backed by an expression tree.

    Parameters are consumed at construction time via the
    @CutRegistry.quality_cut decorator. Compose instances with &, |,
    and ~ to produce new QualityCut instances with merged expression trees.
    """

    def __init__(
        self,
        predicate: Callable[[pd.DataFrame], pd.Series],
        node: CutNode,
        _left: QualityCut | None = None,
        _right: QualityCut | None = None,
    ) -> None:
        """
        Construct a QualityCut.

        Parameters
        -----------
        predicate: Callable[[pd.DataFrame], pd.Series]
           Vectorized function returning a boolean Series.
        node: CutNode
           Expression tree node representing this cut's provenance.
        _left: QualityCut | None
           Left child cut; used internally for composed cuts.
        _right: QualityCut | None
           Right child cut; used internally for binary compositions.
        """
        self._predicate = predicate
        self.node = node
        self._left = _left
        self._right = _right
        self._required_columns: frozenset[str] = node.all_columns()

    @property
    def name(self) -> str:
        """
        Display name derived from the expression tree.

        Returns
        --------
        name: str
           Human-readable representation of this cut's structure.
        """
        return self.node.name

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply this cut to a DataFrame and return a boolean mask.

        Parameters
        -----------
        df: pd.DataFrame
           The spill data to evaluate.

        Returns
        --------
        mask: pd.Series
           Boolean Series, True for rows passing the cut.

        Raises
        -------
        KeyError
           If any required columns are absent from df.
        """
        missing = self._required_columns - set(df.columns)
        if missing:
            raise KeyError(
                f"QualityCut '{self.name}' requires columns absent from the "
                f"DataFrame: {sorted(missing)}"
            )
        return self._predicate(df)

    def _leaf_cuts(self) -> list[QualityCut]:
        """Return all leaf QualityCut instances in left-to-right expression-tree order."""
        if self.node.op == "leaf":
            return [self]
        result: list[QualityCut] = []
        for child in (self._left, self._right):
            if child is not None:
                result.extend(child._leaf_cuts())
        return result

    def _selectable_cuts(self) -> dict[str, QualityCut]:
        """
        Return all named and leaf cuts in the tree, keyed by name.

        Traversal stops at labeled nodes so that a named composed cut
        is treated as an atom rather than being decomposed further.
        The returned dict preserves left-to-right tree order via
        insertion order; duplicate names keep the first occurrence.
        """
        result: dict[str, QualityCut] = {}
        self._collect_selectable(result)
        return result

    def _collect_selectable(self, result: dict[str, QualityCut]) -> None:
        if self.node.labeled or self.node.op == "leaf":
            result.setdefault(self.name, self)
        for child in (self._left, self._right):
            if child is not None:
                child._collect_selectable(result)

    def __and__(self, other: QualityCut) -> QualityCut:
        """
        Compose this cut with another via logical AND.

        Parameters
        -----------
        other: QualityCut
           The cut to AND with.

        Returns
        --------
        cut: QualityCut
           A new cut passing rows where both self and other pass.
        """
        node = CutNode.make_composite("and", self.node, other.node)
        return QualityCut(
            lambda df, a=self, b=other: a(df) & b(df),
            node, _left=self, _right=other,
        )

    def __or__(self, other: QualityCut) -> QualityCut:
        """
        Compose this cut with another via logical OR.

        Parameters
        -----------
        other: QualityCut
           The cut to OR with.

        Returns
        --------
        cut: QualityCut
           A new cut passing rows where either self or other pass.
        """
        node = CutNode.make_composite("or", self.node, other.node)
        return QualityCut(
            lambda df, a=self, b=other: a(df) | b(df),
            node, _left=self, _right=other,
        )

    def __invert__(self) -> QualityCut:
        """
        Negate this cut via logical NOT.

        Returns
        --------
        cut: QualityCut
           A new cut passing rows where self does not pass.
        """
        node = CutNode.make_composite("not", self.node)
        return QualityCut(
            lambda df, a=self: ~a(df),
            node, _left=self,
        )

    def _build_node_summary(
        self,
        df: pd.DataFrame,
        cache: dict[int, NodeSummary],
        weights: pd.Series | None = None,
    ) -> NodeSummary:
        """
        Recursively build a NodeSummary tree for this cut.

        Results are memoized by object identity so shared sub-cuts are
        evaluated only once.

        Parameters
        -----------
        df: pd.DataFrame
           The spill data to evaluate.
        cache: dict[int, NodeSummary]
           Memoization cache keyed by id(QualityCut).
        weights: pd.Series | None
           Per-row weights (e.g. POT) to sum instead of counting rows.
           None counts rows.

        Returns
        --------
        node: NodeSummary
           Evaluation result for this cut and all descendants.
        """
        cut_id = id(self)
        if cut_id in cache:
            return cache[cut_id]

        mask = self(df)
        if weights is None:
            passed: int | float = int(mask.sum())
            total: int | float = len(df)
        else:
            passed = float(weights[mask].sum())
            total = float(weights.sum())
        children = [
            child._build_node_summary(df, cache, weights)
            for child in (self._left, self._right)
            if child is not None
        ]
        node = NodeSummary(
            name=self.name, passed=passed, total=total,
            op=self.node.op, labeled=self.node.labeled, children=children,
        )
        cache[cut_id] = node
        return node

    def named(self, label: str) -> QualityCut:
        """
        Return a copy of this cut with a user-assigned display label.

        Labeled cuts are treated as opaque nodes in the summary tree and
        will not be absorbed by the flattener, preserving their level in
        the hierarchy regardless of operator.

        Parameters
        -----------
        label: str
           Human-readable name to display in CutSummary output.

        Returns
        --------
        cut: QualityCut
           A new QualityCut with the same predicate and an updated node.
        """
        new_node = CutNode(
            op=self.node.op,
            name=label,
            params=self.node.params,
            columns=self.node.columns,
            left=self.node.left,
            right=self.node.right,
            labeled=True,
        )
        return QualityCut(self._predicate, new_node, _left=self._left, _right=self._right)

    def summary(self, df: pd.DataFrame, *, pot_column: str | None = None) -> CutSummary:
        """
        Evaluate this cut and all sub-cuts against a DataFrame.

        Builds a hierarchical NodeSummary tree mirroring the composition
        structure of this cut. Sub-cuts shared across branches are
        evaluated only once.

        Parameters
        -----------
        df: pd.DataFrame
           The spill data to evaluate.
        pot_column: str | None
           If given, the name of a column holding per-spill POT (e.g.
           'bnb_spill_tor875'). Every passed/total in the resulting
           summary is the summed POT of the relevant rows instead of a
           row count, giving a POT-weighted pass/fail fraction. By
           default None, which counts spills.

        Returns
        --------
        summary: CutSummary
           Total pass count (or POT) and full hierarchical breakdown.
        """
        weights = df[pot_column] if pot_column is not None else None
        tree = self._build_node_summary(df, cache={}, weights=weights)
        return CutSummary(total=tree.total, passed=tree.passed, tree=tree)

    def correlations(
        self,
        df: pd.DataFrame,
        names: list[str] | None = None,
    ) -> CutCorrelations:
        """
        Compute pairwise and individual statistics for a set of cuts.

        By default all leaf cuts are used as the atoms of the analysis,
        collected in left-to-right expression-tree order with duplicates
        removed by name. When *names* is provided, each string is looked
        up in the tree's pool of selectable cuts — labeled (named)
        nodes and leaves — and the analysis uses exactly those cuts in
        the given order. This lets callers analyse composed sub-cuts as
        units rather than decomposing everything to primitives.

        Parameters
        -----------
        df: pd.DataFrame
           The spill data to evaluate.
        names: list[str] | None
           Ordered list of cut names to include in the analysis. Each
           name must match the ``.name`` attribute of a labeled or leaf
           cut reachable from this cut's expression tree. When ``None``
           (default), all unique leaf cuts are used.

        Returns
        --------
        corr: CutCorrelations
           Phi matrix, conditional failure matrix, unique/total
           rejection fractions, and sequential waterfall statistics.

        Raises
        -------
        KeyError
           If any entry in *names* does not match a selectable cut in
           the expression tree.
        """
        if names is not None:
            pool = self._selectable_cuts()
            missing = [n for n in names if n not in pool]
            if missing:
                raise KeyError(
                    f"Cut name(s) not found in expression tree: {missing}. "
                    f"Available: {sorted(pool)}"
                )
            leaf_cuts: list[QualityCut] = [pool[n] for n in names]
        else:
            seen: set[str] = set()
            leaf_cuts = []
            for lc in self._leaf_cuts():
                if lc.name not in seen:
                    seen.add(lc.name)
                    leaf_cuts.append(lc)

        total = len(df)
        names = [lc.name for lc in leaf_cuts]
        fail: dict[str, pd.Series] = {lc.name: ~lc(df) for lc in leaf_cuts}

        fail_df = pd.DataFrame({n: fail[n].values for n in names}, index=df.index)
        phi = fail_df.corr()

        cond_rows: dict[str, dict[str, float]] = {}
        for i in names:
            n_fail_i = int(fail[i].sum())
            cond_rows[i] = {
                j: float((fail[i] & fail[j]).sum()) / n_fail_i if n_fail_i else float("nan")
                for j in names
            }
        conditional = pd.DataFrame(cond_rows).T

        total_rejection = pd.Series({n: float(fail[n].sum()) / total for n in names})
        unique_rejection = pd.Series({
            n: float((fail[n] & ~pd.concat(
                [fail[m] for m in names if m != n], axis=1
            ).any(axis=1)).sum()) / total
            if len(names) > 1 else float(fail[n].sum()) / total
            for n in names
        })

        records = []
        cumulative = pd.Series(True, index=df.index)
        prev_rate = 1.0
        for lc in leaf_cuts:
            cumulative = cumulative & ~fail[lc.name]
            rate = float(cumulative.mean())
            records.append({
                "cut": lc.name,
                "passed": int(cumulative.sum()),
                "total": total,
                "pass_rate": rate,
                "incremental_loss": prev_rate - rate,
            })
            prev_rate = rate
        waterfall = pd.DataFrame(records).set_index("cut")

        return CutCorrelations(
            phi=phi,
            conditional=conditional,
            unique_rejection=unique_rejection,
            total_rejection=total_rejection,
            waterfall=waterfall,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the expression tree to a TOML-compatible dictionary.

        Returns
        --------
        d: dict[str, Any]
           Nested representation of the full expression tree.
        """
        return self.node.to_dict()

    def __repr__(self) -> str:
        return f"QualityCut({self.node!r})"
