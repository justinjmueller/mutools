"""
Expression-tree node for quality-cut provenance tracking and serialization.
"""
from __future__ import annotations

from typing import Any


class CutNode:
    """Node in the expression tree used for provenance tracking and serialization."""

    __slots__ = ("op", "name", "params", "columns", "left", "right", "labeled")

    def __init__(
        self,
        op: str,
        name: str = "",
        params: dict[str, Any] | None = None,
        columns: frozenset[str] | None = None,
        left: CutNode | None = None,
        right: CutNode | None = None,
        labeled: bool = False,
    ) -> None:
        self.op = op
        self.name = name
        self.params: dict[str, Any] = params or {}
        self.columns: frozenset[str] = columns or frozenset()
        self.left = left
        self.right = right
        self.labeled = labeled

    @classmethod
    def make_leaf(cls, name: str, params: dict[str, Any], columns: frozenset[str]) -> CutNode:
        """Construct a leaf node from a named predicate."""
        return cls(op="leaf", name=name, params=params, columns=columns)

    @classmethod
    def make_composite(cls, op: str, left: CutNode, right: CutNode | None = None) -> CutNode:
        """Construct a composite node from an operator and children."""
        name = f"(~{left.name})" if right is None else f"({left.name} {op} {right.name})"
        return cls(op=op, name=name, left=left, right=right)

    def all_columns(self) -> frozenset[str]:
        """Collect all required DataFrame columns in this subtree."""
        result = self.columns
        for child in (self.left, self.right):
            if child is not None:
                result = result | child.all_columns()
        return result

    def all_leaves(self) -> list[CutNode]:
        """Collect all leaf nodes reachable in this subtree."""
        if self.op == "leaf":
            return [self]
        result: list[CutNode] = []
        for child in (self.left, self.right):
            if child is not None:
                result.extend(child.all_leaves())
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this node to a TOML-compatible dictionary.

        Returns
        --------
        d: dict[str, Any]
           For a leaf: {'function': name, **params}.
           For composites: {'operator': op, 'operands': [...]}.
        """
        if self.op == "leaf":
            return {"function": self.name, **self.params}
        children = [c.to_dict() for c in (self.left, self.right) if c is not None]
        return {"operator": self.op, "operands": children}

    def __repr__(self) -> str:
        return self.name or self.op
