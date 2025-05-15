"""
column_alias_manager.py

Utility for mapping user-facing aliases ↔ canonical DataFrame column names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Dict, Hashable, List, Optional

import pandas as pd


class ColumnAliasManager:
    """Map friendly/legacy column names to canonical names."""

    __slots__ = ("_aliases", "_frozen")

    _ATTR_KEY = "alias_map"

    def __init__(self, mapping: Optional[Mapping[str, str]] = None, *, frozen: bool = False):
        self._aliases: Dict[str, str] = {}
        self._frozen: bool = frozen
        if mapping:
            self.add_aliases(mapping)

    # --------------------------------------------------------------------- private helpers
    @staticmethod
    def _normalize(name: str) -> str:
        """Case-fold for reliable matching under all Unicode casings."""
        return name.casefold()

    def _check_frozen(self) -> None:
        if self._frozen:
            raise RuntimeError("ColumnAliasManager is frozen/immutable.")

    # --------------------------------------------------------------------- dunder sugar
    def __getitem__(self, alias: str) -> str:
        return self.canonical(alias)

    def __setitem__(self, alias: str, canonical: str) -> None:
        self.add_alias(alias, canonical)

    def __contains__(self, alias: str) -> bool:
        return self._normalize(alias) in self._aliases

    # --------------------------------------------------------------------- properties
    @property
    def aliases(self) -> Mapping[str, str]:
        """Read-only view of the alias dict (normalized-alias → canonical)."""
        return self._aliases.copy()

    # --------------------------------------------------------------------- mutators
    def add_alias(self, alias: str, canonical: str, *, overwrite: bool = False) -> None:
        self._check_frozen()
        key = self._normalize(alias)
        if not overwrite and key in self._aliases and self._aliases[key] != canonical:
            raise ValueError(f"Alias '{alias}' already mapped to '{self._aliases[key]}'.")
        self._aliases[key] = canonical

    def add_aliases(
        self, mapping: Mapping[str, str], *, overwrite: bool = False
    ) -> None:
        for a, c in mapping.items():
            self.add_alias(a, c, overwrite=overwrite)

    def add_alias_groups(
        self, groups: Mapping[str, Iterable[str]], *, overwrite: bool = False
    ) -> None:
        for canonical, aliases in groups.items():
            for alias in aliases:
                self.add_alias(alias, canonical, overwrite=overwrite)

    def remove_alias(self, alias: str) -> None:
        self._check_frozen()
        self._aliases.pop(self._normalize(alias), None)

    def clear(self) -> None:
        self._check_frozen()
        self._aliases.clear()

    def freeze(self) -> None:
        """Prevent further structural changes."""
        self._frozen = True

    # --------------------------------------------------------------------- queries
    def canonical(self, alias_or_name: str) -> str:
        return self._aliases.get(self._normalize(alias_or_name), alias_or_name)

    def aliases_for(self, canonical: str) -> List[str]:
        return [a for a, c in self._aliases.items() if c == canonical]

    # --------------------------------------------------------------------- batch resolve
    def resolve(
        self, columns: Optional[Iterable[Hashable]]
    ) -> Optional[List[Hashable]]:
        """
        Convert an iterable of column labels to canonical names.
        Non-string labels (e.g., integers, tuples for MultiIndex) are preserved.
        """
        if columns is None:
            return None
        return [
            self.canonical(col) if isinstance(col, str) else col  # pyright: ignore[reportUnknownVariableType]
            for col in columns
        ]

    # --------------------------------------------------------------------- DataFrame helpers
    def to_canonical(
        self,
        df: pd.DataFrame,
        *,
        inplace: bool = False,
        drop_conflicts: bool = False,
        remember: bool = True,
    ) -> pd.DataFrame:
        """
        Rename columns from aliases ➔ canonical. Returns the resulting DataFrame
        (always), so you can chain calls regardless of `inplace`.
        """
        target = df if inplace else df.copy()

        rename_map = {
            col: self._aliases[self._normalize(col)]
            for col in df.columns
            if self._normalize(col) in self._aliases
        }

        # collision detection (case-insensitive)
        collisions = [
            alias
            for alias, canonical in rename_map.items()
            if self._normalize(canonical) in map(self._normalize, df.columns)
            and alias != canonical
        ]
        if collisions:
            if drop_conflicts:
                for a in collisions:
                    rename_map.pop(a, None)
            else:
                raise ValueError(
                    f"Alias(es) {collisions} would overwrite existing canonical columns."
                )

        target.rename(columns=rename_map, inplace=True)

        if remember:
            target.attrs[self._ATTR_KEY] = rename_map

        return target

    def restore_aliases(
        self,
        df: pd.DataFrame,
        *,
        inplace: bool = False,
        strict: bool = True,
    ) -> pd.DataFrame:
        target = df if inplace else df.copy()
        alias_map: Optional[Dict[str, str]] = target.attrs.get(self._ATTR_KEY)

        if alias_map is None:
            if strict:
                raise ValueError(
                    f"No '{self._ATTR_KEY}' in DataFrame.attrs. "
                    "Did you call `to_canonical(..., remember=True)`?"
                )
            return target

        reverse = {canon: alias for alias, canon in alias_map.items()}
        target.rename(columns=reverse, inplace=True)
        return target

    # --------------------------------------------------------------------- persistence
    def to_json(self) -> str:
        """Serialise the mapping to a JSON string."""
        import json

        return json.dumps(self._aliases)

    @classmethod
    def from_json(cls, s: str) -> "ColumnAliasManager":
        import json

        return cls(json.loads(s))


__all__ = ["ColumnAliasManager"]
