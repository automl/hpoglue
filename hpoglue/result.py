from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from hpoglue.config import Config
    from hpoglue.query import Query


@dataclass(kw_only=True)
class Result:
    """The result of a query from a benchmark."""

    query: Query
    """The query that generated this result"""

    fidelity: tuple[str, int | float] | Mapping[str, int | float] | None
    """What fidelity the result is at, usually this will be the same as the query fidelity,
    unless the benchmark has multiple fidelities.
    """

    values: dict[str, Any]
    """Everything returned by the benchmark for a given query at the fideltiy."""

    continuations_cost: float = np.nan
    """The coninuations cost if run.continuations set to True."""

    budget_cost: float = np.nan
    """The amount of budget used to generate this result."""

    budget_used_total: float = np.nan
    """The amount of budget used in total."""

    trajectory: pd.DataFrame | None = None
    """If given, the trajectory of the query up to the given fidelity.

    This will only provided if:
    * The problem says it should be provided.
    * There is only a single fidelity parameter.

    It will be multi-indexed, with the multi indexing consiting of the
    config id and the fidelity.
    """

    @property
    def config(self) -> Config:
        """The config."""
        return self.query.config

    def _to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""
        members = [
            attr for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        members.remove("query")
        members.remove("config")
        members.remove("trajectory")

        _self_dict = {attr: getattr(self, attr) for attr in members}
        _self_dict.update(self.query._to_dict())
        _self_dict["results"] = _self_dict.pop("values")

        return _self_dict