from __future__ import annotations

from typing import TYPE_CHECKING

from lib.optimizers.random_search import RandomSearch

if TYPE_CHECKING:
    from hpo_glue.optimizer import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    RandomSearch.name: RandomSearch,
}

__all__ = [
    "OPTIMIZERS",
]
