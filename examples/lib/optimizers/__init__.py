from __future__ import annotations

from typing import TYPE_CHECKING

from lib.optimizers.random_search import RandomSearch, RandomSearchWithPriors

if TYPE_CHECKING:
    from hpoglue.optimizer import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    RandomSearch.name: RandomSearch,
    RandomSearchWithPriors.name: RandomSearchWithPriors,
}

__all__ = [
    "OPTIMIZERS",
]
