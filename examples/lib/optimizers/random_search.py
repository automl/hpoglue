from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace

from hpo_glue.config import Config
from hpo_glue.optimizer import Optimizer
from hpo_glue.problem import Problem
from hpo_glue.query import Query

if TYPE_CHECKING:
    from hpo_glue.result import Result


class RandomSearch(Optimizer):
    """Random Search Optimizer."""

    name = "RandomSearch"

    # NOTE(eddiebergman): Random search doesn't directly use any of this
    # information but we allow it to be used as it's a common baseline.
    support = Problem.Support(
        fidelities=(None,),
        objectives=("single", "many"),
        cost_awareness=(None, "single", "many"),
        tabular=False,
    )

    mem_req_mb = 100  # noqa: N815

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,  # noqa: ARG002
        config_space: ConfigurationSpace | list[Config],
    ):
        """Create a Random Search Optimizer instance for a given problem statement."""
        match config_space:
            case ConfigurationSpace():
                self.config_space = copy.deepcopy(config_space)
                self.config_space.seed(seed)
            case list():
                self.config_space = config_space
            case _:
                raise TypeError("Config space must be a ConfigSpace or a list of Configs")

        self.problem = problem
        self._counter = 0
        self.rng = np.random.default_rng(seed)

    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate."""
        self._counter += 1
        # We are dealing with a tabular benchmark
        match self.config_space:
            case ConfigurationSpace():
                config = Config(
                    config_id=str(self._counter),
                    values=self.config_space.sample_configuration().get_dictionary(),
                )
            case list():
                index = int(self.rng.integers(len(self.config_space)))
                config = self.config_space[index]
            case _:
                raise TypeError("Config space must be a ConfigSpace or a list of Configs")

        match self.problem.fidelity:
            case None:
                fidelity = None
            case (name, fidelity):
                fidelity = (name, fidelity.max)
            case Mapping():
                fidelity = {name: fidelity.max for name, fidelity in self.problem.fidelity.items()}
            case _:
                raise ValueError("Fidelity must be a string or a list of strings")

        return Query(config=config, fidelity=fidelity)

    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""
        # NOTE(eddiebergman): Random search does nothing with the result
