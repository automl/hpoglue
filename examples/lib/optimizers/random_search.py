from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from ConfigSpace import ConfigurationSpace

from hpoglue import Config, Optimizer, Problem, Query

if TYPE_CHECKING:
    from hpoglue import Result


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

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,  # noqa: ARG002
    ):
        """Create a Random Search Optimizer instance for a given problem statement."""
        match problem.config_space:
            case ConfigurationSpace():
                self.config_space = copy.deepcopy(problem.config_space)
                self.config_space.seed(seed)
            case list():
                self.config_space = problem.config_space
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
                    values=dict(self.config_space.sample_configuration())
                )
            case list():
                index = int(self.rng.integers(len(self.config_space)))
                config = self.config_space[index]
            case _:
                raise TypeError("Config space must be a ConfigSpace or a list of Configs")

        match self.problem.fidelities:
            case None:
                fidelity = None
            case (name, fidelity):
                fidelity = (name, fidelity.max)
            case Mapping():
                fidelity = {
                    name: fidelity.max for name, fidelity in self.problem.fidelities.items()
                }
            case _:
                raise ValueError("Fidelity must be a string or a list of strings")

        return Query(config=config, fidelity=fidelity)

    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""
        # NOTE(eddiebergman): Random search does nothing with the result


class RandomSearchWithPriors(Optimizer):
    """Random Search Optimizer with Priors over objectives."""

    name = "RandomSearchWithPriors"

    support = Problem.Support(
        objectives=("single", "many"),
        fidelities=(None),
        cost_awareness=(None),
        tabular=False,
        priors=True,
    )

    mem_req_mb = 1024

    def __init__(  # noqa: D107
        self,
        problem: Problem,
        seed: int,
        working_directory: Path,  # noqa: ARG002
        mo_prior_sampling: Literal["random", "equal"] = "random",
    ) -> None:
        self.config_space = problem.config_space
        self.config_space.seed(seed)
        self.problem = problem
        _priors: Mapping[str, Config] = self.problem.priors
        self.priors = {
            obj: _create_normal_prior(
                config_space=self.config_space,
                seed=seed,
                prior_defaults=prior,
                std=0.25,
            )
            for obj, prior in _priors.items()
        }
        self._rng = np.random.default_rng(seed)
        if mo_prior_sampling == "equal":
            assert len(self.problem.get_objectives()) <= self.problem.budget.total, (
                "When using `mo_prior_sampling='equal'` the number of objectives "
                "should be less than or equal to the total budget."
            )
        self.mo_prior_sampling = mo_prior_sampling
        self._optmizer_unique_id = 0
        self._priors_used = {key: 0 for key in self.priors}

    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate."""
        self._optmizer_unique_id += 1
        if len(self.priors) > 1:
            match self.mo_prior_sampling:
                case "random":
                    prior = self._rng.choice(list(self.priors.values()))
                case "equal":
                    raise NotImplementedError
                case _:
                    raise ValueError(
                        "Invalid value for `mo_prior_sampling`. "
                        "Expected one of ['random', 'equal']."
                        f"Got {self.mo_prior_sampling}."
                    )
        else:
            prior = next(iter(self.priors.values()))
        config = Config(
            config_id=str(self._optmizer_unique_id),
            values=dict(prior.sample_configuration()),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""
        # NOTE(eddiebergman): Random search does nothing with the result


def _create_normal_prior(
    config_space: ConfigurationSpace,
    seed: int,
    prior_defaults: Config,
    std: float,
) -> ConfigurationSpace:
    from ConfigSpace import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
    prior_space = ConfigurationSpace(seed=seed)
    for hp in list(config_space.values()):
        _default = prior_defaults.values[hp.name]
        match hp:
            case UniformFloatHyperparameter() | BetaFloatHyperparameter():
                prior_space.add(
                    NormalFloatHyperparameter(
                        name=hp.name,
                        mu=_default,
                        sigma=std,
                        lower=hp.lower,
                        upper=hp.upper,
                        default_value=_default,
                        log=hp.log,
                    )
                )
            case UniformIntegerHyperparameter() | BetaIntegerHyperparameter():
                prior_space.add(
                    NormalIntegerHyperparameter(
                        name=hp.name,
                        mu=_default,
                        sigma=std,
                        lower=hp.lower,
                        upper=hp.upper,
                        default_value=_default,
                        log=hp.log,
                    )
                )
            case CategoricalHyperparameter() | OrdinalHyperparameter():
                hp.default_value = _default
                prior_space.add(hp)
            case Constant():
                prior_space.add(hp)
            case NormalFloatHyperparameter() | NormalIntegerHyperparameter():
                hp.mu = _default
                hp.sigma = std
                hp.default_value = _default
                prior_space.add(hp)
            case _:
                raise ValueError(
                    f"Unsupported hyperparameter type: {type(hp).__name__}"
                )
    return prior_space