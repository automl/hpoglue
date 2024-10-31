from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from hpo_glue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpo_glue.env import Env

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from hpo_glue.benchmarks import BenchmarkDescription
    from hpo_glue.budget import BudgetType
    from hpo_glue.config import Config
    from hpo_glue.problem import Problem, Run
    from hpo_glue.query import Query
    from hpo_glue.result import Result


class Optimizer(ABC):
    """Defines the common interface for Optimizers."""

    name: ClassVar[str]
    """The name of the optimizer"""

    support: ClassVar[Problem.Support]
    """What kind of problems the optimizer supports"""

    multi_fideltiy_requires_learning_curve: ClassVar[bool] = False
    """Whether the optimizer requires a learning curve for multi-fidelity optimization.

    If `True` and the problem is multi-fidelity (1 fidelity) and the fidelity
    supports continuations (e.g. epochs), then a trajectory will be provided in
    the `Result` object provided during the `tell` to the optimizer.
    """

    env: ClassVar[Env] = Env.empty()
    """The environment to setup the optimizer in for `isolated` mode.

    If left as `None`, the currently activated environemnt will be used.
    """

    mem_req_mb: ClassVar[int]
    """The memory requirement of the optimizer in mb."""

    @abstractmethod
    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        config_space: list[Config] | ConfigurationSpace,
        working_directory: Path,
        **optimizer_kwargs: Any,
    ) -> None:
        """Initialize the optimizer.

        Args:
            problem: The problem to optimize over
            seed: The random seed for the optimizer
            config_space: The configuration space to optimize over
            working_directory: The directory to save the optimizer's state
            optimizer_kwargs: Any additional hyperparameters for the optimizer
        """

    @abstractmethod
    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate."""

    @abstractmethod
    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""

    @classmethod
    def generate_problems(  # noqa: PLR0913
        cls,
        benchmarks: BenchmarkDescription | list[BenchmarkDescription],
        *,
        expdir: Path | str = DEFAULT_RELATIVE_EXP_DIR,
        hyperparameters: dict[str, Any] | list[dict[str, Any]] | None = None,
        budget: BudgetType | int,
        seeds: int | Iterable[int],
        fidelities: int = 0,
        objectives: int = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        on_error: Literal["warn", "raise", "ignore"] = "warn",
    ) -> Iterator[Run]:
        """Generate a set of problems for the given optimizer and benchmark.

        If there is some incompatibility between the optimizer, the benchmark and the requested
        amount of objectives, fidelities or costs, a ValueError will be raised.

        Args:
            benchmarks: The benchmark to generate problems for.
                Can provide a single benchmark or a list of benchmarks.
            expdir: Which directory to store experiment results into.
            hyperparameters: The hyperparameters to use for the optimizer, if any.
            budget: The budget to use for the problems. Budget defaults to a n_trials budget
                where when multifidelty is enabled, fractional budget can be used and 1 is
                equivalent a full fidelity trial.
            seeds: The seed or seeds to use for the problems.
            fidelities: The number of fidelities to generate problems for.
            objectives: The number of objectives to generate problems for.
            costs: The number of costs to generate problems for.
            multi_objective_generation: The method to generate multiple objectives.
            on_error: The method to handle errors.

                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.
        """
        from hpo_glue._problem_generators import _generate_problem_set

        opt_with_hps: list[tuple[type[Optimizer], Mapping[str, Any]]]
        match hyperparameters:
            case None:
                opt_with_hps = [(cls, {})]
            case dict():
                opt_with_hps = [(cls, hyperparameters)]
            case list():
                opt_with_hps = [(cls, hps) for hps in hyperparameters]

        yield from _generate_problem_set(
            optimizers=opt_with_hps,  # type: ignore
            benchmarks=benchmarks,
            budget=budget,
            seeds=seeds,
            expdir=expdir,
            fidelities=fidelities,
            objectives=objectives,
            costs=costs,
            multi_objective_generation=multi_objective_generation,
            on_error=on_error,
        )
