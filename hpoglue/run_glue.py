from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from hpoglue._run import _run
from hpoglue.problem import Problem

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription
    from hpoglue.optimizer import Optimizer


def run_glue(
    optimizer: Optimizer,
    benchmark: BenchmarkDescription,
    optimizer_hyperparameters: Mapping[str, int | float] = {},
    run_name: str | None = None,
    budget=50,
    seed=0,
):
    """Run the glue function with the given optimizer, benchmark, and hyperparameters.

    Args:
        optimizer (Optimizer): The optimizer to use.
        benchmark (BenchmarkDescription): The benchmark description.
        optimizer_hyperparameters (Mapping[str, int | float]): Hyperparameters for the optimizer.
        run_name (str | None, optional): The name of the run. Defaults to None.
        budget (int, optional): The budget for the run. Defaults to 50.
        seed (int, optional): The seed for random number generation. Defaults to 0.

    Returns:
        The result of the _run function.
    """
    problem = Problem.problem(
        optimizer=optimizer,
        optimizer_hyperparameters=optimizer_hyperparameters,
        benchmark=benchmark,
        budget=budget,
    )

    return _run(
        run_name=run_name,
        problem=problem,
        seed=seed,
    )