from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from hpoglue._run import _run
from hpoglue.benchmark import FunctionalBenchmark
from hpoglue.problem import Problem

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription
    from hpoglue.optimizer import Optimizer


def run_glue(
    optimizer: Optimizer,
    benchmark: BenchmarkDescription | FunctionalBenchmark,
    objectives: int | str | list[str] = 1,
    fidelities: int | str | list[str] | None = None,
    optimizer_hyperparameters: Mapping[str, int | float] = {},
    run_name: str | None = None,
    budget=50,
    seed=0,
    continuations: bool = False,
) -> pd.DataFrame:
    """Run the glue function using the specified optimizer, benchmark, and hyperparameters.

        optimizer: The optimizer instance to be used.

        benchmark: The benchmark to be evaluated.

        objectives: The objectives for the benchmark.
            Defaults to 1, the first objective in the benchmark.

        fidelities: The fidelities for the benchmark.
            Defaults to None.

        optimizer_hyperparameters: Hyperparameters for the optimizer.
            Defaults to an empty dictionary.

        run_name: An optional name for the run. Defaults to None.

        budget: The budget allocated for the run. Defaults to 50.

        seed: The seed for random number generation to ensure reproducibility.
            Defaults to 0.

        continuations: Whether to use continuations for the run. 
            Defaults to False.

    Returns:
        The result of the _run function as a pandas DataFrame.
    """
    if isinstance(benchmark, FunctionalBenchmark):
        benchmark = benchmark.description
    problem = Problem.problem(
        optimizer=optimizer,
        optimizer_hyperparameters=optimizer_hyperparameters,
        benchmark=benchmark,
        objectives=objectives,
        fidelities=fidelities,
        budget=budget,
    )

    history = _run(
        run_name=run_name,
        problem=problem,
        seed=seed,
        continuations=continuations,
    )
    _df = pd.DataFrame([res._to_dict() for res in history])
    return _df.assign(
        seed=seed,
        optimizer=problem.optimizer.name,
        optimizer_hps=problem.optimizer_hyperparameters,
        benchmark=problem.benchmark.name,
        objectives=[problem.get_objectives()]*len(_df),
        fidelities=[problem.get_fidelities()*len(_df)] if problem.get_fidelities() else None,
        costs=[problem.get_costs()*len(_df)] if problem.get_costs() else None,
    )