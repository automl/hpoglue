from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from hpoglue import Config, FunctionalBenchmark, Problem
from hpoglue._run import _run
from hpoglue.utils import dict_to_configpriors

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription
    from hpoglue.optimizer import Optimizer


def run_glue(  # noqa: C901, PLR0912
    optimizer: type[Optimizer],
    benchmark: BenchmarkDescription | FunctionalBenchmark,
    objectives: int | str | list[str] = 1,
    fidelities: int | str | list[str] | None = None,
    optimizer_hyperparameters: Mapping[str, int | float] | None = None,
    run_name: str | None = None,
    budget: int | float = 50,
    seed: int = 0,
    *,
    continuations: bool = True,
    priors: tuple[str, Mapping[str, Config | Mapping[str, Any]]] | None = None,
) -> pd.DataFrame:
    """Run the glue function using the specified optimizer, benchmark, and hyperparameters.

        optimizer: The optimizer instance to be used.

        benchmark: The benchmark to be evaluated.

        objectives: The objectives for the benchmark.
            Defaults to 1, the first objective in the benchmark.

        fidelities: The fidelities for the benchmark.

        optimizer_hyperparameters: Hyperparameters for the optimizer.

        run_name: An optional name for the run.

        budget: The budget allocated for the run.

        seed: The seed for random number generation to ensure reproducibility.

        continuations: Whether to use continuations for the run.

        priors: Priors to use for the run.

                Advised usage for priors:
                --------------------------

                ```python
                            priors = (
                                        prior_identifier,
                                        {
                                            "objective_name": Config(
                                                config_id="some identifier for this config ID",
                                                values={"hyperparameter name": "prior value"},
                                            )
                                        }
                                    )
                ```
                NOTE:
                    > The `prior_identifier` must be unique since it is used to identify the prior.

                    > When generating the results, if priors are used the result includes
                    prior=`prior_identifier` for identification.

                    > We suggest the `prior_identifier` to be the name of the objective(s)
                    over which the prior is defined + some unique identifier
                    (eg: value1_good_value2_bad).

    Returns:
        The result of the _run function as a pandas DataFrame.
    """
    if isinstance(benchmark, FunctionalBenchmark):
        benchmark = benchmark.desc

    if priors:
        priors = dict_to_configpriors(priors)

    optimizer_hyperparameters = optimizer_hyperparameters or {}

    problem = Problem.problem(
        optimizer=optimizer,
        optimizer_hyperparameters=optimizer_hyperparameters,
        benchmark=benchmark,
        objectives=objectives,
        fidelities=fidelities,
        budget=budget,
        continuations=continuations,
        priors=priors,
    )

    history = _run(
        run_name=run_name,
        problem=problem,
        seed=seed,
    )
    _df = pd.DataFrame([res._to_dict() for res in history])
    fidelities = problem.get_fidelities()
    match fidelities:
        case None:
            _fidelities = None
        case str():
            _fidelities = [fidelities] * len(_df)
        case list():
            _fidelities = fidelities * len(_df)
        case _:
            raise ValueError(f"Unsupported fidelities type: {type(fidelities)}")

    costs = problem.get_costs()
    match costs:
        case None:
            _costs = None
        case str():
            _costs = [costs] * len(_df)
        case list():
            _costs = costs * len(_df)
        case _:
            raise ValueError(f"Unsupported costs type: {type(costs)}")

    objs = problem.get_objectives()
    match objs:
        case None:
            _objectives = None
        case str():
            _objectives = [objs] * len(_df)
        case list():
            _objectives = objs * len(_df)
        case _:
            raise ValueError(f"Unsupported objectives type: {type(objs)}")

    _df = _df.assign(
        seed=seed,
        optimizer=problem.optimizer.name,
    )
    if len(problem.optimizer_hyperparameters) > 0:
        _df["optimizer_hps"] = ",".join(
            f"{k}={v}" for k, v in problem.optimizer_hyperparameters.items()
        )
    else:
        _df["optimizer_hps"] = "default"
    return _df.assign(
        benchmark=problem.benchmark.name,
        objectives=_objectives,
        fidelities=_fidelities,
        costs=_costs,
    )