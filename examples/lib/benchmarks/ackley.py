from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace, Float

from hpo_glue.benchmark import FunctionalBenchmark
from hpo_glue.measure import Measure
from hpo_glue.result import Result

if TYPE_CHECKING:

    from hpo_glue.query import Query

def ackley_bench() -> FunctionalBenchmark:
    ackley_space = ConfigurationSpace()
    for i in range(2):
        ackley_space.add(Float(name=f"x{i}", bounds=[-32.768, 32.768]))
    return FunctionalBenchmark(
        name="ackley",
        config_space=ackley_space,
        metrics={
            "value": Measure.metric((0.0, np.inf), minimize=True),
        },
        query=ackley,
    )

def ackley(query: Query) -> Result:

    x = np.array(query.config.to_tuple())
    n_var=2
    a=20
    b=1/5
    c=2 * np.pi
    part1 = -1. * a * np.exp(-1. * b * np.sqrt((1. / n_var) * np.sum(x * x)))
    part2 = -1. * np.exp((1. / n_var) * np.sum(np.cos(c * x)))
    out = part1 + part2 + a + np.exp(1)

    return Result(
        query=query,
        fidelity=None,
        values={"value": out},
    )
