from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace, Float

from hpo_glue.benchmark import FunctionalBenchmark
from hpo_glue.measure import Measure
from hpo_glue.result import Result

if TYPE_CHECKING:

    from hpo_glue.query import Query

def branin_bench() -> FunctionalBenchmark:
    branin_space = ConfigurationSpace()
    for i in range(2):
        branin_space.add(Float(name=f"x{i}", bounds=[-32.768, 32.768]))
    return FunctionalBenchmark(
        name="branin",
        config_space=branin_space,
        metrics={
            "value": Measure.metric((0.397887, np.inf), minimize=True),
        },
        query=branin,
    )

def branin(query: Query) -> Result:

    x = np.array(query.config.to_tuple())
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    out = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    return Result(
        query=query,
        fidelity=None,
        values={"value": out},
    )
