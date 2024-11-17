from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace

from hpoglue import FunctionalBenchmark, Measure, Result

if TYPE_CHECKING:

    from hpoglue import Query


def branin_fn(x: np.ndarray) -> float:
    """Compute the value of the Branin function.

    The Branin function is a commonly used test function for optimization algorithms.
    It is defined as:

        f(x) = a * (x2 - b * x1^2 + c * x1 - r)^2 + s * (1 - t) * cos(x1) + s

    where:
        b = 5.1 / (4.0 * pi^2)
        c = 5.0 / pi
        t = 1.0 / (8.0 * pi)

    Args:
        x (np.ndarray): A 2-dimensional input array where x[0] is x1 and x[1] is x2.

    Returns:
        float: The computed value of the Branin function.
    """
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def wrapped_branin(query: Query) -> Result:  # noqa: D103

    y = branin_fn(
        np.array(query.config.to_tuple())
    )

    return Result(
        query=query,
        fidelity=None,
        values={"value": y},
    )


BRANIN_BENCH = FunctionalBenchmark(
    name="branin",
    config_space=ConfigurationSpace(
        {
            f"x{i}": (-32.768, 32.768) for i in range(2)
        }
    ),
    metrics={
            "value": Measure.metric((0.397887, np.inf), minimize=True),
        },
    query=wrapped_branin
)