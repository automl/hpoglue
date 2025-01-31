from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace

from hpoglue import Config, FunctionalBenchmark, Measure, Result

if TYPE_CHECKING:

    from hpoglue.query import Query


def ackley_fn(x: np.ndarray) -> float:
    """Compute the Ackley function.

    The Ackley function is a widely used benchmark function for testing optimization algorithms.
    It is defined as follows:

        f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c * x_i))) + a + exp(1)

    where:
        - x is a numpy array of input values.
        - n is the number of dimensions (variables), which is set to 2 in this implementation.
        - a, b, and c are constants with typical values a=20, b=0.2, and c=2*pi.

    Parameters:
    x (np.ndarray): Input array of shape (n_var,).

    Returns:
    float: The computed value of the Ackley function.
    """
    n_var=2
    a=20
    b=1/5
    c=2 * np.pi
    part1 = -1. * a * np.exp(-1. * b * np.sqrt((1. / n_var) * np.sum(x * x)))
    part2 = -1. * np.exp((1. / n_var) * np.sum(np.cos(c * x)))

    return part1 + part2 + a + np.exp(1)


def wrapped_ackley(query: Query) -> Result:  # noqa: D103

    y = ackley_fn(
        np.array(query.config.to_tuple())
    )

    return Result(
        query=query,
        fidelity=None,
        values={"y": y},
    )


ACKLEY_BENCH = FunctionalBenchmark(
    name="ackley",
    config_space=ConfigurationSpace(
        {
            f"x{i}": (-32.768, 32.768) for i in range(2)
        }
    ),
    metrics={"y": Measure.metric((0.0, np.inf), minimize=True)},
    query=wrapped_ackley,
    predefined_points={
        "min": (
            Config(
                config_id="min",
                description="This point yields a global optimum of y:0.0",
                values={"x0": 0.0, "x1": 0.0}
            )
        )
    }
)