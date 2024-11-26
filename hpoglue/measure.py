from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class OutOfBoundsError(ValueError):
    """Raised when a value is outside of the bounds of a measure."""


@dataclass(kw_only=True, frozen=True)
class Measure:
    """A measure for a becnhmark.

    It's main use is to convert a raw value into a value that can always be
    minimized.
    """

    minimize: bool
    """Whether or not to smaller is better."""

    kind: Measure.Kind
    """What kind of measure this is."""

    bounds: tuple[float, float] = field(default_factory=lambda: (-np.inf, np.inf))
    """The bounds of the measure."""

    @classmethod
    def metric(
        cls,
        bounds: tuple[float, float],
        *,
        minimize: bool,
    ) -> Measure:
        """Create a metric measure.

        Args:
            minimize: Whether or not to minimize the value.
            bounds: The bounds of the metric.

        Returns:
            The metric measure.
        """
        return cls(minimize=minimize, kind=cls.Kind.METRIC, bounds=bounds)

    @classmethod
    def cost(cls, bounds: tuple[float, float], *, minimize: bool = True) -> Measure:
        """Create a cost measure.

        Args:
            bounds: The bounds of the cost.
            minimize: Whether or not to minimize the value.

        Returns:
            The cost measure.
        """
        return cls(minimize=minimize, kind=cls.Kind.COST, bounds=bounds)

    @classmethod
    def test_metric(
        cls,
        bounds: tuple[float, float],
        *,
        minimize: bool,
    ) -> Measure:
        """Create a test metric measure.

        Args:
            minimize: Whether or not to minimize the value.
            bounds: The bounds of the metric.

        Returns:
            The metric measure.
        """
        return cls(minimize=minimize, kind=cls.Kind.TEST_METRIC, bounds=bounds)

    class Kind(str, Enum):
        """Enumeration of measure kinds."""
        METRIC = "metric"
        COST = "cost"
        TEST_METRIC = "test_metric"

        def __str__(self) -> str:
            return self.value

    def __post_init__(self) -> None:
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"bounds[0] must be less than bounds[1], got {self.bounds}",
            )

    @property
    def optimum(self) -> float:
        """Get the optimum value for this metric.

        Returns:
            The optimum value.
        """
        match self.minimize:
            case True:
                return self.bounds[0]
            case False:
                return self.bounds[1]

    def as_minimize(self, value: float) -> float:
        """Convert a raw value into a value that should be minimized."""
        match self.minimize:
            case True:
                return float(value)
            case False:
                return -float(value)

    def as_maximize(self, value: float) -> float:
        """Convert a raw value into a value that should be maximized."""
        match self.minimize:
            case True:
                return -float(value)
            case False:
                return float(value)
