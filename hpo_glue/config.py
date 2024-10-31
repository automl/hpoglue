from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PRECISION = 12

@dataclass
class Config:
    """A configuration to evaluate."""

    config_id: str
    """Some unique identifier"""

    values: dict[str, Any] | None
    """The actual config values to evaluate.

    In the case this config was deserialized, it will likely be `None`.
    """

    def to_tuple(self, precision: int | None = None) -> tuple:
        """Convert the configuration values to a tuple with specified precision.

        Args:
            precision (int | None): The precision to round the float values to.
                                    If None, the default precision is used.

        Returns:
            tuple: A tuple of the configuration values with the specified precision.
        """
        if precision is None:
            precision = PRECISION
        return tuple(
            self.set_precision(
                self.values,
                precision
            ).values()
        )

    def set_precision(self, values: dict, precision: int) -> None:
        """Set the precision of float values in the configuration for continuations.

        Args:
            values (dict): The dictionary of configuration values.
            precision (int): The precision to round the float values to.

        Returns:
            dict: The dictionary with float values rounded to the specified precision.
        """
        for key, value in values.items():
            if isinstance(value, float):
                values[key] = np.round(value, precision)
        return values