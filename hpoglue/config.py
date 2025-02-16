from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

PRECISION = 12
"""The default precision to use for floats in configurations."""


@dataclass
class Config(Mapping[str, Any]):
    """A configuration to evaluate."""

    config_id: str
    """Some unique identifier"""

    values: dict[str, Any] | None
    """The actual config values to evaluate.

    In the case this config was deserialized, it will likely be `None`.
    """

    description: str | None = None
    """A description of the configuration."""


    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration values to a dictionary.

        Returns:
            A dictionary of the configuration values.
        """
        assert self.values is not None
        return self.values


    def to_tuple(self, precision: int | None = None) -> tuple:
        """Convert the configuration values to a tuple with specified precision.

        Args:
            precision: The precision to round the float values to.
                If `None`, the default
                [`PRECISION`][hpoglue.config.PRECISION] is used.

        Returns:
            A tuple of the configuration values with the specified precision.
        """
        if precision is None:
            precision = PRECISION

        assert self.values is not None
        return tuple(self.set_precision(self.values, precision).values())

    @staticmethod
    def set_precision(values: dict, precision: int) -> dict[str, Any]:
        """Set the precision of float values in the configuration for continuations.

        Args:
            values: The dictionary of configuration values.
            precision: The precision to round the float values to.

        Returns:
            The dictionary with float values rounded to the specified precision.
        """
        # NOTE: Make sure not to edit the dictionary in place as we return a value.
        return {k: np.round(v, precision) if isinstance(v, float) else v for k, v in values.items()}

    def __getitem__(self, key: str) -> Any:
        assert self.values is not None
        return self.values[key]

    def __iter__(self):
        assert self.values is not None
        return iter(self.values)

    def __len__(self) -> int:
        assert self.values is not None
        return len(self.values)