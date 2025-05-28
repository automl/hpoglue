from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable
from typing_extensions import Self

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=int | float)


@runtime_checkable
class Fidelity(Protocol[T]):
    """A protocol to represent a Fidelity type, which includes a minimum and maximum value of a
    specific type, along with additional metadata such as the type of the values and whether
    continuation is supported.
    """

    kind: type[T]
    min: T
    max: T
    supports_continuation: bool

    @classmethod
    def frm(
        cls,
        values: tuple[T, T] | tuple[T, T, T] | Sequence[T] | range,
        *,
        supports_continuation: bool = False,
    ) -> Fidelity:
        """Create a Fidelity instance from a tuple, sequence, or range object.

        Args:
            values: A tuple of two or three values, a sequence of values, or a range object.

                - If a tuple of two values, the values are the minimum and maximum values,
                    considered on a conitnuous range.
                - If a tuple of three values, the values are the minimum, maximum, and step size,
                  giving a discrete number of fidelity values.
                - If a sequence of values, the values are the fidelity values, which is just
                    a list of values allowed
                - If a range object, the values are the start, stop, and step values.

            supports_continuation: A flag indicating if continuation is supported.
                Defaults to False.

        Returns:
            Fidelity: An instance of Fidelity with the specified values.
        """
        match values:
            case range():
                return RangeFidelity.from_range(values, supports_continuation=supports_continuation)
            case tuple() if len(values) == 2:  # noqa: PLR2004
                return ContinuousFidelity.from_tuple(
                    values, supports_continuation=supports_continuation
                )
            case tuple() if len(values) == 3:  # noqa: PLR2004
                return RangeFidelity.from_tuple(values, supports_continuation=supports_continuation)
            case Sequence():
                return ListFidelity.from_seq(values, supports_continuation=supports_continuation)
            case _:
                raise ValueError(f"Unsupported values type: {type(values)}")


@dataclass(kw_only=True, frozen=True)
class ListFidelity(Fidelity, Sized, Generic[T]):
    """A class to represent a List Fidelity type, which includes a sorted list of values of a
    specific type, along with additional metadata such as the minimum and maximum values,
    and whether the list supports continuation.

    Attributes:
        kind: The type of the elements in the list.

        values: A sorted tuple of values.

        min: The minimum value in the list.

        max: The maximum value in the list.

        supports_continuation: A boolean flag indicating if the list supports continuation.
    """

    kind: type[T]
    values: tuple[T, ...]
    min: T
    max: T
    supports_continuation: bool

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def from_seq(
        cls,
        values: Sequence[T],
        *,
        supports_continuation: bool = False,
    ) -> ListFidelity[T]:
        """Create a ListFidelity instance from a sequence of values.

        Args:
            values: The sequence of values to create the ListFidelity from.

            supports_continuation: Indicates if continuation is supported.
                Defaults to False.

        Returns:
            ListFidelity: An instance of ListFidelity containing the sorted values.
        """
        vs = sorted(values)
        return cls(
            kind=type(vs[0]),
            values=tuple(vs),
            supports_continuation=supports_continuation,
            min=vs[0],
            max=vs[-1],
        )

    def __iter__(self) -> Iterator[T]:
        return iter(self.values)


@dataclass(kw_only=True, frozen=True)
class RangeFidelity(Fidelity, Sized, Generic[T]):
    """A class to represent a Range Fidelity type that iterates over a range of values with a
        specified step size.

    Attributes:
        kind: The type of the range values (int or float).

        min: The minimum value of the range.

        max: The maximum value of the range.

        stepsize: The step size for iterating over the range.

        supports_continuation: A boolean flag indicating if continuation is supported.
    """

    kind: type[T]
    min: T
    max: T
    stepsize: T
    supports_continuation: bool

    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError(f"min must be less than max, got {self.min} and {self.max}")

        if self.stepsize <= 0:
            raise ValueError(f"stepsize must be greater than 0, got {self.stepsize}")

        # Ensure bounds quantize correctly into `n_values` stepsize chunks
        n_values = int((self.max - self.min) / self.stepsize) + 1
        if not np.isclose(self.min + (n_values * self.stepsize) - 1, self.max):
            raise ValueError(
                f"stepsize {self.stepsize} does not divide range [{self.min}, {self.max}]"
            )

    def __iter__(self) -> Iterator[T]:
        current = self.min
        yield self.min
        while current < self.max:
            current += self.stepsize
            yield max(current, self.max)  # type: ignore

    @property
    def n_values(self) -> int:
        """The number of values in the range."""
        return int((self.max - self.min) / self.stepsize) + 1

    def __len__(self) -> int:
        return self.n_values

    @classmethod
    def from_tuple(
        cls,
        values: tuple[T, T, T],
        *,
        supports_continuation: bool = False,
    ) -> RangeFidelity[T]:
        """Create a RangeFidelity instance from a tuple of values.

        Args:
            values: A tuple containing three values of type T (int or float).

            supports_continuation: A flag indicating if continuation is supported.
                Defaults to False.

        Returns:
            RangeFidelity: An instance of RangeFidelity with the specified values.

        Raises:
            ValueError: If the values are not all of type int or float,
                or if the values are not of the same type.
        """
        _type = type(values[0])
        if _type not in (int, float):
            raise ValueError(f"all values must be of type int or float, got {_type}")

        if len(values) != 3:  # noqa: PLR2004
            raise ValueError(f"expected 3 values, got {len(values)}")

        if not all(isinstance(v, _type) for v in values):
            raise ValueError(f"all values must be of type {_type}, got {values}")

        return cls(
            kind=_type,
            min=values[0],
            max=values[1],
            stepsize=values[2],
            supports_continuation=supports_continuation,
        )

    @classmethod
    def from_range(cls, r: range, *, supports_continuation: bool = False) -> Self:
        """Create a RangeFidelity instance from a range object.

        Args:
            r: A range object.
            supports_continuation: A flag indicating if continuation is supported.

        Returns:
            RangeFidelity: An instance of RangeFidelity with the specified values.
        """
        return cls(
            kind=int,
            min=r.start,
            max=r.stop,
            stepsize=r.step,
            supports_continuation=supports_continuation,
        )


@dataclass(kw_only=True, frozen=True)
class ContinuousFidelity(Fidelity, Generic[T]):
    """A class to represent a continuous fidelity range with a minimum and maximum value.

    Attributes:
        kind: The type of the fidelity values (always float for ContinuousFidelity).

        min: The minimum value of the fidelity range.

        max: The maximum value of the fidelity range.

        supports_continuation: A boolean flag indicating if continuation is supported.
    """

    kind: type[T] = field(default=float, init=False)
    min: T
    max: T
    precision: T
    supports_continuation: bool

    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError(f"min must be less than max, got {self.min} and {self.max}")
        assert isinstance(
            self.min, float
        ), f"min must be of type float for ContinuousFidelity. Got {type(self.min)}"
        assert isinstance(self.max, float), (
            f"max must be of type float for ContinuousFidelity. Got {type(self.max)}"
        )
        assert isinstance(self.precision, float), (
            f"precision must be of type float for ContinuousFidelity. Got {type(self.precision)}"
        )
        assert self.precision < self.max, (
            f"precision must be less than max. "
            f"Got precision={self.precision} and max={self.max}"
        )


    @classmethod
    def from_tuple(
        cls,
        values: tuple[T, T],
        precision: float | None = None,
        *,
        supports_continuation: bool = False,
    ) -> ContinuousFidelity[T]:
        """Create a ContinuousFidelity instance from a tuple of values.

        Args:
            values: A tuple containing two values of type int or float.

            precision: The decimal precision at which the fidelity values are queried.

            supports_continuation: A flag indicating if continuation is supported.
                Defaults to False.

        Returns:
            ContinuousFidelity: An instance of ContinuousFidelity with the specified values.

        Raises:
            ValueError: If the values are not of type int or float,
                or if the values are not of the same type.
        """
        if len(values) != 2:  # noqa: PLR2004
            raise ValueError(f"expected 2 values, got {len(values)}")

        _values = []
        for val in values:
            if type(val) not in (int, float):
                raise ValueError(f"all values must be of type int or float, got {type(val)}")
            _values.append(float(val))
        _values = tuple(_values)

        if precision is None:
            precision = 1e-2

        if _values[0] == 0.0:
            logger.error(
                "WARNING: Continuous fidelity with min value 0.0 is not allowed. "
                f"Using `{precision=}` as `min` instead."
                "Change the min value in the benchmark fidelity to a non-zero value "
                "to avoid defaulting to `precision`."
            )
            _values = (precision, _values[1])


        return cls(
            min=_values[0],
            max=_values[1],
            precision=precision,
            supports_continuation=supports_continuation,
        )
