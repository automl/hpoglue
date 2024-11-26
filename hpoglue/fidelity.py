from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class Orderable(Protocol):
    def __lt__(self, other: Orderable) -> bool: ...
    def __gt__(self, other: Orderable) -> bool: ...
    def __le__(self, other: Orderable) -> bool: ...
    def __ge__(self, other: Orderable) -> bool: ...
    def __eq__(self, other: Orderable) -> bool: ...


T = TypeVar("T", bound=int | float)


@runtime_checkable
class Fidelity(Protocol[T]):
    kind: type[T]
    min: T
    max: T
    supports_continuation: bool

    def __iter__(self) -> Iterator[T]: ...


@dataclass(kw_only=True, frozen=True)
class ListFidelity(Generic[T]):
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
class RangeFidelity(Generic[T]):
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

    def __iter__(self) -> Iterator[T]:
        current = self.min
        yield self.min
        while current < self.max:
            current += self.stepsize
            yield max(current, self.max)  # type: ignore

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

        if not all(isinstance(v, _type) for v in values):
            raise ValueError(f"all values must be of type {_type}, got {values}")

        return cls(
            kind=_type,
            min=values[0],
            max=values[1],
            stepsize=values[2],
            supports_continuation=supports_continuation,
        )

@dataclass(kw_only=True, frozen=True)
class ContinuousFidelity(Generic[T]):
    """A class to represent a continuous fidelity range with a minimum and maximum value.

    Attributes:
        kind: The type of the fidelity values (int or float).

        min: The minimum value of the fidelity range.

        max: The maximum value of the fidelity range.

        supports_continuation: A boolean flag indicating if continuation is supported.
    """
    kind: type[T]
    min: T
    max: T
    supports_continuation: bool

    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError(f"min must be less than max, got {self.min} and {self.max}")

    def __iter__(self) -> Iterator[T]:
        yield self.min
        yield self.max

    @classmethod
    def from_tuple(
        cls,
        values: tuple[T, T],
        *,
        supports_continuation: bool = False,
    ) -> ContinuousFidelity[T]:
        """Create a ContinuousFidelity instance from a tuple of values.

        Args:
            values: A tuple containing two values of type int or float.

            supports_continuation: A flag indicating if continuation is supported.
                Defaults to False.

        Returns:
            ContinuousFidelity: An instance of ContinuousFidelity with the specified values.

        Raises:
            ValueError: If the values are not of type int or float,
                or if the values are not of the same type.
        """
        _type = type(values[0])
        if _type not in (int, float):
            raise ValueError(f"all values must be of type int or float, got {_type}")

        if not all(isinstance(v, _type) for v in values):
            raise ValueError(f"all values must be of type {_type}, got {values}")

        return cls(
            kind=_type,
            min=values[0],
            max=values[1],
            supports_continuation=supports_continuation,
        )
