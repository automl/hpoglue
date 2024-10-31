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
    kind: type[T]
    values: tuple[T, ...]
    supports_continuation: bool
    min: T
    max: T

    @classmethod
    def from_seq(
        cls,
        values: Sequence[T],
        *,
        supports_continuation: bool = False,
    ) -> ListFidelity[T]:
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
        """Create a RangeFidelity from a tuple of (min, max, stepsize)."""
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
