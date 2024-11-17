from __future__ import annotations

from typing import TYPE_CHECKING

from lib.benchmarks.ackley import ACKLEY_BENCH
from lib.benchmarks.branin import BRANIN_BENCH

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription

BENCHMARKS: dict[str, BenchmarkDescription] = {}
BENCHMARKS["ackley"] = ACKLEY_BENCH
BENCHMARKS["branin"] = BRANIN_BENCH

__all__ = [
    "BENCHMARKS"
]
