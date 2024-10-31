from __future__ import annotations

from typing import TYPE_CHECKING

from lib.benchmarks.ackley import ackley_bench
from lib.benchmarks.branin import branin_bench

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription

BENCHMARKS: dict[str, BenchmarkDescription] = {}
BENCHMARKS["ackley"] = ackley_bench().description
BENCHMARKS["branin"] = branin_bench().description

__all__ = [
    "BENCHMARKS"
]
