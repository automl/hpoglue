from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

from more_itertools import roundrobin, take

from hpoglue.budget import CostBudget, TrialBudget
from hpoglue.config import Config
from hpoglue.fidelity import Fidelity, ListFidelity, RangeFidelity
from hpoglue.measure import Measure
from hpoglue.optimizer import Optimizer
from hpoglue.query import Query
from hpoglue.result import Result

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription
    from hpoglue.budget import BudgetType

logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


T = TypeVar("T")


def first(_d: Mapping[str, T]) -> tuple[str, T]:
    return next(iter(_d.items()))


def first_n(n: int, _d: Mapping[str, T]) -> dict[str, T]:
    return dict(take(n, _d.items()))


def mix_n(n: int, _d1: Mapping[str, T], _d2: Mapping[str, T]) -> dict[str, T]:
    return dict(take(n, roundrobin(_d1.items(), _d2.items())))


@dataclass(kw_only=True, unsafe_hash=True)
class Problem:
    """A problem to optimize over."""

    # NOTE: These are mainly for consumers who need to interact beyond forward facing API
    Config: TypeAlias = Config
    Query: TypeAlias = Query
    Result: TypeAlias = Result
    Measure: TypeAlias = Measure
    TrialBudget: TypeAlias = TrialBudget
    CostBudget: TypeAlias = CostBudget
    RangeFidelity: TypeAlias = RangeFidelity
    ListFidelity: TypeAlias = ListFidelity

    objective: tuple[str, Measure] | Mapping[str, Measure] = field(hash=False)
    """The metrics to optimize for this problem, with a specific order.

    If only one metric is specified, this is considered single objective and
    not multiobjective.
    """

    fidelity: tuple[str, Fidelity] | Mapping[str, Fidelity] | None = field(hash=False)
    """Fidelities to use from the Benchmark.

    When `None`, the problem is considered a black-box problem with no fidelity.

    When a single fidelity is specified, the problem is considered a _multi-fidelity_ problem.

    When many fidelities are specified, the problem is considered a _many-fidelity_ problem.
    """

    cost: tuple[str, Measure] | Mapping[str, Measure] | None = field(hash=False)
    """The cost metric to use for this proble.

    When `None`, the problem is considered a black-box problem with no cost.

    When a single cost is specified, the problem is considered a _cost-sensitive_ problem.

    When many costs are specified, the problem is considered a _multi-cost_ problem.
    """

    budget: BudgetType
    """The type of budget to use for the optimizer."""

    optimizer: type[Optimizer] | OptWithHps
    """The optimizer to use for this problem"""

    optimizer_hyperparameters: Mapping[str, int | float] = field(default_factory=dict)
    """The hyperparameters to use for the optimizer"""

    benchmark: BenchmarkDescription
    """The benchmark to use for this problem"""

    is_tabular: bool = field(init=False)
    """Whether the benchmark is tabular"""

    is_multiobjective: bool = field(init=False)
    """Whether the problem has multiple objectives"""

    is_multifidelity: bool = field(init=False)
    """Whether the problem has a fidelity parameter"""

    is_manyfidelity: bool = field(init=False)
    """Whether the problem has many fidelities"""

    supports_trajectory: bool = field(init=False)
    """Whether the problem setup allows for trajectories to be queried."""

    name: str = field(init=False)
    """The name of the problem.

    This is used to identify the problem.
    """

    precision: int = field(default=12) #TODO: Set default

    mem_req_mb: int = field(init=False)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        self.mem_req_mb = self.optimizer.mem_req_mb + self.benchmark.mem_req_mb
        self.is_tabular = self.benchmark.is_tabular
        self.is_manyfidelity: bool
        self.is_multifidelity: bool
        self.supports_trajectory: bool

        name_parts: list[str] = [
            f"optimizer={self.optimizer.name}",
            f"benchmark={self.benchmark.name}",
            self.budget.path_str,
        ]

        if len(self.optimizer_hyperparameters) > 0:
            name_parts.insert(1,
                ",".join(f"{k}={v}" for k, v in self.optimizer_hyperparameters.items())
            )

        self.is_multiobjective: bool
        match self.objective:
            case tuple():
                self.is_multiobjective = False
                # name_parts.append(f"objective={self.objective[0]}")
            case Mapping():
                if len(self.objective) == 1:
                    raise ValueError("Single objective should be a tuple, not a mapping")

                self.is_multiobjective = True
                name_parts.append("objective=" + ",".join(self.objective.keys()))
            case _:
                raise TypeError("Objective must be a tuple (name, measure) or a mapping")

        match self.fidelity:
            case None:
                self.is_multifidelity = False
                self.is_manyfidelity = False
                self.supports_trajectory = False
            case (_name, _fidelity):
                self.is_multifidelity = True
                self.is_manyfidelity = False
                if _fidelity.supports_continuation:
                    self.supports_trajectory = True
                else:
                    self.supports_trajectory = False
                # name_parts.append(f"fidelity={_name}")
            case Mapping():
                if len(self.fidelity) == 1:
                    raise ValueError("Single fidelity should be a tuple, not a mapping")

                self.is_multifidelity = False
                self.is_manyfidelity = True
                self.supports_trajectory = False
                # name_parts.append("fidelity=" + ",".join(self.fidelity.keys()))
            case _:
                raise TypeError("Fidelity must be a tuple (name, fidelity) or a mapping")

        match self.cost:
            case None:
                pass
            case (_name, _measure):
                # name_parts.append(f"cost={_name}")
                pass
            case Mapping():
                if len(self.cost) == 1:
                    raise ValueError("Single cost should be a tuple, not a mapping")

                # name_parts.append("cost=" + ",".join(self.cost.keys()))

        self.name = ".".join(name_parts)


    @classmethod
    def problem(  # noqa: C901, PLR0912, PLR0915
        cls,
        *,
        optimizer: type[Optimizer] | OptWithHps,
        optimizer_hyperparameters: Mapping[str, int | float] = {},
        benchmark: BenchmarkDescription,
        budget: BudgetType | int,
        fidelities: int | None = None,
        objectives: int = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        precision: int | None = None
    ) -> Problem:
        """Generate a problem for this optimizer and benchmark.

        Args:
            optimizer: The optimizer to use for the problem.
            optimizer_hyperparameters: The hyperparameters to use for the optimizer.
            benchmark: The benchmark to use for the problem.
            budget: The budget to use for the problems. Budget defaults to a n_trials budget
                where when multifidelty is enabled, fractional budget can be used and 1 is
                equivalent a full fidelity trial.
            fidelities: The number of fidelities for the problem.
            objectives: The number of objectives for the problem.
            costs: The number of costs for the problem.
            multi_objective_generation: The method to generate multiple objectives.
            precision: The precision to use for the problem.
        """
        _fid: tuple[str, Fidelity] | Mapping[str, Fidelity] | None
        match fidelities:
            case int() if fidelities < 0:
                raise ValueError(f"{fidelities=} must be >= 0")
            case 0:
                _fid = None
            case None:
                _fid = None
            case 1:
                if benchmark.fidelities is None:
                    raise ValueError(
                        (
                            f"Benchmark {benchmark.name} has no fidelities but {fidelities=} "
                            "was requested"
                        ),
                    )
                _fid = first(benchmark.fidelities)
            case int():
                if benchmark.fidelities is None:
                    raise ValueError(
                        (
                            f"Benchmark {benchmark.name} has no fidelities but {fidelities=} "
                            "was requested"
                        ),
                    )

                if fidelities > len(benchmark.fidelities):
                    raise ValueError(
                        f"{fidelities=} is greater than the number of fidelities"
                        f" in benchmark {benchmark.name} which has "
                        f"{len(benchmark.fidelities)} fidelities",
                    )

                _fid = first_n(fidelities, benchmark.fidelities)
            case _:
                raise TypeError(f"{fidelities=} not supported")

        _obj: tuple[str, Measure] | Mapping[str, Measure]
        match objectives, multi_objective_generation:
            # single objective
            case int(), _ if objectives < 0:
                raise ValueError(f"{objectives=} must be >= 0")
            case _, str() if multi_objective_generation not in {"mix_metric_cost", "metric_only"}:
                raise ValueError(
                    f"{multi_objective_generation=} not supported, must be one"
                    " of 'mix_metric_cost', 'metric_only'",
                )
            case 1, _:
                _obj = first(benchmark.metrics)
            case _, "metric_only":
                if objectives > len(benchmark.metrics):
                    raise ValueError(
                        f"{objectives=} is greater than the number of metrics"
                        f" in benchmark {benchmark.name} which has {len(benchmark.metrics)} metrics",
                    )
                _obj = first_n(objectives, benchmark.metrics)
            case _, "mix_metric_cost":
                n_costs = 0 if benchmark.costs is None else len(benchmark.costs)
                n_available = len(benchmark.metrics) + n_costs
                if objectives > n_available:
                    raise ValueError(
                        f"{objectives=} is greater than the number of metrics and costs"
                        f" in benchmark {benchmark.name} which has {n_available} objectives"
                        " when combining metrics and costs",
                    )
                if benchmark.costs is None:
                    _obj = first_n(objectives, benchmark.metrics)
                else:
                    _obj = mix_n(objectives, benchmark.metrics, benchmark.costs)
            case _, _:
                raise RuntimeError(
                    f"Unexpected case with {objectives=}, {multi_objective_generation=}",
                )

        _cost: tuple[str, Measure] | Mapping[str, Measure] | None
        match costs:
            case int() if costs < 0:
                raise ValueError(f"{costs=} must be >= 0")
            case 0:
                _cost = None
            case 1:
                if benchmark.costs is None:
                    raise ValueError(
                        f"Benchmark {benchmark.name} has no costs but {costs=} was requested",
                    )
                _cost = first(benchmark.costs)
            case int():
                if benchmark.costs is None:
                    raise ValueError(
                        f"Benchmark {benchmark.name} has no costs but {costs=} was requested",
                    )
                _cost = first_n(costs, benchmark.costs)
            case _:
                raise TypeError(f"{costs=} not supported")

        _budget: BudgetType
        match budget:
            case int() if budget < 0:
                raise ValueError(f"{budget=} must be >= 0")
            case int():
                _budget = TrialBudget(budget)
            case TrialBudget():
                _budget = budget
            case CostBudget():
                raise NotImplementedError("Cost budgets are not yet supported")
            case _:
                raise TypeError(f"Unexpected type for `{budget=}`: {type(budget)}")

        problem = Problem(
            optimizer=optimizer,
            optimizer_hyperparameters=optimizer_hyperparameters,
            benchmark=benchmark,
            budget=_budget,
            fidelity=_fid,
            objective=_obj,
            cost=_cost,
            precision=precision
        )

        support: Problem.Support = optimizer.support
        support.check_opt_support(who=optimizer.name, problem=problem)

        return problem


    def group_for_optimizer_comparison(
        self,
    ) -> tuple[
        str,
        BudgetType,
        tuple[tuple[str, Measure], ...],
        None | tuple[tuple[str, Fidelity], ...],
        None | tuple[tuple[str, Measure], ...],
    ]:
        match self.objective:
            case (name, measure):
                _obj = ((name, measure),)
            case Mapping():
                _obj = tuple(self.objective.items())

        match self.fidelity:
            case None:
                _fid = None
            case (name, fid):
                _fid = ((name, fid),)
            case Mapping():
                _fid = tuple(self.fidelity.items())

        match self.cost:
            case None:
                _cost = None
            case (name, measure):
                _cost = ((name, measure),)
            case Mapping():
                _cost = tuple(self.cost.items())

        return (self.benchmark.name, self.budget, _obj, _fid, _cost)

    def to_dict(self) -> dict[str, Any]:
        """Convert the problem instance to a dictionary."""
        return {
            "objective": (
                list(self.objective.keys())
                if isinstance(self.objective, Mapping)
                else self.objective[0]
            ),
            "fidelity": (
                None
                if self.fidelity is None
                else (
                    list(self.fidelity.keys())
                    if isinstance(self.fidelity, Mapping)
                    else self.fidelity[0]
                )
            ),
            "cost": (
                None
                if self.cost is None
                else (list(self.cost.keys()) if isinstance(self.cost, Mapping) else self.cost[0])
            ),
            "budget_type": self.budget.name,
            "budget": self.budget.to_dict(),
            "benchmark": self.benchmark.name,
        }

    @classmethod
    def from_dict(  # noqa: C901, PLR0912
        cls,
        data: dict[str, Any],
        benchmarks_dict: Mapping[str, BenchmarkDescription],
    ) -> Problem:
        """Create a Problem instance from a dictionary.

        Args:
            data: A dictionary containing the problem data.
            benchmarks_dict: A mapping of benchmark names to BenchmarkDescription instances.

        Returns:
            A Problem instance created from the dictionary data.
        """
        if data["benchmark"] not in benchmarks_dict:
            raise ValueError(
                f"Benchmark {data['benchmark']} not found in benchmarks!"
                " Please make sure your benchmark is registed in `BENCHMARKS`"
                " before loading/parsing."
            )

        benchmark = benchmarks_dict[data["benchmark"]]
        _obj = data["objective"]
        match _obj:
            case str():
                objective = (_obj, benchmark.metrics[_obj])
            case list():
                objective = {name: benchmark.metrics[name] for name in _obj}
            case _:
                raise ValueError("Objective must be a string or a list of strings")

        _fid = data["fidelity"]
        match _fid:
            case None:
                fidelity = None
            case str():
                assert benchmark.fidelities is not None
                fidelity = (_fid, benchmark.fidelities[_fid])
            case list():
                assert benchmark.fidelities is not None
                fidelity = {name: benchmark.fidelities[name] for name in _fid}
            case _:
                raise ValueError("Fidelity must be a string or a list of strings")

        _cost = data["cost"]
        match _cost:
            case None:
                cost = None
            case str():
                assert benchmark.costs is not None
                cost = (_cost, benchmark.costs[_cost])
            case list():
                assert benchmark.costs is not None
                cost = {name: benchmark.costs[name] for name in _cost}
            case _:
                raise ValueError("Cost must be a string or a list of strings")

        _budget_type = data["budget_type"]
        match _budget_type:
            case "trial_budget":
                budget = TrialBudget.from_dict(data["budget"])
            case "cost_budget":
                budget = CostBudget.from_dict(data["budget"])
            case _:
                raise ValueError("Budget type must be 'trial_budget' or 'cost_budget'")

        return cls(
            objective=objective,
            fidelity=fidelity,
            cost=cost,
            budget=budget,
            benchmark=benchmark,
        )

    @dataclass(kw_only=True)
    class Support:
        """The support of an optimizer for a problem."""

        objectives: tuple[Literal["single", "many"], ...] = field(default=("single",))
        fidelities: tuple[Literal[None, "single", "many"], ...] = field(default=(None,))
        cost_awareness: tuple[Literal[None, "single", "many"], ...] = field(default=(None,))
        tabular: bool = False

        def check_opt_support(self, who: str, *, problem: Problem) -> None:
            """Check if the problem is supported by the support."""
            match problem.fidelity:
                case None if None not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support having no fidelties for {problem.name}!"
                    )
                case tuple() if "single" not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-fidelty for {problem.name}!"
                    )
                case Mapping() if "many" not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support many-fidelty for {problem.name}!"
                    )

            match problem.objective:
                case tuple() if "single" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support single-objective for {problem.name}!"
                    )
                case Mapping() if "many" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-objective for {problem.name}!"
                    )

            match problem.cost:
                case None if None not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support having no cost for {problem.name}!"
                    )
                case tuple() if "single" not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support single-cost for {problem.name}!"
                    )
                case Mapping() if "many" not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-cost for {problem.name}!"
                    )

            match problem.is_tabular:
                case True if not self.tabular:
                    raise ValueError(
                        f"Optimizer {who} does not support tabular benchmarks for {problem.name}!"
                    )
