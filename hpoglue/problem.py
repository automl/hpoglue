from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from hpoglue.budget import CostBudget, TrialBudget
from hpoglue.config import PRECISION, Config
from hpoglue.fidelity import ContinuousFidelity, Fidelity, ListFidelity, RangeFidelity
from hpoglue.measure import Measure
from hpoglue.optimizer import Optimizer
from hpoglue.query import Query
from hpoglue.result import Result
from hpoglue.utils import configpriors_to_dict, dict_to_configpriors, first, first_n, mix_n

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from hpoglue.benchmark import BenchmarkDescription
    from hpoglue.budget import BudgetType

logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


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
    ContinuousFidelity: TypeAlias = ContinuousFidelity

    objectives: tuple[str, Measure] | Mapping[str, Measure] = field(hash=False)
    """The metrics to optimize for this problem, with a specific order.

    If only one metric is specified, this is considered single objective and
    not multiobjective.
    """

    fidelities: tuple[str, Fidelity] | Mapping[str, Fidelity] | None = field(hash=False)
    """Fidelities to use from the Benchmark.

    When `None`, the problem is considered a black-box problem with no fidelity.

    When a single fidelity is specified, the problem is considered a _multi-fidelity_ problem.

    When many fidelities are specified, the problem is considered a _many-fidelity_ problem.
    """

    costs: tuple[str, Measure] | Mapping[str, Measure] | None = field(hash=False)
    """The cost metrics to use for this problem.

    When `None`, the problem is considered a black-box problem with no cost.

    When a single cost is specified, the problem is considered a _cost-sensitive_ problem.

    When many costs are specified, the problem is considered a _multi-cost_ problem.
    """

    budget: BudgetType
    """The type of budget to use for the optimizer."""

    optimizer: type[Optimizer]
    """The optimizer to use for this problem"""

    optimizer_hyperparameters: Mapping[str, int | float] = field(default_factory=dict)
    """The hyperparameters to use for the optimizer"""

    benchmark: BenchmarkDescription
    """The benchmark to use for this problem"""

    config_space: ConfigurationSpace | list[Config] = field(init=False)
    """The configuration space for the problem"""

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

    precision: int = field(default=12)
    """The precision to use for the problem."""

    mem_req_mb: int = field(init=False)
    """The memory requirement in MB for the problem.
        It is the sum of the optimizer and benchmark memory requirements.
    """

    continuations: bool = field(default=True)
    """Whether the problem supports continuations."""

    priors: tuple[str, Mapping[str, Config]] = field(default_factory=dict)
    """Priors to use for the problem's objectives.
        Format: (unique_prior_id, {objective_name: prior Config})
    """

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        self.config_space = self.benchmark.config_space
        self.mem_req_mb = self.optimizer.mem_req_mb + self.benchmark.mem_req_mb
        self.is_tabular = self.benchmark.is_tabular
        self.is_manyfidelity: bool
        self.is_multifidelity: bool
        self.supports_trajectory: bool

        name_parts: list[str] = [
            f"optimizer={self.optimizer.name}",
            f"benchmark={self.benchmark.name}",
            "objectives=" + (
                ",".join(self.get_objectives())
                if isinstance(self.objectives, Mapping)
                else self.get_objectives()
            )
        ]

        if len(self.optimizer_hyperparameters) > 0:
            name_parts.insert(
                1, ",".join(f"{k}={v}" for k, v in self.optimizer_hyperparameters.items())
            )

        if self.fidelities is not None:
            name_parts.append(
                "fidelities=" + (
                    ",".join(self.get_fidelities())
                    if isinstance(self.fidelities, Mapping)
                    else self.get_fidelities()
                )
            )

        if self.costs is not None:
            name_parts.append(
                "costs=" + (
                    ",".join(self.get_costs())
                    if isinstance(self.costs, Mapping)
                    else self.get_costs()
                )
            )

        name_parts.append(self.budget.path_str)

        if self.priors:
            name_parts.append(
                f"priors={self.priors[0]}"
            )

        self.name = ".".join(name_parts)

        self.is_multiobjective: bool
        match self.objectives:
            case tuple():
                self.is_multiobjective = False
            case Mapping():
                if len(self.objectives) == 1:
                    raise ValueError("Single objective should be a tuple, not a mapping")

                self.is_multiobjective = True
            case _:
                raise TypeError("Objectives must be a tuple (name, measure) or a mapping")

        match self.fidelities:
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
            case Mapping():
                if len(self.fidelities) == 1:
                    raise ValueError("Single fidelity should be a tuple, not a mapping")

                self.is_multifidelity = False
                self.is_manyfidelity = True
                self.supports_trajectory = False
            case _:
                raise TypeError("Fidelity must be a tuple (name, fidelity) or a mapping")

        match self.costs:
            case None:
                pass
            case (_name, _measure):
                pass
            case Mapping():
                if len(self.costs) == 1:
                    raise ValueError("Single cost should be a tuple, not a mapping")

    @classmethod
    def problem(  # noqa: C901, PLR0912, PLR0913, PLR0915
        cls,
        *,
        optimizer: type[Optimizer],
        optimizer_hyperparameters: Mapping[str, int | float] = {},
        benchmark: BenchmarkDescription,
        budget: BudgetType | int | float,
        fidelities: int | str | list[str] | None = None,
        objectives: int | str | list[str] = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        precision: int | None = None,
        continuations: bool = True,
        priors: tuple[str, Mapping[str, Config] | Mapping[str, dict[str, Any]]] | None = None,
    ) -> Problem:
        """Generate a problem for this optimizer and benchmark.

        Args:
            optimizer: The optimizer to use for the problem.

            optimizer_hyperparameters: The hyperparameters to use for the optimizer.

            benchmark: The benchmark to use for the problem.

            budget: The budget to use for the problems. Budget defaults to a n_trials budget
                where when multifidelty is enabled, fractional budget can be used and 1 is
                equivalent a full fidelity trial.

            fidelities: The actual fidelities or number of fidelities for the problem.

            objectives: The actual objectives or number of objectives for the problem.

            costs: The number of costs for the problem.

            multi_objective_generation: The method to generate multiple objectives.

            precision: The precision to use for the problem.

            continuations: Whether to use continuations for the problem.

            priors: Priors to use for the problem's objectives.
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
            case str():
                if benchmark.fidelities is None:
                    raise ValueError(
                        (
                            f"Benchmark {benchmark.name} has no fidelities but {fidelities=} "
                            "was requested"
                        ),
                    )
                if fidelities not in benchmark.fidelities:
                    raise ValueError(
                        f"{fidelities=} not found in benchmark {benchmark.name} fidelities",
                    )
                _fid = (fidelities, benchmark.fidelities[fidelities])
            case list():
                if benchmark.fidelities is None:
                    raise ValueError(
                        (
                            f"Benchmark {benchmark.name} has no fidelities but {fidelities=} "
                            "was requested"
                        ),
                    )
                if len(fidelities) > len(benchmark.fidelities):
                    raise ValueError(
                        f"{fidelities=} is greater than the number of fidelities"
                        f" in benchmark {benchmark.name} which has "
                        f"{len(benchmark.fidelities)} fidelities",
                    )
                _fid = {name: benchmark.fidelities[name] for name in fidelities}
            case _:
                raise TypeError(f"{fidelities=} not supported")

        _obj: tuple[str, Measure] | Mapping[str, Measure]
        match objectives, multi_objective_generation:
            # single objective
            case int(), _ if objectives < 0:  # type: ignore
                raise ValueError(f"{objectives=} must be >= 0")
            case _, str() if multi_objective_generation not in {"mix_metric_cost", "metric_only"}:
                raise ValueError(
                    f"{multi_objective_generation=} not supported, must be one"
                    " of 'mix_metric_cost', 'metric_only'",
                )
            case 1, _:
                _obj = first(benchmark.metrics)
            case int(), "metric_only":
                if objectives > len(benchmark.metrics):  # type: ignore
                    raise ValueError(
                        f"{objectives=} is greater than the number of metrics"
                        f" in benchmark {benchmark.name} which has {len(benchmark.metrics)} metrics"
                    )
                _obj = first_n(objectives, benchmark.metrics)  # type: ignore
            case int(), "mix_metric_cost":
                n_costs = 0 if benchmark.costs is None else len(benchmark.costs)
                n_available = len(benchmark.metrics) + n_costs
                if objectives > n_available:  # type: ignore
                    raise ValueError(
                        f"{objectives=} is greater than the number of metrics and costs"
                        f" in benchmark {benchmark.name} which has {n_available} objectives"
                        " when combining metrics and costs",
                    )
                if benchmark.costs is None:
                    _obj = first_n(objectives, benchmark.metrics)  # type: ignore
                else:
                    _obj = mix_n(objectives, benchmark.metrics, benchmark.costs)  # type: ignore
            case str(), _:
                if objectives not in benchmark.metrics:
                    raise ValueError(
                        f"{objectives=} not found in benchmark {benchmark.name} metrics",
                    )
                _obj = (objectives, benchmark.metrics[objectives])  # type: ignore
            case list(), "metric_only":
                if len(objectives) > len(benchmark.metrics):  # type: ignore
                    raise ValueError(
                        f"{objectives=} is greater than the number of metrics"
                        f" in benchmark {benchmark.name} which has {len(benchmark.metrics)} metrics"
                    )
                _obj = {name: benchmark.metrics[name] for name in objectives}  # type: ignore
            case list(), "mix_metric_cost":
                n_costs = 0 if benchmark.costs is None else len(benchmark.costs)
                n_available = len(benchmark.metrics) + n_costs
                if len(objectives) > n_available:  # type: ignore
                    raise ValueError(
                        f"{objectives=} is greater than the number of metrics and costs"
                        f" in benchmark {benchmark.name} which has {n_available} objectives"
                        " when combining metrics and costs",
                    )
                if benchmark.costs is None:
                    for obj in objectives:  # type: ignore
                        if obj not in benchmark.metrics:
                            raise ValueError(
                                f"{obj=} not found in benchmark {benchmark.name} metrics",
                            )
                    _obj = {name: benchmark.metrics[name] for name in objectives}  # type: ignore
                else:
                    _obj = mix_n(len(objectives), benchmark.metrics, benchmark.costs)  # type: ignore
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
            case None:
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
            case float():
                _budget = CostBudget(budget)
            case TrialBudget():
                _budget = budget
            case CostBudget():
                raise NotImplementedError("Cost budgets are not yet supported")
            case _:
                raise TypeError(f"Unexpected type for `{budget=}`: {type(budget)}")

        _opt = optimizer[0] if isinstance(optimizer, tuple) else optimizer

        match priors:
            case None:
                pass
            case tuple():
                priors = dict_to_configpriors(priors)
            case _:
                raise TypeError(f"Unexpected type for priors: {type(priors)}")

        if "single" not in _opt.support.fidelities:
            continuations = False

        problem = Problem(
            optimizer=optimizer,
            optimizer_hyperparameters=optimizer_hyperparameters,
            benchmark=benchmark,
            budget=_budget,
            fidelities=_fid,
            objectives=_obj,
            costs=_cost,
            precision=precision if precision is not None else PRECISION,
            continuations=continuations,
            priors=priors,
        )

        support: Problem.Support = _opt.support
        support.check_opt_support(who=_opt.name, problem=problem)

        return problem

    def get_objectives(self) -> str | list[str]:
        """Retrieve the objectives of the problem.

        Returns:
            objectives of the problem
        """
        return (
            list(self.objectives.keys())
            if isinstance(self.objectives, Mapping)
            else self.objectives[0]
        )

    def get_fidelities(self) -> str | list[str] | None:
        """Retrieve the fidelities associated with the object.

        Returns:
            fidelities of the problem.
        """
        return (
            None
            if self.fidelities is None
            else list(self.fidelities.keys())
            if isinstance(self.fidelities, Mapping)
            else self.fidelities[0]
        )

    def get_costs(self) -> str | list[str] | None:
        """Retrieve the costs associated with the object.

        Returns:
            costs of the problem.
        """
        return (
            None
            if self.costs is None
            else list(self.costs.keys())
            if isinstance(self.costs, Mapping)
            else self.costs[0]
        )

    def group_for_optimizer_comparison(
        self,
    ) -> tuple[
        str,
        BudgetType,
        tuple[tuple[str, Measure], ...],
        None | tuple[tuple[str, Fidelity], ...],
        None | tuple[tuple[str, Measure], ...],
    ]:
        """Groups the objectives, fidelities, and costs for optimizer comparison.

        Returns:
            tuple: A tuple containing:
                - The name of the benchmark.
                - The budget type.
                - The objectives.
                - The fidelities.
                - The costs.
        """
        match self.objectives:
            case (name, measure):
                _obj = ((name, measure),)
            case Mapping():
                _obj = tuple(self.objectives.items())

        match self.fidelities:
            case None:
                _fid = None
            case (name, fid):
                _fid = ((name, fid),)
            case Mapping():
                _fid = tuple(self.fidelities.items())

        match self.costs:
            case None:
                _cost = None
            case (name, measure):
                _cost = ((name, measure),)
            case Mapping():
                _cost = tuple(self.costs.items())

        return (self.benchmark.name, self.budget, _obj, _fid, _cost)

    def to_dict(self) -> dict[str, Any]:
        """Convert the problem instance to a dictionary."""
        return {
            "objectives": self.get_objectives(),
            "fidelities": self.get_fidelities(),
            "costs": self.get_costs(),
            "budget_type": self.budget.name,
            "budget": self.budget.to_dict(),
            "benchmark": self.benchmark.name,
            "optimizer": self.optimizer.name,
            "optimizer_hyperparameters": self.optimizer_hyperparameters,
            "continuations": self.continuations,
            "priors": configpriors_to_dict(self.priors) if self.priors else None,
        }

    @classmethod
    def from_dict(  # noqa: C901, PLR0912
        cls,
        data: dict[str, Any],
        benchmarks_dict: Mapping[str, BenchmarkDescription],
        optimizers_dict: Mapping[str, Optimizer],
    ) -> Problem:
        """Create a Problem instance from a dictionary.

        Args:
            data: A dictionary containing the problem data.
            benchmarks_dict: A mapping of benchmark names to BenchmarkDescription instances.
            optimizers_dict: A mapping of optimizer names to Optimizer instances.

        Returns:
            A Problem instance created from the dictionary data.
        """
        if data["benchmark"] not in benchmarks_dict:
            raise ValueError(
                f"Benchmark {data['benchmark']} not found in benchmarks!"
                " Please make sure your benchmark is registed in `BENCHMARKS`"
                " before loading/parsing."
            )

        if data["optimizer"] not in optimizers_dict:
            raise ValueError(
                f"Optimizer {data['optimizer']} not found in optimizers!"
                " Please make sure your optimizer is registed in `OPTIMIZERS`"
                " before loading/parsing."
            )

        benchmark = benchmarks_dict[data["benchmark"]]
        optimizer = optimizers_dict[data["optimizer"]]
        _obj = data["objectives"]
        match _obj:
            case str():
                objectives = (_obj, benchmark.metrics[_obj])
            case list():
                objectives = {name: benchmark.metrics[name] for name in _obj}
            case _:
                raise ValueError("Objectives must be a string or a list of strings")

        _fid = data["fidelities"]
        match _fid:
            case None:
                fidelities = None
            case str():
                assert benchmark.fidelities is not None
                fidelities = (_fid, benchmark.fidelities[_fid])
            case list():
                assert benchmark.fidelities is not None
                fidelities = {name: benchmark.fidelities[name] for name in _fid}
            case _:
                raise ValueError("Fidelity must be a string or a list of strings")

        _cost = data["costs"]
        match _cost:
            case None:
                costs = None
            case str():
                assert benchmark.costs is not None
                costs = (_cost, benchmark.costs[_cost])
            case list():
                assert benchmark.costs is not None
                costs = {name: benchmark.costs[name] for name in _cost}
            case _:
                raise ValueError("Costs must be a string or a list of strings")

        _budget_type = data["budget_type"]
        match _budget_type:
            case "trial_budget":
                budget = TrialBudget.from_dict(data["budget"])
            case "cost_budget":
                budget = CostBudget.from_dict(data["budget"])
            case _:
                raise ValueError("Budget type must be 'trial_budget' or 'cost_budget'")

        return cls(
            objectives=objectives,
            fidelities=fidelities,
            costs=costs,
            budget=budget,
            benchmark=benchmark,
            optimizer=optimizer,
            optimizer_hyperparameters=data["optimizer_hyperparameters"],
            continuations=data["continuations"],
            priors = dict_to_configpriors(data["priors"]) if data.get("priors") else None,
        )

    @dataclass(kw_only=True)
    class Support:
        """The support of an optimizer for a problem."""

        objectives: tuple[Literal["single", "many"], ...] = field(default=("single",))
        fidelities: tuple[Literal[None, "single", "many"], ...] = field(default=(None,))
        cost_awareness: tuple[Literal[None, "single", "many"], ...] = field(default=(None,))
        tabular: bool = False
        priors: bool = False

        def __post_init__(self) -> None:
            match self.objectives:
                case tuple():
                    pass
                case str():
                    self.objectives = (self.objectives,)
                case _:
                    raise ValueError(
                        "Invalid type for optimizer support objectives: "
                        f"{type(self.objectives)}, expected tuple!"
                    )
            match self.fidelities:
                case tuple():
                    pass
                case str():
                    self.fidelities = (self.fidelities,)
                case None:
                    self.fidelities = (None,)
                case _:
                    raise ValueError(
                        "Invalid type for optimizer support fidelities: "
                        f"{type(self.fidelities)}, expected tuple!"
                    )
            match self.cost_awareness:
                case tuple():
                    pass
                case str():
                    self.cost_awareness = (self.cost_awareness,)
                case None:
                    self.cost_awareness = (None,)
                case _:
                    raise ValueError(
                        "Invalid type for optimizer support cost_awareness: "
                        f"{type(self.cost_awareness)}, expected tuple!"
                    )

        def check_opt_support(self, who: str, *, problem: Problem) -> None:  # noqa: C901
            """Check if the problem is supported by the support."""
            match problem.fidelities:
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

            match problem.objectives:
                case tuple() if "single" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support single-objective for {problem.name}!"
                    )
                case Mapping() if "many" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-objective for {problem.name}!"
                    )

            match problem.costs:
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

            if problem.priors and not self.priors:
                warnings.warn(
                    f"Optimizer {who} does not support priors",
                    stacklevel=2,
                )
                problem.priors = None
                problem.name = problem.name.split(".priors=")[0]
