from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias


@dataclass(frozen=True)
class TrialBudget:
    """A budget for the number of trials to run."""

    name: ClassVar[str] = "trial_budget"

    total: int
    """Total amount of budget allowed for the optimizer for this problem.

    How this is interpreted is depending on fidelity type.

    If the problem **does not** include a fidelity, then this is assumed
    to be a black-box problem, and each fully complete trial counts as
    1 towards the budget.

    If the problem **does** include a **single** fidelity, then the fidelity
    at which the trial was evaluated is taken as a fraction of the full fidelity
    and added to the used budget. For example, 40 epochs of a fidelity that
    maxes out at 100 epochs would count as 0.4 towards the budget.

    If the problem **does** include **many** fidelities, then the fraction as calculated
    for a single fidelity is applied to all fidelities, and then summed, normalized by
    the total number of fidelities. For example, 40 epochs of a fidelity that maxes out
    at 100 epochs and data percentage of 0.6 of a fidelity that maxes out at 1.0 would
    equate to (0.4 + 0.6) / 2 = 0.5 towards the budget.
    """

    minimum_fidelity_normalized_value: float = field(default=0.01, repr=False)

    @property
    def path_str(self) -> str:
        """Return a string representation of the budget."""
        clsname = self.__class__.__name__
        return f"{clsname}={self.total}"

    def to_dict(self) -> dict[str, int]:
        """Convert the budget to a dictionary."""
        return {"total": self.total}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> TrialBudget:
        """Convert a dictionary to a budget."""
        return cls(total=data["total"])


@dataclass(frozen=True)
class CostBudget:
    """A budget for the cost of the trials to run."""

    name: ClassVar[str] = "cost_budget"

    total: int | float

    def __post_init__(self):
        raise NotImplementedError("Cost budgets not yet supported")

    def to_dict(self) -> dict[str, int | float]:
        """Convert the budget to a dictionary."""
        return {"total": self.total}

    @classmethod
    def from_dict(cls, data: dict[str, int | float]) -> CostBudget:
        """Convert a dictionary to a budget."""
        return cls(total=data["total"])

    @property
    def path_str(self) -> str:
        """Return a string representation of the budget."""
        clsname = self.__class__.__name__
        return f"{clsname}={self.total}"


BudgetType: TypeAlias = TrialBudget | CostBudget
