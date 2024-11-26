from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from hpoglue.env import Env

if TYPE_CHECKING:

    from hpoglue.problem import Problem
    from hpoglue.query import Query
    from hpoglue.result import Result


class Optimizer(ABC):
    """Defines the common interface for Optimizers."""

    name: ClassVar[str]
    """The name of the optimizer"""

    support: ClassVar[Problem.Support]
    """What kind of problems the optimizer supports"""

    multi_fideltiy_requires_learning_curve: ClassVar[bool] = False
    """Whether the optimizer requires a learning curve for multi-fidelity optimization.

    If `True` and the problem is multi-fidelity (1 fidelity) and the fidelity
    supports continuations (e.g. epochs), then a trajectory will be provided in
    the `Result` object provided during the `tell` to the optimizer.
    """

    env: ClassVar[Env] = Env.empty()
    """The environment to setup the optimizer in for `isolated` mode.

    If left as `None`, the currently activated environemnt will be used.
    """

    mem_req_mb: ClassVar[int] = 1024
    """The memory requirement of the optimizer in mb."""

    @abstractmethod
    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        **optimizer_kwargs: Any,
    ) -> None:
        """Initialize the optimizer.

        Args:
            problem: The problem to optimize over
            seed: The random seed for the optimizer
            working_directory: The directory to save the optimizer's state
            optimizer_kwargs: Any additional hyperparameters for the optimizer
        """

    @abstractmethod
    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate."""

    @abstractmethod
    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""

