from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from hpoglue.result import Result

if TYPE_CHECKING:
    from hpoglue.config import Config


@dataclass(kw_only=True)
class Query:
    """A query to a benchmark."""

    config: Config
    """ The config to evaluate """

    fidelity: tuple[str, int | float] | Mapping[str, int | float] | None = None
    """What fidelity to evaluate at."""

    optimizer_info: Any | None = None
    """Any optimizer specific info required across ask and tell"""

    request_trajectory: bool | tuple[int, int] | tuple[float | float] = False
    """Whether the optimizer requires a trajectory curve for multi-fidelity optimization.

    If a specific range is requested, then a tuple can be provided.
    """

    query_id: str = field(init=False)
    """The id of the query.

    This includes information about the config id and fidelity.
    """

    def __post_init__(self) -> None:
        match self.fidelity:
            case None:
                self.query_id = self.config.config_id
            case (name, value):
                self.query_id = f"{self.config.config_id}-{name}={value}"
            case Mapping():
                self.query_id = (
                    f"{self.config.config_id}-"
                    f"{'-'.join(f'{k}={v}' for k, v in self.fidelity.items())}"
                )
            case _:
                raise NotImplementedError("Unexpected fidelity type")

        if self.request_trajectory:
            match self.fidelity:
                case None:
                    raise ValueError("Learning curve requested but no fidelity provided")
                case tuple():
                    pass
                case Mapping():
                    raise ValueError(
                        "Learning curve requested but more than a single fidelity provided"
                    )

    @property
    def config_id(self) -> str:
        """The id of the config."""
        return self.config.config_id

    def with_fidelity(
        self,
        fidelity: tuple[str, int | float] | Mapping[str, int | float] | None,
    ) -> Query:
        """Create a new query with a different fidelity."""
        return replace(self, fidelity=fidelity)

    def make_result(self, results: dict[str, Any]) -> Result:
        """Create a result from the query."""
        return Result(
            query=self,
            fidelity=self.fidelity,
            values=results,
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "config": self.config.values,
            "fidelity": self.fidelity,
        }
