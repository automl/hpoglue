from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import pandas as pd

from hpoglue.config import Config
from hpoglue.env import Env
from hpoglue.optimizer import Optimizer
from hpoglue.result import Result

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from hpoglue.fidelity import Fidelity
    from hpoglue.measure import Measure
    from hpoglue.query import Query

    class TrajectoryF(Protocol):  # noqa: D101
        def __call__(  # noqa: D102
            self,
            *,
            query: Query,
            frm: int | float | None = None,
            to: int | float | None = None,
        ) -> pd.DataFrame: ...


logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


@dataclass(kw_only=True, frozen=True)
class BenchmarkDescription:
    """Describes a benchmark without loading it all in."""

    name: str
    """Unique name of the benchmark."""

    config_space: ConfigurationSpace | list[Config]
    """The configuration space for the benchmark."""

    load: Callable[[BenchmarkDescription], Benchmark] = field(compare=False)
    """Function to load the benchmark."""

    metrics: Mapping[str, Measure]
    """All the metrics that the benchmark supports."""

    test_metrics: Mapping[str, Measure] | None = None
    """All the test metrics that the benchmark supports."""

    costs: Mapping[str, Measure] | None = None
    """All the costs that the benchmark supports."""

    fidelities: Mapping[str, Fidelity] | None = None
    """All the fidelities that the benchmark supports."""

    has_conditionals: bool = False
    """Whether the benchmark has conditionals."""

    is_tabular: bool = False
    """Whether the benchmark is tabular."""

    env: Env = field(default_factory=Env.empty)
    """The environment needed to run this benchmark."""

    mem_req_mb: int = 1024
    """The memory requirement of the benchmark in mb."""

    predefined_points: Mapping[str, Config] | None = None
    """
    Predefined points for the benchmark with their names and descriptions.

    Example:
    ``` python
    {
        "optimum": Config(
                        config_id="optimum",
                        description="This point yields the global optimum",
                        values={"x": 0.0, "y": 0.0}
                    ),
        "worst": Config(
                        config_id="worst",
                        description="This point yields the analytical worst value",
                        values={"x": 1.0, "y": 1.0}
                    ),
    }
    ```
    """

    extra: Mapping[str, Any] = field(default_factory=dict)
    """Extra information about the benchmark."""


@dataclass(kw_only=True)
class SurrogateBenchmark:
    """Defines the interface for a surrogate benchmark."""

    name: str = field(init=False)
    """The name of the Surrogate benchmark."""

    desc: BenchmarkDescription
    """The description of the benchmark."""

    config_space: ConfigurationSpace
    """ The configuration space for the benchmark """

    benchmark: Any
    """The wrapped benchmark object."""

    query: Callable[[Query], Result]
    """The query function for the benchmark."""

    trajectory_f: TrajectoryF | None = None
    """The trajectory function for the benchmark, if one exists.

    This function should return a DataFrame with the trajectory of the query up
    to the given fidelity. The index should be the fidelity parameter with the
    columns as the values.

    ```
    def __call__(
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        ...
    ```

    If not provided, the query will be called repeatedly to generate this.
    """

    def __post_init__(self) -> None:
        self.name = self.desc.name

    def trajectory(  # noqa: D102
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        if self.trajectory_f is not None:
            return self.trajectory_f(query=query, frm=frm, to=to)

        assert isinstance(query.fidelity, tuple)
        assert self.desc.fidelities is not None

        fid_name, fid_value = query.fidelity
        fid = self.desc.fidelities[fid_name]
        frm = frm if frm is not None else fid.min
        to = to if to is not None else fid_value

        index: list[int] | list[float] = []
        results: list[Result] = []
        for val in iter(fid):
            if val < frm:
                continue

            if val > to:
                break

            index.append(val)
            result = self.query(query.with_fidelity((fid_name, val)))
            results.append(result)

        # Return in trajectory format
        # fid_name    **results
        # 0         | . | . | ...
        # 1         | . | . | ...
        # ...
        return pd.DataFrame.from_records(
            [result.values for result in results],
            index=pd.Index(index, name=fid_name),
        )


class TabularBenchmark:
    """Defines the interface for a tabular benchmark."""

    name: str
    """The name of the Tabular benchmark."""

    desc: BenchmarkDescription
    """The description of the benchmark."""

    table: pd.DataFrame
    """ The table holding all information """

    id_key: str
    """The key in the table that we want to use as the id."""

    config_space: list[Config]
    """ All possible configs for the benchmark """

    config_keys: list[str]
    """The keys in the table to use as the config keys."""

    result_keys: list[str]
    """The keys in the table to use as the result keys.

    This is inferred from the `desc=`.
    """

    def __init__(
        self,
        *,
        desc: BenchmarkDescription,
        table: pd.DataFrame,
        id_key: str,
        config_keys: list[str],
    ) -> None:
        """Create a tabular benchmark.

        The result and fidelity keys will be inferred from the `desc=`.

        Args:
            desc: The description of the benchmark.
            table: The table holding all information.
            id_key: The key in the table that we want to use as the id.
            config_keys: The keys in the table that we want to use as the config.
        """
        self.name = desc.name
        # Make sure we work with a clean slate, no issue with index.
        table = table.reset_index()

        for key in config_keys:
            if key not in table.columns:
                raise KeyError(
                    f"Config key '{key}' not in columns {table.columns}."
                    "This is most likely from a misspecified BecnhmarkDescription for "
                    f"{desc.name}.",
                )

        result_keys = [
            *desc.metrics.keys(),
            *(desc.test_metrics.keys() if desc.test_metrics else []),
            *(desc.costs.keys() if desc.costs else []),
        ]
        for key in result_keys:
            if key not in table.columns:
                raise KeyError(
                    f"Result key '{key}' not in columns {table.columns}."
                    "This is most likely from a misspecified BecnhmarkDescription for "
                    f"{desc.name}.",
                )

        match desc.fidelities:
            case None:
                fidelity_keys = None
            case Mapping():
                fidelity_keys = list(desc.fidelities.keys())
                for key in fidelity_keys:
                    if key not in table.columns:
                        raise KeyError(
                            f"Fidelity key '{key}' not in columns {table.columns}."
                            "This is most likely from a misspecified BecnhmarkDescription for "
                            f"{desc.name}.",
                        )
            case _:
                raise TypeError(f"{desc.fidelities=} not supported")

        # Make sure that the column `id` only exist if it's the `id_key`
        if "id" in table.columns and id_key != "id":
            raise ValueError(
                f"Can't have `id` in the columns if it's not the {id_key=}."
                " Please drop it or rename it.",
            )

        table[id_key] = table[id_key].astype(str)   # enforcing str for id

        # Remap their id column to `id`
        table = table.rename(columns={id_key: "id"})

        # We will create a multi-index for the table, done by the if and
        # the remaining fidelity keys
        _fid_cols = [] if fidelity_keys is None else fidelity_keys

        # Drop all the columns that are not relevant
        relevant_cols: list[str] = ["id", *_fid_cols, *result_keys, *config_keys]
        table = table[relevant_cols]  # type: ignore
        table = table.set_index(["id", *_fid_cols]).sort_index()

        # We now have the following table
        #
        #     id    fidelity | **metrics, **config_values
        #     0         0    |
        #               1    |
        #               2    |
        #     1         0    |
        #               1    |
        #               2    |
        #   ...
        self.table = table
        self.id_key = id_key
        self.desc = desc
        self.config_keys = config_keys
        self.result_keys = result_keys
        self.fidelity_keys = fidelity_keys
        self.config_space = self.get_tabular_config_space(table, config_keys)


    @classmethod
    def get_tabular_config_space(
        cls,
        table: pd.DataFrame,
        config_keys: list[str],
    ) -> list[Config]:
        """Get the configuration space from the table."""
        return [
            Config(config_id=str(i), values=config)  # enforcing str for id
            for i, config in enumerate(
                table[config_keys]
                .drop_duplicates()
                .sort_values(by=config_keys)  # Sorting to ensure table config order is consistent
                .to_dict(orient="records"),
            )
        ]


    def query(self, query: Query) -> Result:
        """Query the benchmark for a result."""
        # NOTE(eddiebergman):
        # Some annoying logic here to basically be able to handle partially specified fidelties,
        # even if it does not match the order of what the table has. In the case where a fidelity
        # is not specified, we select ALL (slice(None)) for that fidelity. Later, we will just take
        # the last row then. Important that we go in the order of `self.fidelity_keys`
        ALL = slice(None)
        fidelity_order = self.table.index.names[1:]

        match query.fidelity:
            case None:
                slices = {col: ALL for col in fidelity_order}
            case (key, value):
                assert self.fidelity_keys is not None
                slices = {col: (value if key == col else ALL) for col in fidelity_order}
            case Mapping():
                assert self.fidelity_keys is not None
                slices = {col: query.fidelity.get(col, ALL) for col in fidelity_order}
            case _:
                raise TypeError(f"type of {query.fidelity=} ({type(query.fidelity)}) supported")

        result = self.table.loc[(query.config_id, *slices.values())]
        row: pd.Series
        match result:
            case pd.Series():
                # If it's a series, a row was uniquely specified, meaning that all
                # of the fidelity values were specified.
                retrieved_results = result[self.result_keys]
                assert isinstance(retrieved_results, pd.Series)
                unspecified_fids = {}
            case pd.DataFrame():
                # If it's a DataFrame, we have multiple rows, we take the last one,
                # under the assumption that:
                # 1. Larger fidelity values are what is requested.
                # 2. The table is sorted by fidelity values.
                retrieved_results = result[self.result_keys]

                # Get the non-specified fidelity values
                # We have to keep it as a dataframe using `[-1:]`
                # for the moment so we can get the correct fidelity names and values.
                row = result.iloc[-1:]
                assert isinstance(row, pd.DataFrame)
                assert len(row.index) == 1
                unspecified_fids = dict(zip(row.index.names, row.index, strict=True))

                retrieved_results = retrieved_results.iloc[-1]
            case _:
                raise TypeError(f"type of {result=} ({type(result)}) not supported")

        match query.fidelity:
            case None:
                fidelities_retrieved = None
            case (key, value):
                fidelities_retrieved = (key, value)
            case Mapping():
                fidelities_retrieved = {**unspecified_fids, **query.fidelity}

        return Result(
            query=query,
            values=retrieved_results.to_dict(),
            fidelity=fidelities_retrieved,
        )

    def trajectory(
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        """Query the benchmark for a result."""
        assert isinstance(query.fidelity, tuple)
        fid_name, fid_value = query.fidelity
        if self.fidelity_keys is None:
            raise ValueError("No fidelities to query for this benchmark!")

        if self.fidelity_keys != [fid_name]:
            raise NotImplementedError(
                f"Can't get trajectory for {fid_name=} when more than one"
                f" fidelity {self.fidelity_keys=}!",
            )

        assert self.desc.fidelities is not None
        frm = frm if frm is not None else self.desc.fidelities[fid_name].min
        to = to if to is not None else fid_value

        # Return in trajectory format
        # fid_name    **results
        # 0         | . | . | ...
        # 1         | . | . | ...
        # ...
        return self.table[self.result_keys].loc[query.config_id, frm:to].droplevel(0).sort_index()


class FunctionalBenchmark:
    """Defines the interface for a functional benchmark."""

    name: str = field(init=False)
    """The name of the Functional benchmark."""

    desc: BenchmarkDescription
    """The description of the functional benchmark."""

    query: Callable[[Query], Result]
    """The query function for the benchmark."""

    config_space: ConfigurationSpace | list[Config] | None = None
    """The configuration space for the benchmark."""

    trajectory_f: TrajectoryF | None = None
    """The trajectory function for the benchmark, if one exists.

    This function should return a DataFrame with the trajectory of the query up
    to the given fidelity. The index should be the fidelity parameter with the
    columns as the values.

    ```
    def __call__(
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        ...
    ```

    If not provided, the query will be called repeatedly to generate this.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        metrics: Mapping[str, Measure],
        query: Callable[[Query], Result],
        fidelities: Mapping[str, Fidelity] | None = None,
        costs: Mapping[str, Measure] | None = None,
        test_metrics: Mapping[str, Measure] | None = None,
        config_space: ConfigurationSpace | list[Config] | None = None,
        env: Env | None = None,
        mem_req_mb: int = 1024,
        predefined_points: Mapping[str, tuple[Config, str]] | None = None,
        extra: Mapping[str, Any] = {},
    ):
        """Create a functional benchmark.

        Args:
            name: The name of the benchmark.

            metrics: The metrics that the benchmark supports.

            query: The query function for the benchmark.

            fidelities: The fidelities that the benchmark supports.

            costs: The costs that the benchmark supports.

            test_metrics: The test metrics that the benchmark supports.

            config_space: The configuration space for the benchmark.

            env: The environment needed to run this benchmark.

            mem_req_mb: The memory requirement of the benchmark in mb.

            predefined_points: Predefined points for the benchmark with their names and
                                descriptions.

            extra: Extra information about the benchmark.

        """
        self.name = name
        self.query = query
        self.config_space = config_space
        self.desc = BenchmarkDescription(
            name=name,
            config_space=config_space,
            load=partial(self.load),
            metrics=metrics,
            test_metrics=test_metrics,
            costs=costs,
            fidelities=fidelities,
            has_conditionals=False,
            is_tabular=False,
            env=env,
            mem_req_mb=mem_req_mb,
            predefined_points=predefined_points,
            extra=extra,
        )


    def load(
        self,
        desc: BenchmarkDescription,  # noqa: ARG002
    ) -> FunctionalBenchmark:
        """Load the FunctionalBenchmark."""
        return self


    def trajectory(
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        """The trajectory function for the benchmark, if one exists."""
        if self.trajectory_f is not None:
            return self.trajectory_f(query=query, frm=frm, to=to)

        raise NotImplementedError("Trajectory not implemented for this benchmark.")


# NOTE(eddiebergman): Not using a base class as we really don't expect to need
# more than just these two types of benchmarks.
Benchmark: TypeAlias = TabularBenchmark | SurrogateBenchmark | FunctionalBenchmark
