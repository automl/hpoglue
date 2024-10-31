from __future__ import annotations

import logging
import warnings
from collections.abc import Container
from typing import TypeAlias, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DataContainer: TypeAlias = np.ndarray | (pd.DataFrame | pd.Series)
D = TypeVar("D", bound=DataContainer)


def _inc_trace(
    df: pd.DataFrame,
    *,
    x_start_col: str,
    x_col: str,
    y_col: str,
    minimize: bool,
    test_y_col: str,
) -> pd.Series:
    # We now have each individual std group to plot, i.e. the fold
    _start = df[x_start_col].min()
    _x = (df[x_col] - _start).dt.total_seconds()

    ind = pd.Index(_x, name="time (s)")
    _df = (
        df[[y_col, test_y_col]]
        .rename(columns={y_col: "y", test_y_col: "test_y"})
        .set_index(ind)
        .sort_index()
        .dropna()
    )

    # Transform everything
    match minimize:
        case True:
            _df["cumulative"] = _df["y"].cummin()
        case False:
            _df["cumulative"] = _df["y"].cummax()

    _df = _df.drop_duplicates(subset="cumulative", keep="first").drop(
        columns="cumulative",
    )

    return pd.Series(
        # We do this concat operation so that data is contiguous for later opertions
        data=np.concatenate(
            [_df["y"].to_numpy(), _df["test_y"].to_numpy()],
        ),
        index=pd.MultiIndex.from_product([["val", "test"], _df.index]),
    )


def reduce_floating_precision(x: D) -> D:
    """Reduce the floating point precision of the data.

    For a float array, will reduce by one step, i.e. float32 -> float16, float64
    -> float32.

    Args:
        x: The data to reduce.

    Returns:
        The reduced data.
    """
    # For a dataframe, we recurse over all columns
    if isinstance(x, pd.DataFrame):
        # Using `apply` doesn't work
        for col in x.columns:
            x[col] = reduce_floating_precision(x[col])
        return x  # type: ignore

    if x.dtype.kind != "f":
        return x

    _reduction_map = {
        # Base numpy dtypes
        "float128": "float64",
        "float96": "float64",
        "float64": "float32",
        "float32": "float16",
        # Nullable pandas dtypes (only supports 64 and 32 bit)
        "Float64": "Float32",
    }

    if (dtype := _reduction_map.get(x.dtype.name)) is not None:
        return x.astype(dtype)  # type: ignore

    return x


def reduce_int_span(x: D) -> D:
    """Reduce the integer span of the data.

    For an int array, will reduce to the smallest dtype that can hold the
    minimum and maximum values of the array.

    Args:
        x: The data to reduce.

    Returns:
        The reduced data.
    """
    # For a dataframe, we recurse over all columns
    if isinstance(x, pd.DataFrame):
        # Using `apply` doesn't work
        for col in x.columns:
            x[col] = reduce_int_span(x[col])
        return x  # type: ignore

    if x.dtype.kind not in "iu":
        return x

    min_dtype = np.min_scalar_type(x.min())  # type: ignore
    max_dtype = np.min_scalar_type(x.max())  # type: ignore
    dtype = np.result_type(min_dtype, max_dtype)

    # The above dtype is a numpy dtype and may not allow for nullable values,
    # which are permissible in pandas. `to_numeric` will convert to appropriate
    # pandas nullable dtypes.
    if isinstance(x, pd.Series):
        dc = "unsigned" if "uint" in dtype.name else "integer"
        return pd.to_numeric(x, downcast=dc)

    return x.astype(dtype)


def reduce_dtypes(
    x: D,
    *,
    reduce_int: bool = True,
    reduce_float: bool = True,
    categories: bool = True,
    categories_exclude: Container[str] | None = None,
    ignore_convert_dtypes_warning: bool = True,
) -> D:
    """Reduce the dtypes of data.

    When a dataframe, will reduce the dtypes of all columns.
    When applied to an iterable, will apply to all elements of the iterable.

    For an int array, will reduce to the smallest dtype that can hold the
    minimum and maximum values of the array. Otherwise for floats, will reduce
    by one step, i.e. float32 -> float16, float64 -> float32.

    Args:
        x: The data to reduce.
        reduce_int: Whether to reduce integer dtypes.
        reduce_float: Whether to reduce floating point dtypes.
        categories: Whether to convert string/object columns to categories.
        categories_exclude: Columns to exclude from conversion to categories if they are string/object.
        ignore_convert_dtypes_warning: Whether to ignore the warning when converting dtypes.
    """
    if not isinstance(x, pd.DataFrame | pd.Series | np.ndarray):
        raise TypeError(f"Cannot reduce data of type {type(x)}.")

    if isinstance(x, pd.Series | pd.DataFrame):
        if ignore_convert_dtypes_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                x = x.convert_dtypes()
        else:
            x = x.convert_dtypes()

    if reduce_int:
        x = reduce_int_span(x)
    if reduce_float:
        x = reduce_floating_precision(x)

    if categories is True and isinstance(x, pd.DataFrame):
        category_exclude = categories_exclude or ()
        cat_cols = [
            c
            for c in x.select_dtypes(include=["string", object]).columns
            if c not in category_exclude
        ]
        x[cat_cols] = x[cat_cols].astype("category")

    return x
