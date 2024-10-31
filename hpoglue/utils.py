from __future__ import annotations

import logging
from typing import TypeAlias, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DataContainer: TypeAlias = np.ndarray | (pd.DataFrame | pd.Series)
D = TypeVar("D", bound=DataContainer)


def scale(
    unit_xs: int | float | np.number | np.ndarray | pd.Series,
    to: tuple[int | float | np.number, int | float | np.number],
) -> float | np.number | np.ndarray | pd.Series:
    """Scale values from unit range to a new range.

    >>> scale(np.array([0.0, 0.5, 1.0]), to=(0, 10))
    array([ 0.,  5., 10.])

    Parameters
    ----------
    unit_xs:
        The values to scale

    to:
        The new range

    Returns:
    -------
        The scaled values
    """
    return unit_xs * (to[1] - to[0]) + to[0]  # type: ignore


def normalize(
    x: int | float | np.number | np.ndarray | pd.Series,
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
) -> float | np.number | np.ndarray | pd.Series:
    """Normalize values to the unit range.

    >>> normalize(np.array([0.0, 5.0, 10.0]), bounds=(0, 10))
    array([0. , 0.5, 1. ])

    Parameters
    ----------
    x:
        The values to normalize

    bounds:
        The bounds of the range

    Returns:
    -------
        The normalized values
    """
    if bounds == (0, 1):
        return x

    return (x - bounds[0]) / (bounds[1] - bounds[0])  # type: ignore


def rescale(
    x: int | float | np.number | np.ndarray | pd.Series,
    frm: tuple[int | float | np.number, int | float | np.number],
    to: tuple[int | float | np.number, int | float | np.number],
) -> float | np.ndarray | pd.Series:
    """Rescale values from one range to another.

    >>> rescale(np.array([0, 10, 20]), frm=(0, 100), to=(0, 10))
    array([0, 1, 2])

    Parameters
    ----------
    x:
        The values to rescale

    frm:
        The original range

    to:
        The new range

    Returns:
    -------
        The rescaled values
    """
    if frm != to:
        normed = normalize(x, bounds=frm)
        scaled = scale(unit_xs=normed, to=to)
    else:
        scaled = x

    match scaled:
        case int() | float() | np.number():
            return float(scaled)
        case np.ndarray() | pd.Series():
            return scaled.astype(np.float64)
        case _:
            raise ValueError(f"Unsupported type {type(x)}")

