from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from more_itertools import roundrobin, take
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from hpoglue.config import Config

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


T = TypeVar("T")

def first(_d: Mapping[str, T]) -> tuple[str, T]:
    """Return the first key-value pair from a dictionary.
        Used for retrieving the first objectives/fidelities/costs of a benchmark.

    Args:
        _d: The dictionary from which to retrieve the first item.

    Returns:
        A tuple containing the first key and its corresponding value from the dictionary.

    Raises:
        StopIteration: If the dictionary is empty.
    """
    return next(iter(_d.items()))


def first_n(n: int, _d: Mapping[str, T]) -> dict[str, T]:
    """Return the first `n` items from the given dictionary.
        Used for retrieving the first `n` objectives/fidelities/costs of a benchmark.

    Args:
        n (int): The number of items to return.
        _d (Mapping[str, T]): The dictionary from which to take the items.

    Returns:
        dict[str, T]: A dictionary containing the first `n` items from `_d`.
    """
    return dict(take(n, _d.items()))


def mix_n(n: int, _d1: Mapping[str, T], _d2: Mapping[str, T]) -> dict[str, T]:
    """Mixes items from two dictionaries in a round-robin fashion up to a specified number of items.
        Used for mixing objectives/costs of benchmark in case of multi-objective optimization.

    Args:
        n (int): The number of items to take from the combined dictionaries.
        _d1 (Mapping[str, T]): The first dictionary to mix.
        _d2 (Mapping[str, T]): The second dictionary to mix.

    Returns:
        dict[str, T]: A dictionary containing up to `n` items from the combined dictionaries.
    """
    return dict(take(n, roundrobin(_d1.items(), _d2.items())))


def configpriors_to_dict(
        priors: tuple[str, Mapping[str, Config]]
    ) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    """Converts a tuple of priors to a dictionary.

    Args:
        priors: A tuple with the priors as Config objects..

    Returns:
        A tuple with the priors converted to dictionaries.
    """
    assert isinstance(priors, tuple | list) and len(priors) == 2, (  # noqa: PLR2004, PT018
        "Priors should be a tuple or list of length 2 with the format: "
        "(str, Mapping[str, Config])."
    )
    name, _priors = priors
    prior_dict = {}
    for obj, prior in _priors.items():
        match prior:
            case Config():
                prior_dict[obj] = prior.values
            case dict():
                prior_dict[obj] = prior
            case _:
                raise TypeError(f"Unsupported type for priors: {type(prior)}")

    return name, prior_dict


def dict_to_configpriors(
    priors: tuple[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[str, Mapping[str, Config]]:
    """Converts priors given as dictionaries into Config objects.

    Args:
        priors: The tuple of str and dictionary of priors to convert,
        with the priors themselves as dictionaries.

    Returns:
        A tuple with the priors converted into Config objects.
    """
    assert isinstance(priors, tuple | list) and len(priors) == 2, (  # noqa: PLR2004, PT018
        "Priors should be a tuple or list of length 2 with the format: "
        "(str, Mapping[str, dict[str, Any]])."
    )
    name, _priors = priors
    prior_dict = {}
    for obj, prior in _priors.items():
        match prior:
            case dict():
                prior_dict[obj] = Config(
                    config_id=obj,
                    values=prior,
                )
            case Config():
                prior_dict[obj] = prior
            case _:
                raise TypeError(f"Unsupported type for priors: {type(prior)}")

    return name, prior_dict


def env_pkg_version_compat(
    package1: str,
    package2: str,
):
    """Check if two package versions are compatible.

    Args:
        package1: The first package version.
        package2: The second package version.

    Returns:
        bool: True if the package versions are compatible, False otherwise.
    """
    name1, key1, spec1 = _split_pkg_ver(package1)
    name2, key2, spec2 = _split_pkg_ver(package2)

    if name1 != name2:
        return True

    if not spec1 or not spec2:
        return True

    key_spec1, key_spec2 = SpecifierSet(key1 + spec1), SpecifierSet(key2 + spec2)

    return bool(key_spec1.contains(Version(spec2)) or key_spec2.contains(Version(spec1)))


def _split_pkg_ver(package_name: str) -> tuple[str, str, str]:
    """Split package name into name, operator, and version.

    Args:
        package_name: The package name to split.

    Returns:
        tuple: A tuple containing the package name and specifier set.
    """
    version_constraints = ["==", ">=", "<=", ">", "<"]
    for constraint in version_constraints:
        if constraint in package_name:
            package_name, version_spec = package_name.split(constraint)
            return package_name, constraint, version_spec

    return package_name, "", ""