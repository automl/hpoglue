from __future__ import annotations

import logging
import subprocess
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hpoglue.utils import env_pkg_version_compat

logger = logging.getLogger(__name__)

WHEN_MODULE_INITIALIZED = datetime.now().isoformat()

GLUE_REPO = "github.com/automl/hpoglue.git"
GLUE_PYPI = "hpoglue"
GLUE_GIT_SSH_INSTALL = "git+ssh://git@github.com/automl/hpoglue.git"


@dataclass
class Env:
    """Represents an hpoglue environment configuration with a name, Python version,
    requirements, and post-install commands.
    """
    name: str
    """The name of the environment."""

    python_version: str = field(default="3.10", repr=False)
    """The Python version of the environment."""

    requirements: tuple[str, ...] = field(default=(), repr=False)
    """Environment requirements."""

    post_install: tuple[str, ...] = field(default=(), repr=False)
    """Post-install commands."""

    def __post_init__(self) -> None:
        match self.requirements:
            case tuple():
                pass
            case str():
                self.requirements = (self.requirements,)
            case None:
                self.requirements = ()
            case _:
                raise ValueError(
                    f"Invalid requirements type: {type(self.requirements)}, expected tuple!"
                )
        match self.post_install:
            case tuple():
                _cmd = []
                for command in self.post_install:
                    command = command.replace("python", "").replace("-m", "").strip()  # noqa: PLW2901
                    if command:
                        _cmd.append(command)
                self.post_install = tuple(_cmd)
            case str():
                self.post_install = (
                    self.post_install.replace("python", "")
                    .replace("-m", "")
                    .strip()
                )
                self.post_install = (self.post_install,)
            case None:
                self.post_install = ()
            case _:
                raise ValueError(
                    f"Invalid post_install type: {type(self.post_install)}, expected tuple!"
                )

    @classmethod
    def empty(cls) -> Env:
        """Create an empty environment instance."""
        return cls(name="empty")

    @classmethod
    def merge(cls, one: Env, two: Env) -> Env:
        """Merges two Env instances into one.

        If the python versions of the two environments differ, the lowest version is chosen
        and a warning is issued. The resulting environment will have a combination of the
        requirements and post_install scripts from both environments.

        Args:
            one: The first environment to merge.
            two: The second environment to merge.

        Returns:
            A new environment instance that is the result of merging the two input environments.
        """
        if one.python_version != two.python_version:
            this = tuple(map(int, one.python_version.split(".")))
            that = tuple(map(int, two.python_version.split(".")))
            resolved = min(this, that)
            warnings.warn(
                f"Different python versions for:"
                f" {this} - {one.name}\n"
                f" {that} - {two.name}\n"
                f"Resolving to the lowest version {resolved}.",
                UserWarning,
                stacklevel=2,
            )
            python_version = ".".join(map(str, resolved))
        else:
            python_version = one.python_version

        for package in one.requirements:
            for other_package in two.requirements:
                if not env_pkg_version_compat(package, other_package):
                    raise ValueError(
                        f"Package versions are not compatible: {package}, {other_package}"
                    )


        if one.name == two.name:
            return Env(
                name=one.name,
                python_version=python_version,
                requirements=tuple(set(one.requirements + two.requirements)),
                post_install=one.post_install + two.post_install,
            )

        return Env(
            name=one.name + "_" + two.name,
            python_version=python_version,
            requirements=tuple(set(one.requirements + two.requirements)),
            post_install=one.post_install + two.post_install,
        )

    @property
    def identifier(self) -> str:
        """Returns the name of the current environment."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Converts the environment configuration to a dictionary."""
        return {
            "name": self.name,
            "python_version": self.python_version,
            "requirements": list(self.requirements),
            "post_install": list(self.post_install),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Env:
        """Creates an environment configuration from a dictionary."""
        return cls(
            name=data["name"],
            python_version=data["python_version"],
            requirements=tuple(data["requirements"]),
            post_install=tuple(data["post_install"]),
        )


def _get_current_installed_python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_current_installed_hpoglue_version() -> str:
    """Retrieve the currently installed version of hpoglue."""
    cmd = ["pip", "show", "hpoglue"]
    logger.debug(cmd)
    output = subprocess.run(  # noqa: S603
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = output.stdout.strip().splitlines()
    for line in lines:
        if "Version: " in line:
            return line.split(": ")[1]

    raise RuntimeError(f"Could not find hpoglue version in {lines}.")


@dataclass
class Venv:
    """Represents a Venv configuration with a path, activate, python, and pip commands."""
    def __init__(self, path: Path) -> None:
        """Initialize the environment with the given virtual environment path."""
        self.venv_path = path
        self.activate = f"{self.venv_path / 'bin' / 'activate'}"
        self.python = f"{self.venv_path / 'bin' / 'python'}"
        self.pip = f"{self.venv_path / 'bin' / 'pip'}"

    def create(
        self,
        path: Path,
        *,
        python_version: str,
        requirements_file: Path,
        exists_ok: bool = False,
    ) -> None:
        """Create a virtual environment at the specified path."""
        if self.venv_path.exists() and not exists_ok:
            raise FileExistsError(f"Virtual environment already exists at {self.venv_path}")

        if self.venv_path.exists():
            return

        current_python_version = _get_current_installed_python_version()
        if current_python_version != python_version:
            raise RuntimeError(
                f"Python version between current env and required env:"
                f" {current_python_version} != {python_version}."
                " Please use some other method of creating virtual environments"
                " that support python versions, for example `'conda'`."
            )

        path = self.venv_path.resolve().absolute()
        cmd = ["python", "-m", "venv", str(path)]
        logger.debug(cmd)
        subprocess.run(cmd, check=True)  # noqa: S603

        if requirements_file.exists():
            cmd = [self.pip, "install", "-r", str(requirements_file)]
            logger.debug(cmd)
            subprocess.run(cmd, check=True)  # noqa: S603

    def run(self, cmd: Sequence[str]) -> None:
        """Run the given command in the virtual environment."""
        cmd = [self.python, "-m", *cmd]
        logger.debug(cmd)
        subprocess.run(cmd, check=True)  # noqa: S603
