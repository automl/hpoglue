[![image](https://img.shields.io/pypi/v/hpoglue.svg)](https://pypi.python.org/pypi/hpoglue)
[![image](https://img.shields.io/pypi/l/hpoglue.svg)](https://pypi.python.org/pypi/hpoglue)
[![image](https://img.shields.io/pypi/pyversions/hpoglue.svg)](https://pypi.python.org/pypi/hpoglue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# hpoglue
HPO tool with a modular API that allows for the easy interfacing of a new Optimizer and a new Benchmark

## Minimal Example to run hpoglue

```python
from hpoglue.run_glue import run_glue
df = run_glue(
    run_name="hpoglue_ex",
    optimizer = ...,    # type[hpoglue.Optimizer]
    benchmark = ...,    # type[hpoglue.BenchmarkDescription] | type[hpoglue.FunctionalBenchmark]
    seed = 1,
    budget = 50
)
```

> [!TIP]
> * See below for examples of an [Optimizer](#example-optimizer-definition) and [Benchmark](#example-benchmark-definition)
> * Check this example [notebook](examples/glue_demo.ipynb) for more
> * Check out [hposuite](https://github.com/automl/hposuite) for some already implemented Optimizers and Benchmarks for hpoglue

## Installation

### Create a Virtual Environment using Venv
```bash
python -m venv hpoglue_env
source hpoglue_env/bin/activate
```
### Installing from PyPI

```bash
pip install hpoglue
```

> [!TIP]
> * `pip install hpoglue["notebook"]` - For usage in a notebook

### Installation from source

```bash
git clone https://github.com/automl/hpoglue.git
cd hpoglue

pip install -e . # -e for editable install
```


## Example Optimizer Definition

```python
from pathlib import Path
from hpoglue import Config, Optimizer, Problem, Query, Result


class RandomSearch(Optimizer):
    name = "RandomSearch"
    support = Problem.Support()
    def __init__(
        self,
        problem: Problem,
        working_directory: str | Path,
        seed: int | None = None,
    ):
        """Args:
        problem: Source of task information.
        working_directory: Directory to save the optimizer's state.
        seed: seed for random number generator.
        """
        self.config_space = problem.config_space
        self.config_space.seed(seed)
        self.problem = problem
        self._optmizer_unique_id = 0

    def ask(self) -> Query:
        self._optmizer_unique_id += 1
        config = Config(
            config_id=str(self._optmizer_unique_id),
            values=dict(self.config_space.sample_configuration()),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result: Result) -> None:
        # Update the optimizer (not needed for RandomSearch)
        return
```

## Example Benchmark Definition

```python
import numpy as np
from ConfigSpace import ConfigurationSpace
from hpoglue import FunctionalBenchmark, Measure, Result, Query


def ackley_fn(x1: float, x2: float) -> float:
    x = np.array([x1, x2])
    n_var=len(x)
    a=20
    b=1/5
    c=2 * np.pi
    part1 = -1. * a * np.exp(-1. * b * np.sqrt((1. / n_var) * np.sum(x * x)))
    part2 = -1. * np.exp((1. / n_var) * np.sum(np.cos(c * x)))
    out = part1 + part2 + a + np.exp(1)
    return out

def wrapped_ackley(query: Query) -> Result:
    y = ackley_fn(x1=query.config.values["x1"], x2=query.config.values["x2"])
    return Result(query=query, fidelity=None, values={"y": y})

ACKLEY_BENCH = FunctionalBenchmark(
    name="ackley",
    config_space=ConfigurationSpace({"x1": (-32.768, 32.768), "x2": (-32.768, 32.768)}),
    metrics={"y": Measure.metric((0.0, np.inf), minimize=True)},
    query=wrapped_ackley,
)
```

## Run hpoglue on the examples

```python
from hpoglue.run_glue import run_glue
df = run_glue(
    run_name="hpoglue_demo",
    optimizer = RandomSearch,
    benchmark = ACKLEY_BENCH,
    seed = 1,
    budget = 50
)
```

## Using Environments from `hpoglue.env.Env`

Environments for Optimizers and Benchmarks can be defined in the following way:

```python
from hpoglue.env import Env
environ = Env(
    name="Dummy_env",
    python_version="3.10"
    requirements=("numpy==1.24.4", "scikit-learn==1.5.2")
    post_install=("mybenchmark --download_data mydataset --datadir /home/someuser/data/")
    # Only python CLI commands are supported. We automatically append the python -m with the
    # appropriate python executable from the environment's absolute path
)
```
