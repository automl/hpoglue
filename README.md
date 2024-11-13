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
    optimizer = ...,
    benchmark = ...,
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
from ConfigSpace import ConfigurationSpace
from hpoglue.config import Config
from hpoglue.optimizer import Optimizer
from hpoglue.problem import Problem
from hpoglue.query import Query


class RandomSearch(Optimizer):
    name = "RandomSearch"
    support = Problem.Support()
    def __init__(self, problem, seed, working_directory, config_space):
        self.config_space = config_space
        self.config_space.seed(seed)
        self.problem = problem
        self._counter = 0

    def ask(self):
        self._counter += 1
        config = Config(
            config_id=str(self._counter),
            values=self.config_space.sample_configuration().get_dictionary(),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result):
        return
```

## Example Benchmark Definition

```python
import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from hpoglue.benchmark import FunctionalBenchmark
from hpoglue.measure import Measure
from hpoglue.result import Result

def ackley_bench():
    ackley_space = ConfigurationSpace()
    for i in range(2):
        ackley_space.add(Float(name=f"x{i}", bounds=[-32.768, 32.768]))
    return FunctionalBenchmark(
        name="ackley",
        config_space=ackley_space,
        metrics={"value": Measure.metric((0.0, np.inf), minimize=True)},
        query=ackley,
    )

def ackley(query):
    x = np.array(query.config.to_tuple())
    n_var=2
    a=20
    b=1/5
    c=2 * np.pi
    part1 = -1. * a * np.exp(-1. * b * np.sqrt((1. / n_var) * np.sum(x * x)))
    part2 = -1. * np.exp((1. / n_var) * np.sum(np.cos(c * x)))
    out = part1 + part2 + a + np.exp(1)
    return Result(
        query=query,
        fidelity=None,
        values={"value": out},
    )
```
