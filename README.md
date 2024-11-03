# hpoglue
HPO tool with a modular API that allows for the easy interfacing of a new Optimizer and a new Benchmark

## Minimal Example

```python
from hpo_glue import run_glue
df = run_glue(
    run_name="hpo_glue_ex"
    optimizer = ...
    benchmark = ...
    seed = 1
    budget = 50
)
```

## Example Optimizer Definition

```python
from ConfigSpace import ConfigurationSpace
from hpoglue.config import Config
from hpoglue.optimizer import Optimizer
from hpoglue.query import Query


class RandomSearch(Optimizer):
    name = "RandomSearch"
    def __init__(self, problem, seed, working_directory, config_space):
        self.config_space = config_space
        self.config_space.seed(seed)
        self.problem = problem

    def ask(self):
        self._counter += 1
        config = Config(
            config_id=str(self._counter),
            values=self.config_space.sample_configuration().get_dictionary(),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result) -> None:
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

def ackley(query: Query):
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