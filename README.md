# random-generators-EA

This project investigates the impact of different random number generators (RNGs) on the performance of an evolutionary algorithm. The study includes both traditional RNGs and quasi-random generators. CEC-2017 benchmark functions are used to evaluate and compare the efficiency and effectiveness of each RNG type in the context of evolutionary optimization. The goal is to understand how the choice of RNG affects the algorithm's convergence speed, accuracy, and overall performance in solving optimization problems.

## Authors

- **Aleksander Bujnowski** 
- **Wojciech Kondracki**

## Repo structure

1. `docs` folder - report from the study  
2. `src` folder:
    * `main.py` - entry point of the project: initializes the experiment, runs the algorithm, and collects results.
    * `evolutionary_alg.py` - implements the core evolutionary algorithm, including selection, crossover, and mutation.
    * `rng_factory.py` - provides a unified wrapper for multiple random number generators (pseudo and quasi-random).
    * `stats_analyzis.py` - performs statistical analysis and visualizes performance differences between RNGs.
3. `tests` folder - tests of custom functions from `RNG` wrapper.
4. `.github/workflows` - CI/CD definition.

## How to Use

This project supports two modes of execution:

### 1. Standard Mode (fresh experiment)

Runs the evolutionary algorithm with newly generated random seeds.  
All seeds will be saved to a file (`src/previous_seeds.txt`) for reproducibility.
From `/src`  run:

```bash
python3 main.py
```

### 2. Replay Mode (repeat previous experiment)
Runs the exact same experiment as before, using a previously saved list of seeds.
From `/src`  run:

```bash
python main.py --replay previous_seeds.txt
```

This guarantees:
- identical population initialization,
- fully reproducible convergence,
- identical random decisions during evolutionary runs.