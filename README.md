# random-generators-EA

This project investigates the impact of different random number generators (RNGs) on the performance of an evolutionary algorithm. The study includes both traditional RNGs and quasi-random generators. CEC-2017 benchmark functions are used to evaluate and compare the efficiency and effectiveness of each RNG type in the context of evolutionary optimization. The goal is to understand how the choice of RNG affects the algorithm's convergence speed, accuracy, and overall performance in solving optimization problems.

## Authors

- **Aleksander Bujnowski** 
- **Wojciech Kondracki**

## Repo structure

1. `docs` folder - report from the study
2. `study_results` folder - data used in the report. Contains also two SEED files: `previous_seeds_full.txt` and `previous_seeds_short_budget_only` if the user wishes to reproduce results from the study.
2. `src` folder:
    * `main.py` - entry point of the project: initializes the experiment, runs the algorithm, and collects results.
    * `evolutionary_alg.py` - implements the core evolutionary algorithm.
    * `rng_factory.py` - provides a unified wrapper for multiple random number generators.
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
python main.py --replay <SEED_FILENAME>.txt
```

This guarantees:
- identical population initialization,
- fully reproducible convergence,
- identical random decisions during evolutionary runs.

## Generated results

Running the `main.py` script generates two main directories containing experiment results:
1. `results_data/` - contains all numerical results, statistics, and test outputs.
    - `main_results.csv` - a flat table aggregating the final results of all runs.
    - `convergence.csv` - detailed convergence data for all runs.
    - `{budget}/{function}/` - each tested function under each budget gets a dedicated subfolder. 
        - `score_results.csv` - statistical summary of optimization results for each generator.
        - `runtime_results.csv` - statistical summary of runtime for each generator.
        - `kruskal_results.csv` - result of the Kruskal-Wallis H-test, reporting overall statistical significance.
        - (Optional) `posthoc_dunn.csv` - pairwise comparisons using the Dunn test with Bonferroni correction, if Kruskal-Wallis detected significance (p < 0.05).

2. `plots/{budget}/{function}/` - contains visualizations of the results. Each tested function under each budget has its own subdirectory.
    - `boxplot.png` - boxplot of the optimization results across all generators.
    - `convergence_curve.png` - lineplot of average convergence curves (score vs. iteration) across runs.
    - `time_vs_score.png` - scatterplot comparing runtime to final score for each run.
    - (Optional) `posthoc_dunn.png` - heatmap visualization of the post hoc Dunn test (only if Kruskal-Wallis indicated significance).