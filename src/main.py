import numpy as np
from cec2017.functions import f2, f13
import cec2017
from evolutionary_alg import evolutionary_classic
from rng_factory import RNG

if __name__ == "__main__":
    MAX_X = 100  # Boundary limit for the solution
    DIMENSIONALITY = 10

    RUNS = 5   # Number of runs for the optimization algorithm
    SIGMA = 1    # Mutation strength
    U = 10       # Population size
    FES = 50000  # Number of objective function evaluations
    PC = 0.5

    res = []

    for i in range(RUNS):
        p0 = []
        rng = RNG("sobol", DIMENSIONALITY, seed=i)
        # rng = RNG("numpy", DIMENSIONALITY, seed=i)  # dla pseudolosowego
        # rng = RNG("halton", DIMENSIONALITY, seed=i)  # dla innego quasi


        for j in range(U):
            x = rng.uniform(-MAX_X, MAX_X, 1)[0]
            p0.append(x)

        t_max = FES / U  # Maximum number of generations (iterations)
        o, x = evolutionary_classic(f2, p0, U, SIGMA, PC, t_max, MAX_X)
        res.append(o)

    print(
        f"min: {np.min(res):.3f}, max: {np.max(res):.3f}, avg: {np.mean(res):.3f}, std: {np.std(res):.3f}"
    )
