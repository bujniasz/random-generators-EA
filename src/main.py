import numpy as np
from cec2017.functions import f2, f13
from evolutionary_alg import evolutionary_classic
from rng_factory import RNG
import time
import secrets

if __name__ == "__main__":
    MAX_X = 100  # Boundary limit for the solution
    DIMENSIONALITY = 10

    RUNS = 3        # Number of runs for the optimization algorithm
    U = 10          # Population size
    FES = 50000     # Number of objective function evaluationsW
    PC = 0.5
    DELTA_S = 0.1
    DELTA_B = 10
    P_BIG_JUMP = 0.02

    res = []

    for i in range(RUNS):
        p0 = []
        #seed = int(time.time() * 1e6) % 2**32
        seed = secrets.randbits(32)
        print(seed)
        rng = RNG("random", DIMENSIONALITY, seed=seed)      #Mersenne Twister
        #rng = RNG("numpy", DIMENSIONALITY, seed=seed)      #PCG
        #rng = RNG("xoshiro", DIMENSIONALITY, seed=seed)    #xoshiro256
        #rng = RNG("sobol", DIMENSIONALITY, seed=seed)      #Sobol   
        #rng = RNG("halton", DIMENSIONALITY, seed=seed)     #Halton
        #rng = RNG("lattice", DIMENSIONALITY, seed=seed)    #Lattice


        for j in range(U):
            x = np.array(rng.uniform(-MAX_X, MAX_X, 1)).reshape(-1)
            p0.append(x)

        t_max = FES / U  # Maximum number of generations (iterations)
        o, x = evolutionary_classic(f2, p0, U, DELTA_S, DELTA_B, P_BIG_JUMP, PC, t_max, MAX_X, rng)
        res.append(o)

    print(
        f"min: {np.min(res):.3f}, max: {np.max(res):.3f}, avg: {np.mean(res):.3f}, std: {np.std(res):.3f}"
    )
