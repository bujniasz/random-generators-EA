from scipy.stats import qmc
import numpy as np
import random

class RNG:
    def __init__(self, name, dim, seed=None):
        self.name = name
        self.dim = dim
        self.seed = seed
        self.rng = self._create_rng()


    def _create_rng(self):
        if self.name == "random":
            random.seed(self.seed)
            return random
        elif self.name == "numpy":
            return np.random.default_rng(self.seed)
        if self.name == "sobol":
            return qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        elif self.name == "halton":
            return qmc.Halton(d=self.dim, seed=self.seed)
        else:
            raise ValueError(f"Unknown generator: {self.name}")


    def uniform(self, low, high, size):
        if self.name == "random":
            if size == 1:
                return np.array([self.rng.uniform(low, high) for _ in range(self.dim)])
            return np.array([
                [self.rng.uniform(low, high) for _ in range(self.dim)]
                for _ in range(size)
            ])
        elif self.name == "numpy":
            return self.rng.uniform(low, high, size=(size, self.dim))
        else:
            samples = self.rng.random(size)
            return qmc.scale(samples, low, high)


    def choice(self, array):
        if self.name == "random":
            return self.rng.choice(array)
        elif self.name == "numpy":
            idx = self.rng.integers(0, len(array))
            return array[idx]
        else:
            idx = int(self.rng.random(1)[0, 0] * len(array))
            return array[min(idx, len(array) - 1)]


    def rand(self):
        if self.name == "random":
            return self.rng.random()
        elif self.name == "numpy":
            return self.rng.random()
        else:
            return self.rng.random(1)[0, 0]
        

    def randrange(self, start, stop=None):
        if stop is None:
            stop = start
            start = 0
        width = stop - start

        if self.name == "random":
            return self.rng.randrange(start, stop)
        elif self.name == "numpy":
            return self.rng.integers(start, stop)
        else:
            val = self.rng.random(1)[0, 0]
            return start + int(val * width)