from scipy.stats import qmc
import numpy as np

class RNG:
    def __init__(self, name, dim, seed=None):
        self.name = name
        self.dim = dim
        self.seed = seed
        self.rng = self._create_rng()

    def _create_rng(self):
        if self.name == "sobol":
            return qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        elif self.name == "halton":
            return qmc.Halton(d=self.dim, seed=self.seed)
        elif self.name == "numpy":
            return np.random.default_rng(self.seed)
        else:
            raise ValueError(f"Unknown generator: {self.name}")

    def uniform(self, low, high, size):
        if self.name in ["sobol", "halton"]:
            samples = self.rng.random(size)
            return qmc.scale(samples, low, high)
        elif self.name == "numpy":
            return self.rng.uniform(low, high, size=(size, self.dim))
