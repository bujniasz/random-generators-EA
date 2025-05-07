from scipy.stats import qmc
import numpy as np
import random
from randomgen import Xoshiro256
from qmcpy import Lattice

class RNG:
    def __init__(self, name, dim, seed):
        self.name = name
        self.dim = dim
        self.seed = seed
        self.rng = self._create_rng()
        self._qrng_buffer = np.array([])
        self._qrng_index = 0


    def _create_rng(self):
        if self.name == "random":
            random.seed(self.seed)
            return random
        elif self.name == "numpy":
            return np.random.default_rng(seed=self.seed)
        elif self.name == "xoshiro":
            return np.random.Generator(Xoshiro256(seed=self.seed))
        elif self.name == "lattice":
            return Lattice(dimension=self.dim, seed=self.seed)
        elif self.name == "sobol":
            return qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        elif self.name == "halton":
            return qmc.Halton(d=self.dim, seed=self.seed)
        else:
            raise ValueError(f"Unknown generator: {self.name}")

    def _next_qrng_value(self):
        if self._qrng_index >= len(self._qrng_buffer):
            if self.name in ["sobol", "halton"]:
                self._qrng_buffer = self.rng.random(1)[0]
            elif self.name == "lattice":
                self._qrng_buffer = self.rng.gen_samples(1)[0]
            else:
                raise RuntimeError("QRNG buffer called on non-QRNG generator.")
            self._qrng_index = 0
        val = self._qrng_buffer[self._qrng_index]
        self._qrng_index += 1
        return val

    def uniform(self, low, high, size):
        if self.name == "random":
            if size == 1:
                return np.array([self.rng.uniform(low, high) for _ in range(self.dim)])
            return np.array([
                [self.rng.uniform(low, high) for _ in range(self.dim)]
                for _ in range(size)
            ])
        elif self.name in ["numpy", "xoshiro"]:
            return self.rng.uniform(low, high, size=(size, self.dim))
        elif self.name == "lattice":
            samples = self.rng.gen_samples(size)
            return low + (high - low) * samples
        else:
            samples = self.rng.random(size)
            return qmc.scale(samples, low, high)


    def choice(self, array):
        if self.name == "random":
            return self.rng.choice(array)
        elif self.name in ["numpy", "xoshiro"]:
            idx = self.rng.integers(0, len(array))
            return array[idx]
        else:
            val = self._next_qrng_value()
            idx = int(val * len(array))
            return array[min(idx, len(array) - 1)]


    def rand(self):
        if self.name == "random":
            return self.rng.random()
        elif self.name in ["numpy", "xoshiro"]:
            return self.rng.random()
        else:
            return self._next_qrng_value()
        

    def randrange(self, start, stop=None):
        if stop is None:
            stop = start
            start = 0

        if self.name == "random":
            return self.rng.randrange(start, stop)
        elif self.name in ["numpy", "xoshiro"]:
            return self.rng.integers(start, stop)
        else:
            val = self._next_qrng_value()
            return start + int(val * (stop - start))