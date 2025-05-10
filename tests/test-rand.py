import pytest
import numpy as np
import subprocess
import os
from src.rng_factory import RNG

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton", "lattice"])
def test_rand_value_range(generator_name):
    rng = RNG(generator_name, dim=5, seed=42)

    values = [rng.rand() for _ in range(1000)]
    assert all(0.0 <= val < 1.0 for val in values)

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton", "lattice"])
def test_rand_reproducibility(generator_name):
    code = f"""
from src.rng_factory import RNG
rng = RNG('{generator_name}', dim=5, seed=42)
print(rng.rand())
"""

    result1 = subprocess.check_output(["python3", "-c", code], cwd=os.path.abspath(".")).decode("utf-8").strip()
    result2 = subprocess.check_output(["python3", "-c", code], cwd=os.path.abspath(".")).decode("utf-8").strip()

    assert result1 == result2

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton", "lattice"])
def test_rand_variability(generator_name):
    rng = RNG(generator_name, dim=5, seed=42)

    values = [rng.rand() for _ in range(1000)]
    unique_values = set(values)

    assert len(unique_values) > 1