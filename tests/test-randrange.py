import pytest
import numpy as np
import subprocess
import os
from src.rng_factory import RNG

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_randrange_value_range_and_type(generator_name):
    rng = RNG(generator_name, dim=5, seed=42)

    # Test na 1000 próbek
    values = [rng.randrange(1, 10) for _ in range(1000)]
    assert all(1 <= val < 10 for val in values)
    assert all(isinstance(val, int) for val in values)

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_randrange_reproducibility(generator_name):
    code = f"""
from src.rng_factory import RNG
rng = RNG('{generator_name}', dim=5, seed=42)
print(rng.randrange(1, 10))
"""

    result1 = subprocess.check_output(["python3", "-c", code], cwd=os.path.abspath(".")).decode("utf-8").strip()
    result2 = subprocess.check_output(["python3", "-c", code], cwd=os.path.abspath(".")).decode("utf-8").strip()

    assert result1 == result2

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_randrange_different_values(generator_name):
    rng = RNG(generator_name, dim=5, seed=42)

    values = set([rng.randrange(-5, 5) for _ in range(1000)])
    assert len(values) > 1