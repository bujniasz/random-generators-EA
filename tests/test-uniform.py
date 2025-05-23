import pytest
import numpy as np
import subprocess
from src.rng_factory import RNG

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_uniform_value_range(generator_name):
    rng = RNG(generator_name, dim=5, seed=42)
    result = rng.uniform(-10, 10, 1).reshape(-1)
    assert np.all(result >= -10)
    assert np.all(result < 10)

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_uniform_output_shape(generator_name):
    rng = RNG(generator_name, dim=5, seed=42)
    result = rng.uniform(-10, 10, 1).reshape(-1)
    assert result.shape == (5,)

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_uniform_reproducibility(generator_name):
    code = f"""
from src.rng_factory import RNG
import numpy as np
rng = RNG('{generator_name}', dim=5, seed=42)
result = rng.uniform(-10, 10, 1).reshape(-1)
print(",".join(map(str, result)))
"""

    result1 = subprocess.check_output(["python3", "-c", code]).decode("utf-8").strip()
    result2 = subprocess.check_output(["python3", "-c", code]).decode("utf-8").strip()

    arr1 = np.fromstring(result1, sep=',')
    arr2 = np.fromstring(result2, sep=',')

    assert np.array_equal(arr1, arr2)