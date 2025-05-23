import pytest
import numpy as np
import subprocess
import os
from src.rng_factory import RNG

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_choice_from_list(generator_name):
    rng = RNG(generator_name, dim=5, seed=42)

    values = [1, 2, 3, 4, 5]
    assert rng.choice(values) in values

    tuples = [(1, 'a'), (2, 'b'), (3, 'c')]
    assert rng.choice(tuples) in tuples

    arrays = [np.array([1, 2]), np.array([3, 4])]
    chosen = rng.choice(arrays)
    assert any(np.array_equal(chosen, arr) for arr in arrays)

    objects = [{"id": 1}, {"id": 2}]
    assert rng.choice(objects) in objects

@pytest.mark.parametrize("generator_name", ["random", "numpy", "xoshiro", "sobol", "halton"])
def test_choice_reproducibility(generator_name):
    code = f"""
from src.rng_factory import RNG
import numpy as np
rng = RNG('{generator_name}', dim=5, seed=42)
values = [1, 2, 3, 4, 5]
print(rng.choice(values))
"""

    result1 = subprocess.check_output(["python3", "-c", code], cwd=os.path.abspath(".")).decode("utf-8").strip()
    result2 = subprocess.check_output(["python3", "-c", code], cwd=os.path.abspath(".")).decode("utf-8").strip()

    assert result1 == result2