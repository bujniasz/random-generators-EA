import numpy as np
from cec2017.functions import f2, f3, f8, f13, f15
from evolutionary_alg import evolutionary_classic
from rng_factory import RNG
import time
import secrets
import pandas as pd
import os
import subprocess
import sys
import shutil

for folder in ["plots", "results_data"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

if __name__ == "__main__":
    MAX_X = 100
    DIMENSIONALITY = 10
    RUNS = 50
    U = 20
    PC = 0.5
    DELTA_S = 0.1
    DELTA_B = 10
    P_BIG_JUMP = 0.05

    FES_SETTINGS = {
        "short_budget": 1000,
        "long_budget": 75000
    }

    FUNCTIONS = {
        "f2": f2,
        "f3": f3,
        "f8": f8,
        "f13": f13,
        "f15": f15
    }

    GENERATORS = [
        "random",   # Mersenne Twister
        "numpy",    # PCG
        "xoshiro",  # xoshi#ro256
        "sobol",    # Sobol
        "halton",   # Halton
        "lattice"   # Lattice
    ]

    results = []

    for fes_label, FES in FES_SETTINGS.items():
        for func_name, func in FUNCTIONS.items():
            for gen_name in GENERATORS:
                print(f"\n=== {fes_label.upper()} | {func_name} | {gen_name} ===")
                for i in range(RUNS):
                    seed = secrets.randbits(32)
                    rng = RNG(gen_name, DIMENSIONALITY, seed=seed)

                    # Population initialization
                    p0 = [
                        np.array(rng.uniform(-MAX_X, MAX_X, 1)).reshape(-1)
                        for _ in range(U)
                    ]

                    t_max = FES / U

                    start_time = time.time()
                    score, _ = evolutionary_classic(
                        func, p0, U, DELTA_S, DELTA_B,
                        P_BIG_JUMP, PC, t_max, MAX_X, rng
                    )
                    run_time = time.time() - start_time

                    results.append({
                        "fes_type": fes_label,
                        "function": func_name,
                        "generator": gen_name,
                        "score": f"{score:.2f}",
                        "run_time": f"{run_time:.2f}",
                    })


    os.makedirs("results_data", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("results_data/main_results.csv", index=False)
    print("\n[OK] Zapisano wyniki do results_data/main_results.csv")

    print("[INFO] Uruchamiam analizę statystyczną...")
    subprocess.run([sys.executable, "src/stats_analysis.py"])

    print("[OK] Analiza statystyczna zakończona")