import numpy as np
from cec2017.functions import f2, f5, f8, f11, f14, f17, f20, f23, f26, f29
from evolutionary_alg import evolutionary_classic
from rng_factory import RNG
import time
import secrets
import pandas as pd
import os
import subprocess
import sys
import shutil
import argparse

for folder in ["../plots", "../results_data"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)


parser = argparse.ArgumentParser()
parser.add_argument("--replay", help="Ścieżka do pliku z seedami", default=None)
args = parser.parse_args()


if __name__ == "__main__":
    MAX_X = 100
    DIMENSIONALITY = 30
    RUNS = 30
    U = 50
    PC = 0.5
    DELTA_S = 0.1
    DELTA_B = 10
    P_BIG_JUMP = 0.03

    FES_SETTINGS = {
        "short_budget": 950,
        "long_budget": 49950
    }

    FUNCTIONS = {
        "f2": f2,
        "f5": f5,
        "f8": f8,
        "f11": f11,
        "f14": f14,
        "f17": f17,
        "f20": f20,
        "f23": f23,
        "f26": f26,
        "f29": f29
    }

    GENERATORS = [
        "random",   # Mersenne Twister
        "numpy",    # PCG
        "xoshiro",  # xoshiro256
        "sobol",    # Sobol
        "halton"    # Halton
    ]

    results = []
    convergence_data = []
    all_seeds = []
    if args.replay:
        print(f"[INFO] Tryb odtworzenia eksperymentu z pliku {args.replay}")
        with open(args.replay, "r") as f:
            all_seeds = [int(line.strip()) for line in f.readlines()]
        seed_index = 0
    else:
        print("[INFO] Tryb nowego eksperymentu – generuję nowe seedy.")

    for fes_label, FES in FES_SETTINGS.items():
        for func_name, func in FUNCTIONS.items():
            for gen_name in GENERATORS:
                print(f"\n=== {fes_label.upper()} | {func_name} | {gen_name} ===")
                for i in range(RUNS):
                    if args.replay:
                        seed = all_seeds[seed_index]
                        seed_index += 1
                    else:
                        seed = secrets.randbits(32)
                        all_seeds.append(seed)
                    rng = RNG(gen_name, DIMENSIONALITY, seed=seed)

                    p0 = [
                        np.array(rng.uniform(-MAX_X, MAX_X, 1)).reshape(-1)
                        for _ in range(U)
                    ]

                    t_max = FES / U

                    start_time = time.time()
                    score, history = evolutionary_classic(
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

                    for iter_idx, iter_score in enumerate(history):
                        convergence_data.append({
                            "fes_type": fes_label,
                            "function": func_name,
                            "generator": gen_name,
                            "run": i,
                            "iteration": iter_idx,
                            "score": iter_score
                        })


    os.makedirs("../results_data", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("../results_data/main_results.csv", index=False)
    print("\n[OK] Zapisano wyniki końcowe do results_data/main_results.csv")

    df_conv = pd.DataFrame(convergence_data)
    df_conv.to_csv("../results_data/convergence.csv", index=False)
    print("\n[OK] Zapisano wyniki historyczne do results_data/convergence.csv")

    if not args.replay:
        with open("previous_seeds.txt", "w") as f:
            for s in all_seeds:
                f.write(f"{s}\n")
        print("\n[OK] Zapisano użyte seedy do src/previous_seeds.txt")

    print("\n[INFO] Uruchamiam analizę statystyczną...")
    subprocess.run([sys.executable, "stats_analysis.py"])

    print("\n[OK] Analiza statystyczna zakończona")