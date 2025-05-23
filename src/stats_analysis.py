import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import LogLocator


# Wczytanie danych
df = pd.read_csv("results_data/main_results.csv")
df["score"] = df["score"].astype(float)
df["run_time"] = df["run_time"].astype(float)

GENERATOR_ORDER = ["random", "numpy", "xoshiro", "sobol", "halton", "lattice"]  

for fes_type in df["fes_type"].unique():
    for func in df["function"].unique():
        print(f"\n{fes_type.upper()} | Funkcja testowa: {func}")
        sub = df[(df["fes_type"] == fes_type) & (df["function"] == func)]

        # Foldery
        func_dir = f"results_data/{fes_type}/{func}"
        plot_dir = f"plots/{fes_type}/{func}"
        os.makedirs(func_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        # Statystyki opisowe: wynik
        desc_stats = (
            sub.groupby("generator")["score"]
            .agg(["mean", "std", "min", "max"])
            .reindex(GENERATOR_ORDER)
            .applymap(lambda x: f"{x:.3e}")
        )
        desc_stats.to_csv(os.path.join(func_dir, "score_results.csv"))

        # Statystyki opisowe: czas
        time_stats = (
            sub.groupby("generator")["run_time"]
            .agg(["mean", "std"])
            .reindex(GENERATOR_ORDER)
            .applymap(lambda x: f"{x:.3f}")
        )
        time_stats.to_csv(os.path.join(func_dir, "runtime_results.csv"))

        # Test Kruskala-Wallisa
        grouped = [group["score"].values for name, group in sub.groupby("generator")]
        stat, p = stats.kruskal(*grouped)
        kruskal_df = pd.DataFrame({
            "H_statistic": [round(stat, 3)],
            "p_value": [round(p, 3)]
        })
        kruskal_df.to_csv(os.path.join(func_dir, "kruskal_results.csv"), index=False)

        if p < 0.05:
            print("-> Istotne różnice, wykonuję test post hoc (Dunna)...")
            posthoc = sp.posthoc_dunn(sub, val_col='score', group_col='generator', p_adjust='bonferroni')
            posthoc = posthoc.round(3)
            posthoc.to_csv(os.path.join(func_dir, "posthoc_dunn.csv"))

            plt.figure(figsize=(8, 6))
            sns.heatmap(posthoc, annot=True, cmap="coolwarm", fmt=".3f")
            plt.title(f"Test post hoc Dunn – {func} ({fes_type})")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "posthoc_dunn.png"))
            plt.close()
        else:
            print("-> Brak istotnych różnic, nie wykonuję testu post hoc")

        # Boxplot pełny
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="generator", y="score", data=sub, hue="generator", dodge=False, showfliers=True)
        sns.stripplot(x="generator", y="score", data=sub, color="black", alpha=0.5, jitter=True, dodge=False)
        plt.title(f"Rozkład wyników – {func} ({fes_type})")
        plt.xlabel("Generator")
        plt.ylabel("Najlepszy wynik")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "boxplot.png"))
        plt.close()

        # Boxplot bez lattice
        filtered = sub[sub["generator"] != "lattice"]
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="generator", y="score", data=filtered, hue="generator", dodge=False, showfliers=True)
        sns.stripplot(x="generator", y="score", data=filtered, color="black", alpha=0.5, jitter=True, dodge=False)
        plt.title(f"Rozkład wyników – {func} ({fes_type}) bez lattice")
        plt.xlabel("Generator")
        plt.ylabel("Najlepszy wynik")
        plt.yscale("log")
        plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "boxplot_nolattice.png"))
        plt.close()

        # Scatterplot pełny
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sub, x="run_time", y="score", hue="generator", style="generator", s=100)
        plt.title(f"Czas działania vs wynik – {func} ({fes_type})")
        plt.xlabel("Czas działania (s)")
        plt.ylabel("Najlepszy wynik")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "time_vs_score.png"))
        plt.close()

        # Scatterplot bez lattice
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=filtered, x="run_time", y="score", hue="generator", style="generator", s=100)
        plt.title(f"Czas działania vs wynik – {func} ({fes_type}) bez lattice")
        plt.xlabel("Czas działania (s)")
        plt.ylabel("Najlepszy wynik")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "time_vs_score_nolattice.png"))
        plt.close()