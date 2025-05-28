import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
import numpy as np

df = pd.read_csv("../results_data/main_results.csv")
df["score"] = df["score"].astype(float)
df["run_time"] = df["run_time"].astype(float)

df_conv = pd.read_csv("../results_data/convergence.csv")
df_conv["score"] = df_conv["score"].astype(float)

GENERATOR_ORDER = ["random", "numpy", "xoshiro", "sobol", "halton"]

for fes_type in df["fes_type"].unique():
    for func in df["function"].unique():
        print(f"\n{fes_type.upper()} | Funkcja testowa: {func}")
        sub = df[(df["fes_type"] == fes_type) & (df["function"] == func)]

        func_dir = f"../results_data/{fes_type}/{func}"
        plot_dir = f"../plots/{fes_type}/{func}"
        os.makedirs(func_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        desc_stats = (
            sub.groupby("generator")["score"]
            .agg(["mean", "std", "min", "max"])
            .reindex(GENERATOR_ORDER)
            .round(3)
        )
        desc_stats.to_csv(os.path.join(func_dir, "score_results.csv"))

        time_stats = (
            sub.groupby("generator")["run_time"]
            .agg(["mean", "std"])
            .reindex(GENERATOR_ORDER)
            .round(3)
        )
        time_stats.to_csv(os.path.join(func_dir, "runtime_results.csv"))

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

        plt.figure(figsize=(12, 6))
        outliers = pd.DataFrame()
        non_outliers = pd.DataFrame()
        for gen in sub["generator"].unique():
            gen_data = sub[sub["generator"] == gen].copy()
            q1 = gen_data["score"].quantile(0.25)
            q3 = gen_data["score"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            out_mask = (gen_data["score"] < lower_bound) | (gen_data["score"] > upper_bound)

            outliers = pd.concat([outliers, gen_data[out_mask]])
            non_outliers = pd.concat([non_outliers, gen_data[~out_mask]])
        sns.boxplot(
            x="generator", y="score", data=sub,
            hue="generator", dodge=False, showfliers=False
        )
        sns.stripplot(
            x="generator", y="score", data=non_outliers,
            color="black", alpha=0.6, jitter=True, dodge=False, size=5
        )
        sns.stripplot(
            x="generator", y="score", data=outliers,
            marker='o', facecolors='white', edgecolor='black', linewidth=0.8,
            jitter=True, dodge=False, size=6
        )
        plt.title(f"Rozkład wyników – {func} ({fes_type})")
        plt.xlabel("Generator")
        plt.ylabel("Wynik Run'a")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "boxplot.png"))
        plt.close()


        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sub, x="run_time", y="score", hue="generator", style="generator", s=100)
        plt.title(f"Czas działania vs wynik – {func} ({fes_type})")
        plt.xlabel("Czas działania (s)")
        plt.ylabel("Wynik Run'a")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "time_vs_score.png"))
        plt.close()

        conv_sub = df_conv[(df_conv["fes_type"] == fes_type) & (df_conv["function"] == func)]
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=conv_sub,
            x="iteration", y="score",
            hue="generator", estimator="mean", errorbar=None, palette="tab10"
        )
        plt.title(f"Krzywa konwergencji – {func} ({fes_type})")
        plt.xlabel("Iteracja")
        plt.ylabel("Średni wynik z iteracji")
        plt.yscale("log")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "convergence_curve.png"))
        plt.close()