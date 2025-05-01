import scipy.stats as stats
import scikit_posthocs as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def kruskal_dunn(results_dict):
    """
    Perform Kruskal-Wallis and Dunn's post hoc test.

    Parameters:
        results_dict (dict): keys = method names, values = list of scores
    """
    print("==== KRUSKAL-WALLIS TEST ====")
    data = [results for results in results_dict.values()]
    stat, p = stats.kruskal(*data)
    print(f"H-statistic: {stat:.4f}, p-value: {p:.4g}")

    # Flatten for plotting and Dunn test
    flat_data = []
    groups = []
    for name, values in results_dict.items():
        flat_data.extend(values)
        groups.extend([name] * len(values))

    df = pd.DataFrame({"group": groups, "value": flat_data})

    # Boxplot
    sns.boxplot(data=df, x="group", y="value")
    sns.swarmplot(data=df, x="group", y="value", color=".25")
    plt.title("Porównanie wyników algorytmu dla różnych RNG")
    plt.tight_layout()
    plt.savefig("wyniki_rng.png")
    plt.show()

    if p < 0.05:
        print("\n==== DUNN POST HOC TEST (Bonferroni) ====")
        pvals = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')
        print(pvals.round(4))
    else:
        print("Brak istotnych różnic między grupami (p >= 0.05)")
