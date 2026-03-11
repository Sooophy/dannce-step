import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def simulate_5_animals(left_prob: float, right_prob: float, total_events: int) -> np.ndarray:
    simulated_statistics = []
    for _ in range(5):
        labels = np.random.choice(['l', 'r'], size=total_events, p=[left_prob, right_prob])
        l_count = np.sum(labels == 'l')
        r_count = np.sum(labels == 'r')
        simulated_statistics.append((l_count - r_count) / total_events)
    return np.array(simulated_statistics)


def compute_power(total_events_ls, mean_ob_stat_ls, num_simulations, csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        fieldnames = ["l-r/total ratio", "rearing events", "power"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for total_events in total_events_ls:
            for mean_ob_stat in mean_ob_stat_ls:
                count_extreme = 0

                left_prob = (1 + mean_ob_stat) / 2
                right_prob = (1 - mean_ob_stat) / 2

                for _ in range(num_simulations):
                    healthy_ratio = simulate_5_animals(0.5, 0.5, total_events)
                    stroke_ratio = simulate_5_animals(left_prob, right_prob, total_events)

                    _, p_value = ttest_ind(
                        healthy_ratio,
                        stroke_ratio,
                        alternative="less"
                    )

                    if p_value < 0.05:
                        count_extreme += 1

                power = count_extreme / num_simulations

                writer.writerow(
                    {
                        "l-r/total ratio": mean_ob_stat,
                        "rearing events": total_events,
                        "power": power,
                    }
                )


def visualize_power(data: pd.DataFrame, output_path: str):
    pivot_data = (
        data.pivot(
            index="l-r/total ratio",
            columns="rearing events",
            values="power",
        )
        .sort_index(ascending=False)
    )

    n = 128
    yellows = np.zeros((n, 4))
    for i in range(n):
        val = i / (n - 1)
        r = 0.8 + (1 - 0.8) * val
        g = 0.8 + (1 - 0.8) * val
        b = 0.2 + (1 - 0.2) * val
        yellows[i] = [r, g, b, 1]

    blues = plt.cm.Blues(np.linspace(0, 1, 128))
    colors_combined = np.vstack((yellows, blues))
    custom_cmap = mcolors.ListedColormap(colors_combined)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.8, vmax=1)

    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(
        pivot_data,
        cmap=custom_cmap,
        norm=norm,
        annot=False,
        cbar_kws={"ticks": [0, 0.8, 1.0]},
    )

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([0, 0.8, 0.95, 1.0])
    colorbar.set_ticklabels([])

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    total_events_ls = [1, 3, 5, 10, 15, 20, 30, 50]
    mean_ob_stat_ls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_simulations = 1000
    csv_filename = "rearing_data.csv"
    output_plot = "power_heatmap.pdf"

    compute_power(total_events_ls, mean_ob_stat_ls, num_simulations, csv_filename)

    data = pd.read_csv(csv_filename)
    visualize_power(data, output_plot)


if __name__ == "__main__":
    main()