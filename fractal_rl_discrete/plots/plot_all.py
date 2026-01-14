import json
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_single_runs(data, title, outpath):
    plt.figure()
    for run in data["runs"]:
        y = np.array(run["v_start"], dtype=float)
        plt.plot(np.arange(len(y)), y, label=run["name"])
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("V(start)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_mean_std(data, title, outpath):
    plt.figure()
    for run in data["runs"]:
        m = np.array(run["v_start_mean"], dtype=float)
        s = np.array(run["v_start_std"], dtype=float)
        x = np.arange(len(m))
        plt.plot(x, m, label=run["name"])
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("V(start)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)

    with open("results/results_chain.json", "r") as f:
        chain = json.load(f)
    plot_single_runs(
        chain,
        f"Chain (N={chain['N']}, gamma={chain['gamma']})",
        "figures/fig_chain_vstart.png",
    )

    with open("results/results_bottleneck.json", "r") as f:
        bn = json.load(f)
    plot_mean_std(
        bn,
        "Bottleneck: V(start) mean ± std (5 seeds)",
        "figures/fig_bottleneck_vstart.png",
    )

    with open("results/results_gridworld.json", "r") as f:
        gw = json.load(f)
    plot_mean_std(
        gw,
        "Gridworld: V(start) mean ± std (5 seeds)",
        "figures/fig_gridworld_vstart.png",
    )

    print("Wrote figures/fig_chain_vstart.png")
    print("Wrote figures/fig_bottleneck_vstart.png")
    print("Wrote figures/fig_gridworld_vstart.png")


if __name__ == "__main__":
    main()
