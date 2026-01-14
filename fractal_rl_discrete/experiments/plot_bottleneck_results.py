# fractal_rl_discrete/experiments/plot_bottleneck_results.py
# Bottleneck diagnostics with shortest-path distance plot.

from __future__ import annotations

import os, json
from collections import deque
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _adj_from_edges(n: int, edges: List[List[int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for i, j in edges:
        i = int(i); j = int(j)
        if i == j:
            continue
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _bfs_dist(adj: List[List[int]], src: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1, dtype=int)
    q = deque([src])
    dist[src] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _pad(a, L):
    if not a:
        return np.zeros(L, dtype=float)
    a = np.array(a, dtype=float)
    if len(a) < L:
        a = np.concatenate([a, np.full(L - len(a), a[-1])])
    return a[:L]


def main() -> None:
    root = _repo_root()
    in_path = os.path.join(root, "results", "results_bottleneck.json")
    out_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    d = json.load(open(in_path, "r"))
    run = d["runs"][0]
    meta: Dict[str, Any] = run["meta"]

    n = int(meta["S"])
    term = int(meta["terminal_state"])
    edges = meta["edges"]
    bridge = meta.get("bridge_nodes", None)

    vb = np.array(run["baseline"]["V_end"], dtype=float)
    vf = np.array(run["fractional"]["V_end"], dtype=float)

    adj = _adj_from_edges(n, edges)
    dist = _bfs_dist(adj, term)

    # Filter unreachable (shouldn't happen)
    mask = dist >= 0
    dist_m = dist[mask]
    vb_m = vb[mask]
    vf_m = vf[mask]

    # --- Value vs distance plot (key figure) ---
    # Scatter
    plt.figure()
    plt.scatter(dist_m, vb_m, s=10, alpha=0.6, label="baseline")
    plt.scatter(dist_m, vf_m, s=10, alpha=0.6, label="fractional")
    plt.xlabel("shortest-path distance to terminal")
    plt.ylabel("V(s)")
    plt.title("Bottleneck: value vs distance to terminal (final)")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, "diag_bottleneck_value_vs_dist_scatter.png")
    plt.savefig(p); plt.close()
    print("[wrote]", p)

    # Means by distance (cleaner for paper)
    maxd = int(dist_m.max())
    xs = np.arange(maxd + 1)
    mean_b = np.full(maxd + 1, np.nan)
    mean_f = np.full(maxd + 1, np.nan)
    std_b = np.full(maxd + 1, np.nan)
    std_f = np.full(maxd + 1, np.nan)

    for k in xs:
        sel = dist_m == k
        if np.any(sel):
            mean_b[k] = float(np.mean(vb_m[sel]))
            mean_f[k] = float(np.mean(vf_m[sel]))
            std_b[k] = float(np.std(vb_m[sel]))
            std_f[k] = float(np.std(vf_m[sel]))

    plt.figure()
    plt.plot(xs, mean_b, label="baseline")
    plt.plot(xs, mean_f, label="fractional")
    # shading where std exists
    plt.fill_between(xs, mean_b - std_b, mean_b + std_b, alpha=0.2)
    plt.fill_between(xs, mean_f - std_f, mean_f + std_f, alpha=0.2)
    plt.xlabel("shortest-path distance to terminal")
    plt.ylabel("V(s)")
    plt.title("Bottleneck: mean ± std value vs distance (final)")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, "diag_bottleneck_value_vs_dist_mean.png")
    plt.savefig(p); plt.close()
    print("[wrote]", p)

    # --- Convergence curves ---
    db = run["baseline"]["hist"]["deltas"]
    df = run["fractional"]["hist"]["deltas"]
    L = max(len(db), len(df))
    db = _pad(db, L); df = _pad(df, L)
    it = np.arange(1, L + 1)

    plt.figure()
    plt.plot(it, db, label="baseline")
    plt.plot(it, df, label="fractional")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("max |V_{k+1}-V_k|")
    plt.title("Bottleneck: convergence")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, "diag_bottleneck_deltas.png")
    plt.savefig(p); plt.close()
    print("[wrote]", p)

    # --- Print bridge values (useful sanity) ---
    if bridge is not None:
        b1, b2 = bridge
        print("[bridge]", bridge,
              "dist(b1),dist(b2)=", int(dist[b1]), int(dist[b2]),
              "Vb(b1),Vb(b2)=", float(vb[b1]), float(vb[b2]),
              "Vf(b1),Vf(b2)=", float(vf[b1]), float(vf[b2]))

    print("[summary] max|Vf-Vb| =", float(np.max(np.abs(vf - vb))),
          "iters(base,frac)=", run["baseline"]["hist"]["iters"], run["fractional"]["hist"]["iters"])

if __name__ == "__main__":
    main()
