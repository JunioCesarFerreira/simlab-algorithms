r"""Plot the ind2 instance graph and the mobile motes' trajectories.

Produces two figures under ``results/instance/``:

* ``ind2_graph.png``        — static reach-graph of the instance: sink + candidate
  relays as nodes, an edge whenever two nodes are within the reach radius ``R``.
* ``ind2_mobile_paths.png`` — the same layout (candidates faint) overlaid with the
  ``M`` mobile sensors' trajectories (start/end markers).

Both share the canonical geometry from ``adjacency_builder`` so they match exactly
what Methods 2/3 consume.
"""

from __future__ import annotations

import numpy as np

import adjacency_builder as ab
from methods.common import INSTANCE_PATH, RESULTS_DIR, ensure_dir, get_plt

OUT_DIR = RESULTS_DIR / "instance"


def _load():
    inst = ab.load_instance(str(INSTANCE_PATH))
    res = ab.build_from_instance(inst)
    layout = res["layout"]
    positions = res["positions"]          # (T, K, 2)
    R = res["radius"]
    N, M = layout.N, layout.M
    sink = np.asarray(inst.sink, dtype=float)
    cand = np.asarray(inst.candidates, dtype=float)               # (N, 2)
    # mobiles occupy the last M node slots: [sink, candidates(N), mobiles(M)]
    mob = positions[:, 1 + N:, :]                                 # (T, M, 2)
    return inst, sink, cand, mob, R, N, M


def _reach_edges(nodes: np.ndarray, R: float):
    """Yield index pairs (i, j) of nodes within reach R (i < j)."""
    d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    iu, ju = np.where(np.triu(d <= R, k=1))
    return list(zip(iu.tolist(), ju.tolist()))


# ---------------------------------------------------------------------------
# Figure 1 — static instance reach-graph
# ---------------------------------------------------------------------------
def plot_graph():
    plt = get_plt()
    inst, sink, cand, mob, R, N, M = _load()
    nodes = np.vstack([sink[None, :], cand])                      # 0 = sink, 1..N candidates
    edges = _reach_edges(nodes, R)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for i, j in edges:
        ax.plot(nodes[[i, j], 0], nodes[[i, j], 1], "-", color="0.75",
                lw=0.6, zorder=1)
    ax.scatter(cand[:, 0], cand[:, 1], s=45, color="C0", edgecolor="k",
               linewidth=0.4, zorder=3, label=f"candidatos ($N={N}$)")
    ax.scatter([sink[0]], [sink[1]], marker="*", s=420, color="red",
               edgecolor="k", linewidth=0.6, zorder=4, label="sorvedouro")
    # one reach-radius circle around the sink for scale
    ax.add_patch(plt.Circle((sink[0], sink[1]), R, fill=False, ls="--",
                            color="red", alpha=0.5, lw=1.0, zorder=2))
    ax.set_aspect("equal")
    ax.set_title(fr"Instância ind2 — grafo de alcance ($R={R:.0f}$): "
                 f"{len(edges)} arestas")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "ind2_graph.png"
    fig.savefig(out); plt.close(fig)
    return out, len(edges)


# ---------------------------------------------------------------------------
# Figure 2 — mobile trajectories
# ---------------------------------------------------------------------------
def plot_paths():
    plt = get_plt()
    inst, sink, cand, mob, R, N, M = _load()
    T = mob.shape[0]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(cand[:, 0], cand[:, 1], s=28, color="0.7", edgecolor="k",
               linewidth=0.2, zorder=2, label="candidatos")
    cmap = plt.get_cmap("tab10")
    for m in range(M):
        xy = mob[:, m, :]
        c = cmap(m % 10)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=1.8, alpha=0.9,
                zorder=3, label=f"móvel {m}")
        ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=70, color=c,
                   edgecolor="k", linewidth=0.6, zorder=4)        # início
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="s", s=60, color=c,
                   edgecolor="k", linewidth=0.6, zorder=4)        # fim
    ax.scatter([sink[0]], [sink[1]], marker="*", s=420, color="red",
               edgecolor="k", linewidth=0.6, zorder=5, label="sorvedouro")
    ax.set_aspect("equal")
    ax.set_title(fr"Instância ind2 — trajetórias dos {M} móveis ($T={T}$ passos)"
                 "\n(círculo = início, quadrado = fim)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "ind2_mobile_paths.png"
    fig.savefig(out); plt.close(fig)
    return out, T


def main() -> int:
    ensure_dir(OUT_DIR)
    g, n_edges = plot_graph()
    p, T = plot_paths()
    print("=== ind2 instance plots ===")
    print(f"  graph  -> {g}   ({n_edges} reach edges)")
    print(f"  paths  -> {p}   (T={T} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
