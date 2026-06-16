"""Plotting utilities for the P2 MILP sweep.

Ported from wsn-design-space-exploration/milp/mobile-model/utils/plot_utils.py
with minor adaptations for the P2 candidate key format ("j", f"cand{i}").
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

COLOR_SINK      = "blue"
COLOR_CANDIDATE = "black"
COLOR_FIXED0    = "gray"    # not installed
COLOR_FIXED1    = "red"     # installed
COLOR_MOBILE    = "green"

S_SINK   = 260
S_FIXED0 = 40
S_FIXED1 = 120
S_MOBILE = 18


# ---------------------------------------------------------------------------
# Trajectory helper
# ---------------------------------------------------------------------------

def _traj_with_breaks(r_mobile_fn, T: int, close: bool = False,
                      jump_factor: float = 5.0) -> np.ndarray:
    """Build (N, 2) trajectory array inserting NaN rows at large jumps."""
    pts = np.array([r_mobile_fn(t) for t in range(1, T + 1)], dtype=float)
    if close:
        pts = np.vstack([pts, pts[0]])
    if len(pts) >= 2:
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        med   = np.median(dists) if np.any(dists > 0) else 0.0
        thr   = max(1e-9, med * jump_factor)
        rows  = [pts[0]]
        for k in range(1, len(pts)):
            if dists[k - 1] > thr:
                rows.append([np.nan, np.nan])
            rows.append(pts[k])
        pts = np.array(rows, dtype=float)
    return pts


# ---------------------------------------------------------------------------
# Figure 1 — candidates + trajectories overview
# ---------------------------------------------------------------------------

def plot_candidates_and_paths(J, p_cand, p_sink, R_comm, mob_names,
                               r_mobile_fns, T, region, out_path="pic_candidates.png"):
    fig, ax = plt.subplots(figsize=(10, 7))
    for j in J:
        q = p_cand[j]
        ax.add_patch(Circle((q[0], q[1]), R_comm, fill=False,
                             linewidth=1, ls="--", edgecolor="gray"))
        ax.scatter([q[0]], [q[1]], marker="s", s=S_FIXED0, c=COLOR_CANDIDATE)
    ax.scatter([p_sink[0]], [p_sink[1]], marker="*", s=S_SINK,
               c=COLOR_SINK, label="sink")
    ax.add_patch(Circle((p_sink[0], p_sink[1]), R_comm, fill=False,
                         linewidth=1, ls="--", edgecolor=COLOR_SINK, alpha=0.6))
    for name in mob_names:
        traj = _traj_with_breaks(r_mobile_fns[name], T)
        ax.plot(traj[:, 0], traj[:, 1], ls="--", lw=2, alpha=0.6,
                c=COLOR_MOBILE)
        ax.scatter(traj[:, 0], traj[:, 1], marker="o", s=S_MOBILE,
                   c=COLOR_MOBILE, alpha=0.7)
    ax.set_title(f"Candidates, sink and mobile trajectories  (R_comm={R_comm})")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — installed graph (static relays only)
# ---------------------------------------------------------------------------

def plot_installed_graph(installed, p_cand, p_sink, R_comm, region,
                          out_path="pic_installed_graph.png"):
    """Plot the static relay graph: installed nodes + edges within R_comm."""
    fig, ax = plt.subplots(figsize=(10, 7))

    def _dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    # Edges between installed nodes
    for i in range(len(installed)):
        for j in range(i + 1, len(installed)):
            pa, pb = p_cand[installed[i]], p_cand[installed[j]]
            if _dist(pa, pb) <= R_comm + 1e-9:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], lw=2, c="red")
    # Sink ↔ installed edges
    for j in installed:
        p = p_cand[j]
        if _dist(p_sink, p) <= R_comm + 1e-9:
            ax.plot([p_sink[0], p[0]], [p_sink[1], p[1]], lw=2, c="red")
    # Installed nodes
    for j in installed:
        q = p_cand[j]
        ax.scatter([q[0]], [q[1]], marker="s", s=S_FIXED1, c=COLOR_FIXED1)
    # Sink
    ax.scatter([p_sink[0]], [p_sink[1]], marker="*", s=S_SINK, c=COLOR_SINK)

    ax.set_title(f"Installed relay graph  (edges ≤ R_comm={R_comm})")
    ax.axis("equal")
    ax.grid(True)
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — sweep summary scatter
# ---------------------------------------------------------------------------

def plot_sweep_summary(records: list[dict], out_path: str = "sweep_summary.png"):
    """Scatter plot of (B, installed_nodes) coloured by C0 for each kdecay."""
    solved = [r for r in records if r.get("installed_nodes") is not None]
    if not solved:
        return

    kdecay_vals = sorted({r["k_decay"] for r in solved})
    fig, axes = plt.subplots(1, len(kdecay_vals),
                              figsize=(5 * len(kdecay_vals), 4), sharey=True)
    if len(kdecay_vals) == 1:
        axes = [axes]

    c0_all   = sorted({r["C0"] for r in solved})
    cmap     = plt.cm.viridis
    norm     = plt.Normalize(vmin=min(c0_all), vmax=max(c0_all))

    for ax, kd in zip(axes, kdecay_vals):
        sub = [r for r in solved if r["k_decay"] == kd]
        sc  = ax.scatter(
            [r["B"] for r in sub],
            [r["installed_nodes"] for r in sub],
            c=[r["C0"] for r in sub], cmap=cmap, norm=norm,
            s=30, alpha=0.8,
        )
        ax.set_title(f"k_decay = {kd}")
        ax.set_xlabel("B (demand per mobile)")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("installed relays")
    fig.colorbar(sc, ax=axes[-1], label="C0")
    fig.suptitle("P2 MILP sweep — installed relays vs B", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
