#!/usr/bin/env python3
"""
gsRQ4_camera_ready.py

Camera-ready RQ4 figure for GrayShield.

Panels:
(a) Pareto scatter: RR vs |Delta Acc|
(b) Deployment criteria satisfaction
(c) Runtime overhead
(d) Weight-distribution fidelity
(e) Deployment radar

Input:
    Uses summary values from the GrayShield statistical analysis.

Output:
    figures/rq4_camera_ready_tradeoff.png
    figures/rq4_camera_ready_tradeoff.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Output
# ============================================================

os.makedirs("figures", exist_ok=True)


# ============================================================
# Constants and summary data
# ============================================================

COLORS = {
    "GS":  "#1a6faf",
    "PM":  "#e07b39",
    "PTQ": "#6aaf3d",
    "FT":  "#9b59b6",
    "GN":  "#e74c3c",
    "SWP": "#f39c12",
    "RF":  "#95a5a6",
}

DEFENSES = ["GS", "PM", "PTQ", "FT", "GN", "SWP", "RF"]

# mean_RR, std_RR, delta_acc, runtime_ms
RQ4_TABLE = {
    "GS":  (49.96, 0.66, -0.098,    54),
    "PM":  (50.00, 3.25,  0.134,    44),
    "PTQ": (49.83, 0.54,  0.166,   882),
    "FT":  (45.51, 1.23,  0.449, 13221),
    "GN":  (38.25, 5.85, -0.037,   620),
    "SWP": ( 8.76, 2.56,  0.068,  1483),
    "RF":  ( 1.73, 3.16,  0.018,  2590),
}

# Wasserstein-1 weight fidelity
W1 = {
    "GS": 2.80e-5,
    "PM": 7.10e-4,
    "PTQ": 1.17e-3,
    "FT": 4.00e-6,
    "GN": 4.00e-6,
    "SWP": 4.70e-5,
    "RF": 4.00e-6,
}


# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.6,
    "legend.fontsize": 5.9,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.50,
})


# ============================================================
# Helpers
# ============================================================

def style_ax(ax, grid_axis="y"):
    ax.set_facecolor("#f8f9fa")
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, alpha=0.22, zorder=0)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, alpha=0.22, zorder=0)
    ax.set_axisbelow(True)


def pareto_mask(rr, acc):
    """Pareto-efficient mask for maximizing RR and minimizing |Delta Acc|."""
    rr = np.asarray(rr, dtype=float)
    acc = np.asarray(acc, dtype=float)
    efficient = np.ones(len(rr), dtype=bool)

    for i in range(len(rr)):
        dominates_i = (
            (rr >= rr[i]) &
            (acc <= acc[i]) &
            ((rr > rr[i]) | (acc < acc[i]))
        )
        if np.any(dominates_i):
            efficient[i] = False

    return efficient


def criteria_scores():
    """Four deployment criteria: high RR, stable RR, low accuracy loss, fast runtime."""
    scores = {}
    for d in DEFENSES:
        rr, sig, da, t_ms = RQ4_TABLE[d]
        scores[d] = (
            int(rr >= 48.0) +
            int(sig < 2.0) +
            int(abs(da) < 1.0) +
            int(t_ms < 500)
        )
    return scores


def radar_scores():
    """
    Normalize metrics to [0, 100], where 100 is best.

    Axes:
      RR        : higher better
      Stability : lower sigma_RR better
      Utility   : lower |Delta Acc| better
      Speed     : lower runtime better, log-normalized
      Fidelity  : lower W1 better, log-normalized
    """
    rr = np.array([RQ4_TABLE[d][0] for d in DEFENSES], dtype=float)
    sig = np.array([RQ4_TABLE[d][1] for d in DEFENSES], dtype=float)
    da = np.array([abs(RQ4_TABLE[d][2]) for d in DEFENSES], dtype=float)
    t_ms = np.array([RQ4_TABLE[d][3] for d in DEFENSES], dtype=float)
    w1 = np.array([W1[d] for d in DEFENSES], dtype=float)

    s_rr = rr / 50.0 * 100.0
    s_rr = np.clip(s_rr, 0, 100)

    s_sig = (1 - sig / sig.max()) * 100.0
    s_da = (1 - da / da.max()) * 100.0

    log_t = np.log10(t_ms)
    s_t = (1 - (log_t - log_t.min()) / (log_t.max() - log_t.min())) * 100.0

    log_w = np.log10(w1)
    s_w1 = (1 - (log_w - log_w.min()) / (log_w.max() - log_w.min())) * 100.0

    return {
        d: [s_rr[i], s_sig[i], s_da[i], s_t[i], s_w1[i]]
        for i, d in enumerate(DEFENSES)
    }


def draw_radar(ax, scores, show_legend=True):
    labels = [
        "RR",
        "",#Stability\n(1-σ)
        "Utility\n(-|ΔAcc|)",
        "Speed\n(-log t)",
        "Fidelity\n(-log W₁)",
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    ax.text(
        angles[1],      # Stability axis
        97,             # ← move inward (tune between 88–95)
        "Stability (1-σ)",
        ha="center",
        va="center",
        fontsize=6.0
    )


    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=5.3, color="#666")
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#f8f9fa")


    for d in DEFENSES:
        vals = scores[d]
        closed = vals + vals[:1]
        ax.plot(angles, closed, color=COLORS[d], lw=1.30, alpha=0.88, label=d)
        ax.fill(angles, closed, color=COLORS[d], alpha=0.045)
        ax.scatter(angles[:-1], vals, color=COLORS[d], s=14, zorder=4)

    ax.set_title("(e) Deployment radar\n100 = best per axis",
                 fontweight="bold", pad=13)

    if show_legend:
        radar_handles = [
            Line2D([0], [0], color=COLORS[d], lw=1.8, label=d)
            for d in DEFENSES
        ]
        ax.legend(
            handles=radar_handles,
            title="Defense",
            loc="upper right",
            bbox_to_anchor=(1.05, 0.25),  # ← tighter and closer
            fontsize=5.6,
            title_fontsize=5.8,
            frameon=True,
            framealpha=0.90,
            handlelength=1.4,
            borderpad=0.35,
            labelspacing=0.25,
        )


# ============================================================
# Figure
# ============================================================

fig = plt.figure(figsize=(18.0, 4.35), constrained_layout=False)

gs = fig.add_gridspec(
    1, 5,
    width_ratios=[1.20, 0.88, 1.00, 1.00, 1.18],
    wspace=0.38,
)

# fig.suptitle(
#     "RQ4 — Pareto Trade-off and Deployment Profile",
#     fontsize=11.6,
#     fontweight="bold",
#     y=0.985,
# )


# ============================================================
# (a) Pareto scatter
# ============================================================

ax_a = fig.add_subplot(gs[0, 0])

rr_vals = np.array([RQ4_TABLE[d][0] for d in DEFENSES])
sig_vals = np.array([RQ4_TABLE[d][1] for d in DEFENSES])
acc_vals = np.array([abs(RQ4_TABLE[d][2]) for d in DEFENSES])

efficient = pareto_mask(rr_vals, acc_vals)

ax_a.axvspan(48, 52, color="#27ae60", alpha=0.075)
ax_a.axhspan(0, 1.0, color="#27ae60", alpha=0.075)
ax_a.axvline(48, color="#2980b9", lw=0.85, ls=":", alpha=0.85, label="RR≥48%")
ax_a.axhline(1.0, color="#e74c3c", lw=0.85, ls=":", alpha=0.85, label="|ΔAcc|≤1%")

for i, d in enumerate(DEFENSES):
    size = 58 + 26 * min(sig_vals[i], 6)
    edge = "#111" if efficient[i] else "#444"
    lw = 1.35 if efficient[i] else 0.55
    ax_a.scatter(
        rr_vals[i],
        acc_vals[i],
        s=size,
        color=COLORS[d],
        edgecolors=edge,
        linewidths=lw,
        alpha=0.88,
        zorder=5 if efficient[i] else 3,
    )
    ax_a.annotate(
        d,
        (rr_vals[i], acc_vals[i]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=6.4,
        fontweight="bold",
        color=COLORS[d],
    )

frontier = sorted(
    [(rr_vals[i], acc_vals[i]) for i in range(len(DEFENSES)) if efficient[i]],
    key=lambda x: x[0],
)
if len(frontier) >= 2:
    ax_a.plot(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        color="#111",
        lw=0.90,
        alpha=0.72,
        label="Pareto frontier",
    )

ax_a.set_xlabel("Recovery Reduction (RR) %")
ax_a.set_ylabel(r"$|\Delta Acc|$ (%)")
ax_a.set_title("(a) Pareto scatter\nhigh RR, low utility loss", fontweight="bold")
ax_a.text(
    0.03, 0.97,
    "marker size ∝ σRR",
    transform=ax_a.transAxes,
    ha="left",
    va="top",
    fontsize=5.7,
    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#bbbbbb", alpha=0.90),
)
ax_a.legend(fontsize=5.5, frameon=True, loc="upper left", bbox_to_anchor=(0.02, 0.74))
style_ax(ax_a, grid_axis="both")


# ============================================================
# (b) Deployment criteria
# ============================================================

ax_b = fig.add_subplot(gs[0, 1])

scores = criteria_scores()
order = sorted(DEFENSES, key=lambda d: scores[d])
y = np.arange(len(order))

ax_b.barh(
    y,
    [scores[d] for d in order],
    color=[COLORS[d] for d in order],
    alpha=0.86,
    zorder=3,
)

for i, d in enumerate(order):
    ax_b.text(
        scores[d] + 0.06,
        i,
        f"{scores[d]}/4",
        va="center",
        fontsize=6.9,
        fontweight="bold",
    )

ax_b.axvline(4, color="#555", ls="--", lw=0.85, alpha=0.75, label="max=4")
ax_b.set_yticks(y)
ax_b.set_yticklabels(order)
ax_b.set_xlim(0, 4.7)
ax_b.set_xlabel("Criteria satisfied")
ax_b.set_title("(b) Deployment criteria\nbinary constraints", fontweight="bold")
ax_b.legend(fontsize=5.6, frameon=True, loc="lower right")
style_ax(ax_b, grid_axis="x")


# ============================================================
# (c) Runtime
# ============================================================

ax_c = fig.add_subplot(gs[0, 2])

times = np.array([RQ4_TABLE[d][3] for d in DEFENSES], dtype=float)
log_t = np.log10(times)

bars = ax_c.bar(
    DEFENSES,
    log_t,
    color=[COLORS[d] for d in DEFENSES],
    alpha=0.86,
    width=0.62,
    zorder=3,
)

ax_c.axhline(np.log10(500), color="#e74c3c", ls="--", lw=0.90, alpha=0.85, label="500 ms")

for bar, t_ms, lt in zip(bars, times, log_t):
    label = f"{t_ms:.0f} ms" if t_ms < 1000 else f"{t_ms/1000:.1f}s"
    ax_c.text(
        bar.get_x() + bar.get_width() / 2,
        lt + 0.045,
        label,
        ha="center",
        va="bottom",
        fontsize=5.8,
        rotation=14,
    )

ax_c.set_ylabel(r"$\log_{10}$(runtime ms)")
ax_c.set_title("(c) Runtime overhead\nlower is better", fontweight="bold")
ax_c.legend(fontsize=5.7, frameon=True, loc="upper left")
style_ax(ax_c)


# ============================================================
# (d) Weight-distribution fidelity
# ============================================================

ax_d = fig.add_subplot(gs[0, 3])

fid = np.array([-np.log10(W1[d]) for d in DEFENSES])

bars = ax_d.bar(
    DEFENSES,
    fid,
    color=[COLORS[d] for d in DEFENSES],
    alpha=0.86,
    width=0.62,
    zorder=3,
)

for d in ["PM", "PTQ"]:
    i = DEFENSES.index(d)
    ratio = W1[d] / max(W1["GS"], 1e-30)
    ax_d.text(
        i,
        fid[i] + 0.08,
        f"×{ratio:.0f}\nvs GS",
        ha="center",
        va="bottom",
        fontsize=5.9,
        color="#c0392b",
        fontweight="bold",
    )

ax_d.set_ylabel(r"$-\log_{10}(W_1)$")
ax_d.set_title("(d) Weight fidelity\nhigher is better", fontweight="bold")
style_ax(ax_d)


# ============================================================
# (e) Radar
# ============================================================

ax_e = fig.add_subplot(gs[0, 4], polar=True)
draw_radar(ax_e, radar_scores(), show_legend=True)


# Shared legend for defenses
handles = [Line2D([0], [0], color=COLORS[d], lw=2.0, label=d) for d in DEFENSES]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.50, 0.925),
    ncol=7,
    frameon=True,
    fontsize=5.9,
    columnspacing=0.95,
    handlelength=1.45,
)


# ============================================================
# Save
# ============================================================

fig.subplots_adjust(
    left=0.045,
    right=0.985,
    top=0.80,
    bottom=0.18,
    wspace=0.38,
)

out_png = "figures/rq4_camera_ready_tradeoff.png"
out_pdf = "figures/rq4_camera_ready_tradeoff.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="#f8f9fa")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="#f8f9fa")
plt.close(fig)

print(f"✓ Saved: {out_png}")
print(f"✓ Saved: {out_pdf}")
