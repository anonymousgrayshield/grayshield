#!/usr/bin/env python3
"""
gsRQ3_camera_ready.py

Final camera-ready compact RQ3 figure in a 1 × 4 layout.

Panels:
(a) Recovery Reduction across attacker variants
(b) Recovery Reduction distribution by defense
(c) Effectiveness–stability trade-off
(d) Attacker × defense RR heatmap

Input:
    datasets/rq3.jsonl

Output:
    figures/rq3_camera_ready_adaptive_robustness.png
    figures/rq3_camera_ready_adaptive_robustness.pdf
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

BASE = Path("datasets")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RQ3_PATH = BASE / "rq3.jsonl"

DEFENSES = [
    "GrayShield",
    "PatternMask",
    "PTQ",
    "FineTune",
    "GaussianNoise",
    "SWP",
    "RandomFlip",
]

SHORT = {
    "GrayShield": "GS",
    "PatternMask": "PM",
    "PTQ": "PTQ",
    "FineTune": "FT",
    "GaussianNoise": "GN",
    "SWP": "SWP",
    "RandomFlip": "RF",
}

PAL = {
    "GrayShield": "#e45756",
    "PatternMask": "#ffcf33",
    "PTQ": "#8d6e63",
    "FineTune": "#59a14f",
    "GaussianNoise": "#78909c",
    "SWP": "#26c6da",
    "RandomFlip": "#2f7ed8",
}

MAP = {
    "GrayShield": "GrayShield", "gray": "GrayShield", "grayshield": "GrayShield",
    "PatternMask": "PatternMask", "pattern": "PatternMask", "patternmask": "PatternMask",
    "PTQ": "PTQ", "ptq": "PTQ",
    "FineTune": "FineTune", "finetune": "FineTune", "fine_tune": "FineTune",
    "GaussianNoise": "GaussianNoise", "gaussian": "GaussianNoise", "gaussiannoise": "GaussianNoise",
    "SWP": "SWP", "swp": "SWP",
    "RandomFlip": "RandomFlip", "random": "RandomFlip", "randomflip": "RandomFlip",
}

ATTACKER_ORDER = ["naive", "interleave", "repeat3", "repeat5", "rs"]
ATTACKER_LABELS = {
    "naive": "Naïve",
    "interleave": "Interleave",
    "repeat3": "Repeat×3",
    "repeat5": "Repeat×5",
    "rs": "RS",
    "rs255127": "RS",
    "RS(255,127)": "RS",
}


def norm_strategy(x):
    s = str(x)
    return MAP.get(s, MAP.get(s.lower(), s))


def norm_attacker(x):
    low = str(x).lower()
    if "inter" in low:
        return "interleave"
    if "repeat3" in low or "rep3" in low:
        return "repeat3"
    if "repeat5" in low or "rep5" in low:
        return "repeat5"
    if "rs" in low or "reed" in low:
        return "rs"
    if "naive" in low or low == "none":
        return "naive"
    return low


def first(d, keys, default=np.nan):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def to_pct(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    return s * 100 if s.dropna().abs().max() <= 1.5 else s


records = [json.loads(line) for line in open(RQ3_PATH, encoding="utf-8") if line.strip()]
rows = []

for obj in records:
    points = obj.get("points", [obj]) if isinstance(obj.get("points"), list) else [obj]

    for p in points:
        strategy_raw = first(
            p, ["strategy", "defense"],
            default=first(obj, ["strategy", "defense"], default="")
        )
        attacker_raw = first(
            p, ["attacker_variant"],
            default=first(obj, ["attacker_variant"], default="naive")
        )

        rows.append({
            "strategy": norm_strategy(strategy_raw),
            "attacker": norm_attacker(attacker_raw),
            "rr": first(p, ["recovery_reduction_strict", "recovery_reduction"]),
            "post_recovery": first(p, ["post_recovery_strict", "post_recovery"]),
            "acc": first(p, ["acc_drop", "accuracy_drop"]),
            "w1": first(p, ["wasserstein_distance", "w1"]),
            "runtime_ms": first(p, ["defense_time_ms", "runtime_ms"]),
        })

df = pd.DataFrame(rows)

for c in ["rr", "post_recovery", "acc", "w1", "runtime_ms"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["rr_pct"] = to_pct(df["rr"])
df = df[df["strategy"].isin(DEFENSES)].copy()

observed_attackers = [a for a in ATTACKER_ORDER if a in set(df["attacker"].dropna())]
for a in sorted(set(df["attacker"].dropna())):
    if a not in observed_attackers:
        observed_attackers.append(a)

attacker_labels = [ATTACKER_LABELS.get(a, a) for a in observed_attackers]

mat = (
    df.pivot_table(index="attacker", columns="strategy", values="rr_pct", aggfunc="mean")
      .reindex(index=observed_attackers, columns=DEFENSES)
)

mean_rr = mat.mean(axis=0)
std_rr = mat.std(axis=0, ddof=1)

friedman_text = ""
try:
    complete = mat.dropna(axis=1).dropna(axis=0)
    if complete.shape[0] >= 2 and complete.shape[1] >= 2:
        fried = stats.friedmanchisquare(*[complete[c].values for c in complete.columns])
        friedman_text = f"Friedman p={fried.pvalue:.3g}"
except Exception:
    friedman_text = ""

gs_text = ""
try:
    gs_vals = mat["GrayShield"].dropna().values
    if len(gs_vals) >= 2:
        ttest = stats.ttest_1samp(gs_vals, 50.0)
        decision = "fail to reject" if ttest.pvalue >= 0.05 else "reject"
        gs_text = f"GS vs 50%: p={ttest.pvalue:.3f} ({decision} H0)"
except Exception:
    pass

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.titlesize": 8.1,
    "axes.labelsize": 7.5,
    "legend.fontsize": 5.8,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.50,
})

fig = plt.figure(figsize=(16.0, 4.2), constrained_layout=False)
gs = fig.add_gridspec(
    1, 4,
    width_ratios=[1.35, 0.95, 1.0, 1.10],
    wspace=0.34,
)

# fig.suptitle(
#     "RQ3 — Adaptive Attacker Robustness",
#     fontsize=11.6,
#     fontweight="bold",
#     y=0.985,
# )

# (a) RR trends across attacker variants
ax_a = fig.add_subplot(gs[0, 0])
x = np.arange(len(observed_attackers))

for s in DEFENSES:
    y = mat[s].values
    ax_a.plot(
        x, y,
        marker="o",
        lw=1.70 if s == "GrayShield" else 1.05,
        ms=4.4 if s == "GrayShield" else 3.4,
        color=PAL[s],
        label=SHORT[s],
        alpha=0.98 if s == "GrayShield" else 0.72,
        zorder=5 if s == "GrayShield" else 2,
    )

ax_a.axhline(50, color="#555", lw=0.85, ls="--", alpha=0.82, label="50% target")
ax_a.fill_between(
    [-0.35, len(x)-0.65],
    48, 52,
    color="#2ecc71",
    alpha=0.08,
    label="±2% band",
)

ax_a.set_xticks(x)
ax_a.set_xticklabels(attacker_labels, rotation=14, ha="right")
ax_a.set_ylabel("Recovery Reduction (RR) %")
ax_a.set_title("(a) RR across attacker variants", fontweight="bold")
ax_a.set_ylim(-2, 54)
ax_a.grid(True, axis="y", alpha=0.23)

# (b) Distribution by defense
ax_b = fig.add_subplot(gs[0, 1])

box_data = [df[df["strategy"] == s]["rr_pct"].dropna().values for s in DEFENSES]
bp = ax_b.boxplot(
    box_data,
    labels=[SHORT[s] for s in DEFENSES],
    patch_artist=True,
    widths=0.50,
    medianprops=dict(color="#222", lw=1.1),
    boxprops=dict(linewidth=0.7),
    whiskerprops=dict(linewidth=0.7),
    capprops=dict(linewidth=0.7),
    flierprops=dict(marker="o", markersize=2.2, alpha=0.35),
)

for patch, s in zip(bp["boxes"], DEFENSES):
    patch.set_facecolor(PAL[s])
    patch.set_alpha(0.65)

ax_b.axhline(50, color="#555", lw=0.85, ls="--", alpha=0.82)
subtitle = f"\n{friedman_text}" if friedman_text else ""
ax_b.set_ylabel("RR (%)")
ax_b.set_title("(b) Defense distributions" + subtitle, fontweight="bold")
ax_b.grid(True, axis="y", alpha=0.23)

# (c) Effectiveness–stability trade-off
ax_c = fig.add_subplot(gs[0, 2])

ax_c.axvspan(48, 52, color="#2ecc71", alpha=0.08)
ax_c.axhspan(0, 2, color="#2ecc71", alpha=0.08)
ax_c.axhline(2, color="#555", ls="--", lw=0.85, alpha=0.78)
ax_c.axvline(50, color="#555", ls="--", lw=0.85, alpha=0.78)

for s in DEFENSES:
    ax_c.scatter(
        mean_rr[s],
        std_rr[s],
        s=110 if s == "GrayShield" else 66,
        color=PAL[s],
        edgecolors="#222",
        linewidths=0.70,
        zorder=5 if s == "GrayShield" else 3,
    )
    ax_c.annotate(
        SHORT[s],
        (mean_rr[s], std_rr[s]),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=6.2,
        fontweight="bold",
        color=PAL[s],
    )

ax_c.set_xlabel("Mean RR (%)")
ax_c.set_ylabel(r"$\sigma_{RR}$ (%)")
ax_c.set_title("(c) Effectiveness–stability\nideal = 50% RR, low σ", fontweight="bold")
ax_c.grid(True, alpha=0.23)

if gs_text:
    ax_c.text(
        0.03, 0.97,
        gs_text,
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=PAL["GrayShield"], alpha=0.93),
    )

# (d) Attacker × defense heatmap
ax_d = fig.add_subplot(gs[0, 3])

heat = mat[DEFENSES].values
im = ax_d.imshow(heat, aspect="auto", vmin=0, vmax=52, cmap="YlGnBu")

ax_d.set_xticks(np.arange(len(DEFENSES)))
ax_d.set_xticklabels([SHORT[s] for s in DEFENSES])
ax_d.set_yticks(np.arange(len(observed_attackers)))
ax_d.set_yticklabels(attacker_labels)

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat[i, j]
        if not np.isnan(val):
            ax_d.text(
                j, i, f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if val > 35 else "#222",
            )

ax_d.set_title("(d) Attacker × defense RR heatmap", fontweight="bold")
ax_d.set_xlabel("Defense")
ax_d.set_ylabel("Attacker")

divider = make_axes_locatable(ax_d)
cax = divider.append_axes("right", size="3%", pad=0.045)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("RR (%)", fontsize=6.2)
cbar.ax.tick_params(labelsize=5.6)

# Shared legend
handles, labels = ax_a.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.50, 0.925),
    ncol=9,
    frameon=True,
    fontsize=5.8,
    columnspacing=0.85,
    handlelength=1.4,
)

fig.subplots_adjust(
    left=0.050,
    right=0.982,
    top=0.80,
    bottom=0.19,
    wspace=0.34,
)

out_png = OUT / "rq3_camera_ready_adaptive_robustness.png"
out_pdf = OUT / "rq3_camera_ready_adaptive_robustness.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"✓ Saved: {out_png}")
print(f"✓ Saved: {out_pdf}")
