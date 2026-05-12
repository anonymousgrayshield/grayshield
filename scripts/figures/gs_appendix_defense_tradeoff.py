import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
RENAME  = {"random": "RandomFlip", "pattern": "PatternMask"}
ORDER   = ["GrayShield", "PTQ", "PatternMask", "FineTune", "GaussianNoise", "SWP", "RandomFlip"]
COLORS  = {
    "GrayShield":    "#185FA5",
    "PTQ":           "#1D9E75",
    "PatternMask":   "#EF9F27",
    "FineTune":      "#639922",
    "GaussianNoise": "#888780",
    "SWP":           "#D4537E",
    "RandomFlip":    "#D85A30",
}
MARKERS = {
    "GrayShield":    "*",
    "PTQ":           "X",
    "PatternMask":   "s",
    "FineTune":      "v",
    "GaussianNoise": "D",
    "SWP":           "o",
    "RandomFlip":    "P",
}

# ── Load rq3.jsonl (3 models, 5 attackers, parameter sweep) ───────────────────
rq3_rows = []
with open("./datasets/rq3.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        for pt in rec["points"]:
            strat = RENAME.get(pt["strategy"], pt["strategy"])
            rq3_rows.append({
                "source":   "rq3",
                "model":    rec["model_preset"],
                "strategy": strat,
                "attacker": pt["attacker_variant"],
                "acc_drop": pt["acc_drop"] * 100,
                "rec_red":  pt["recovery_reduction"] * 100,
            })
df3 = pd.DataFrame(rq3_rows)

# ── Load rq2.jsonl — keep only swin_cifar10 (unique to rq2) ───────────────────
rq2_rows = []
with open("./datasets/rq2.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if rec["model_preset"] != "swin_cifar10":
            continue                         # other models already covered by rq3
        m = rec["metrics"]
        rq2_rows.append({
            "source":   "rq2",
            "model":    rec["model_preset"],
            "strategy": rec["defense"]["type"],
            "attacker": rec["attacker_variant"],
            "acc_drop": m["acc_drop_vs_base"] * 100,
            "rec_red":  m["recovery_reduction"] * 100,
        })
df2 = pd.DataFrame(rq2_rows)

# ── Combined dataframe ─────────────────────────────────────────────────────────
df = pd.concat([df3, df2], ignore_index=True)

# Panel A: average over attackers → one dot per (source, model, strategy, config)
scatter = df.groupby(["source", "model", "strategy", "acc_drop"],
                     as_index=False)["rec_red"].mean()

# Panel B: per strategy, config with highest avg rec_red → its acc_drop
best_config = (
    df.groupby(["strategy", "acc_drop"])["rec_red"]
    .mean().reset_index()
    .sort_values("rec_red", ascending=False)
    .drop_duplicates("strategy")
    .set_index("strategy")
    .reindex(ORDER)
)
best_config["acc_drop_plot"] = best_config["acc_drop"]   # keep negatives

acc_range = (
    df.groupby("strategy")["acc_drop"]
    .agg(lo="min", hi="max")          # no clip — show full signed range
    .reindex(ORDER)
)

# ── Linear regression ──────────────────────────────────────────────────────────
x_reg = scatter["acc_drop"].values
y_reg = scatter["rec_red"].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x_reg, y_reg)
x_line = np.linspace(-0.1, 2.0, 300)
y_line = slope * x_line + intercept
n      = len(x_reg)
x_mean = x_reg.mean()
se_band = std_err * np.sqrt(
    1/n + (x_line - x_mean)**2 / np.sum((x_reg - x_mean)**2)
)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(8, 10.5),
    gridspec_kw={"height_ratios": [3, 1.8], "hspace": 0.52}
)
fig.patch.set_facecolor("white")
for a in (ax, ax2):
    a.set_facecolor("#F8F8F8")
    a.grid(True, color="white", linewidth=0.8, zorder=0)
    a.spines[["top", "right"]].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# Panel A
# ══════════════════════════════════════════════════════════════════════════════
ax.axvline(0.25, color="#999999", linewidth=1.2, linestyle="--", zorder=1)
ax.axhline(50,   color="#999999", linewidth=1.2, linestyle="--", zorder=1)
ax.text(0.26,  -6,   "0.25% accuracy\nbudget",   fontsize=7.5, color="#888888", va="bottom")
ax.text(-0.08, 50.8, "50% recovery\nthreshold",  fontsize=7.5, color="#888888", va="bottom")

ax.plot(x_line, y_line, color="#444444", linewidth=1.5, zorder=2,
        label=f"Linear fit  (R²={r_value**2:.2f})")
ax.fill_between(x_line,
                y_line - 1.96 * se_band * np.sqrt(n),
                y_line + 1.96 * se_band * np.sqrt(n),
                color="#AAAAAA", alpha=0.12, zorder=1, label="95% CI")

# rq3 dots: solid fill; rq2/swin dots: hollow (white face, colored edge)
for strat in ORDER:
    sub3 = scatter[(scatter["strategy"] == strat) & (scatter["source"] == "rq3")]
    sub2 = scatter[(scatter["strategy"] == strat) & (scatter["source"] == "rq2")]

    if not sub3.empty:
        ax.scatter(sub3["acc_drop"], sub3["rec_red"],
                   s=70, marker=MARKERS[strat],
                   color=COLORS[strat], edgecolors="white",
                   linewidths=0.4, zorder=5)
    if not sub2.empty:
        ax.scatter(sub2["acc_drop"], sub2["rec_red"],
                   s=90, marker=MARKERS[strat],
                   facecolors="white", edgecolors=COLORS[strat],
                   linewidths=1.4, zorder=5)

ax.text(0.27, 63, "Ideal zone", fontsize=8, color="#185FA5", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E6F1FB", ec="none", alpha=0.6))

ax.set_xlim(-0.15, 2.0)
ax.set_ylim(-8, 70)
ax.set_xlabel("Accuracy drop (%)", fontsize=11)
ax.set_ylabel("Recovery reduction (%)", fontsize=11)
ax.set_title(
    "Panel (a) — Trade-off: recovery reduction vs. accuracy cost\n"
    "(rq3: 3 models × 5 attackers  |  rq2 hollow: swin_cifar10, naive attacker only)",
    fontsize=10.5, pad=10
)

# Legend: defense shapes + source markers
legend_handles = [
    Line2D([0], [0], marker=MARKERS[s], color="w",
           markerfacecolor=COLORS[s], markersize=8,
           markeredgewidth=0.4, markeredgecolor="white", label=s)
    for s in ORDER
]
src_rq3 = Line2D([0], [0], marker="o", color="w",
                 markerfacecolor="#777777", markersize=7,
                 markeredgecolor="white", label="rq3 (solid)")
src_rq2 = Line2D([0], [0], marker="o", color="w",
                 markerfacecolor="white", markersize=7,
                 markeredgecolor="#777777", markeredgewidth=1.4,
                 label="rq2/swin (hollow)")
reg_line = Line2D([0], [0], color="#444444", linewidth=1.5,
                  label=f"Linear fit  (R²={r_value**2:.2f})")
ci_patch = mpatches.Patch(color="#AAAAAA", alpha=0.4, label="95% CI")
ax.legend(handles=legend_handles + [src_rq3, src_rq2, reg_line, ci_patch],
          fontsize=8, framealpha=0.9, loc="lower right", ncol=2, columnspacing=1)

# ══════════════════════════════════════════════════════════════════════════════
# Panel B
# ══════════════════════════════════════════════════════════════════════════════
bar_names  = best_config.index.tolist()
bar_drops  = best_config["acc_drop_plot"].values
bar_colors = [COLORS[n] for n in bar_names]

xerr_lo = bar_drops - acc_range.loc[bar_names, "lo"].values
xerr_hi = acc_range.loc[bar_names, "hi"].values - bar_drops

ax2.barh(bar_names, bar_drops, color=bar_colors,
         edgecolor="white", linewidth=0.6, height=0.5, zorder=3)
ax2.errorbar(bar_drops, bar_names,
             xerr=[bar_drops - acc_range.loc[bar_names, "lo"].values,
                   acc_range.loc[bar_names, "hi"].values - bar_drops],
             fmt="none", ecolor="#555555", elinewidth=1.2,
             capsize=4, capthick=1.2, zorder=4)

# Value labels: place to the right of the rightmost whisker cap
for i, (val, name) in enumerate(zip(bar_drops, bar_names)):
    label = f"{val:+.3f}%" if val != 0 else "0.000%"
    hi = acc_range.loc[name, "hi"]
    ax2.text(hi + 0.04, i, label, va="center", fontsize=8.5,
             color=COLORS[name], fontweight="bold")

# Zero baseline — separates "costs accuracy" (right) from "improves accuracy" (left)
ax2.axvline(0,    color="#444444", linewidth=1.0, linestyle="-",  zorder=2)
ax2.axvline(0.25, color="#999999", linewidth=1.2, linestyle="--", zorder=2)
ax2.text( 0.26, -0.65, "0.25% budget",        fontsize=7.5, color="#888888", va="top")
ax2.text(-0.02, -0.65, "← improves accuracy", fontsize=7.5, color="#444444",
         va="top", ha="right")

ax2.set_xlim(-0.55, 2.3)
ax2.set_xlabel("Accuracy drop (%)  —  negative = accuracy improvement", fontsize=11)
ax2.set_title(
    "Panel (b) — Accuracy cost at best operating point  \n"
    "(bar = at max recovery reduction;  whiskers = full range across all configs)",
    fontsize=10.5, pad=10
)
ax2.tick_params(axis="y", labelsize=9)
for tick_label, name in zip(ax2.get_yticklabels(), bar_names):
    tick_label.set_color(COLORS[name])
    tick_label.set_fontweight("bold")

plt.savefig("images/defense_tradeoff_neg.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → defense_tradeoff_neg.png")


