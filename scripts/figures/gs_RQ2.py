#!/usr/bin/env python3
"""
RQ2 — Defense Effectiveness: 6-panel figure (3 columns × 2 rows).

Panel layout
────────────────────────────────────────────────────────────
Row 0  │ (a) RR by payload      │ (b) Accuracy Drop     │ (c) CDF of RR      │
Row 1  │ (d) Radar chart        │ (e) Z-score heatmap   │ (f) Grouped bar    │
────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path("datasets")
OUT  = Path("figures")
OUT.mkdir(exist_ok=True)

# ── strategy ordering & palette ───────────────────────────────────────────────
STRAT        = ["RandomFlip", "SWP", "GaussianNoise", "FineTune", "PTQ", "PatternMask", "GrayShield"]
STRAT_SORTED = ["GrayShield", "PatternMask", "PTQ", "FineTune", "GaussianNoise", "SWP", "RandomFlip"]

SHORT = {
    "GrayShield": "GS",  "PatternMask": "PM",  "PTQ":       "PTQ",
    "FineTune":   "FT",  "GaussianNoise": "GN", "SWP":       "SWP",
    "RandomFlip": "RF",
}
PAL = {
    "GrayShield":   "#e45756", "PatternMask": "#f5a623",
    "PTQ":          "#8d6e63", "FineTune":    "#59a14f",
    "GaussianNoise":"#78909c", "SWP":         "#26c6da",
    "RandomFlip":   "#2f7ed8",
}
HATCH = {"Low-entropy": "", "High-entropy": "///"}

# ── helpers ───────────────────────────────────────────────────────────────────
def safe_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

def norm_score(vals, invert=False):
    vals = np.asarray(vals, dtype=float)
    mn, mx = np.nanmin(vals), np.nanmax(vals)
    out = np.ones_like(vals) * 0.5 if np.isclose(mx, mn) else (vals - mn) / (mx - mn)
    return 1 - out if invert else out

def classify_payload(payload):
    if isinstance(payload, dict):
        for key in ("normalized_entropy", "entropy", "norm_entropy"):
            if key in payload and payload[key] is not None:
                try:
                    return "High-entropy" if float(payload[key]) >= 0.85 else "Low-entropy"
                except Exception:
                    pass
        sha = str(payload.get("sha256", ""))
        if sha.startswith("5704"):  return "High-entropy"
        if sha.startswith("c37c"):  return "Low-entropy"
        if sha:                     return "Low-entropy"
    s = str(payload).upper()
    if any(k in s for k in ("HE", "HIGH", "EXE")): return "High-entropy"
    if any(k in s for k in ("LE", "LOW",  "JS" )): return "Low-entropy"
    return "Payload"

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def style_ax(ax, grid="y"):
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#f5f6f8")
    ax.set_axisbelow(True)
    if "y" in grid: ax.yaxis.grid(True, alpha=0.30, lw=0.7)
    if "x" in grid: ax.xaxis.grid(True, alpha=0.30, lw=0.7)

# ── load data ─────────────────────────────────────────────────────────────────
rq2_raw  = [json.loads(l) for l in open(BASE/"rq2.jsonl", encoding="utf-8") if l.strip()]

rows, rq2_by_s = [], defaultdict(list)
for r in rq2_raw:
    m   = r["metrics"]
    d   = r["defense"]
    s   = d["type"]
    pt  = classify_payload(r.get("payload", r.get("payload_type", "")))
    rec = dict(
        strategy=s, payload_type=pt,
        rr  = m["recovery_reduction"]  * 100,
        acc = m["acc_drop_vs_base"]    * 100,   # signed
        w1  = m["wasserstein_distance"],
        hamming  = m["hamming_distance"],
        time_ms  = r["timing"]["defense_seconds"] * 1000,
    )
    rows.append(rec)
    rq2_by_s[s].append(rec)

df = pd.DataFrame(rows)

# ── per-strategy summary ──────────────────────────────────────────────────────
summary = pd.DataFrame(index=STRAT_SORTED)
for col, fn in [("rr","rr"),("acc","acc_abs"),("time_ms","time_ms"),
                ("w1","w1"),("hamming","hamming")]:
    key = "acc_abs" if col == "acc" else col
    raw = [[abs(r["acc"]) if col == "acc" else r[col] for r in rq2_by_s[s]]
           for s in STRAT_SORTED]
    summary[key]           = [np.nanmean(v) for v in raw]
    summary[key + "_std"]  = [safe_std(v)   for v in raw]
summary["rr"]      = [np.nanmean([r["rr"]      for r in rq2_by_s[s]]) for s in STRAT_SORTED]
summary["rr_std"]  = [safe_std  ([r["rr"]      for r in rq2_by_s[s]]) for s in STRAT_SORTED]

# normalised scores for radar
scores = {
    "RR":       norm_score(summary["rr"].values),
    "Stability":norm_score(summary["rr_std"].values,  invert=True),
    "Utility":  norm_score(summary["acc_abs"].values,  invert=True),
    "Speed":    norm_score(np.log10(np.maximum(summary["time_ms"].values, 1.0)), invert=True),
    "Fidelity": norm_score(np.log10(np.maximum(summary["w1"].values, 1e-12)),    invert=True),
}

# z-scored heatmap matrix
hmap_keys = ["RR", "Stability\n(−σ)", "Utility\n(−|ΔAcc|)",
             "Speed\n(−log t)", "Fidelity\n(−log W₁)", "Hamming\n(norm.)"]
hmap_raw  = np.vstack([
    summary["rr"].values,
    -summary["rr_std"].values,
    -summary["acc_abs"].values,
    -np.log10(np.maximum(summary["time_ms"].values, 1.0)),
    -np.log10(np.maximum(summary["w1"].values, 1e-12)),
    summary["hamming"].values / max(float(summary["hamming"].max()), 1.0),
]).T
hmap_z = np.zeros_like(hmap_raw)
for j in range(hmap_raw.shape[1]):
    col = hmap_raw[:, j]
    sd  = np.nanstd(col)
    hmap_z[:, j] = 0.0 if np.isclose(sd, 0) else (col - np.nanmean(col)) / sd

# one-sample t-tests vs 50% (for panel a annotation)
onesample = {}
for s in STRAT_SORTED:
    vals = np.array([r["rr"] for r in rq2_by_s[s]])
    t, p = stats.ttest_1samp(vals, 50.0)
    onesample[s] = {"t": t, "p": p, "stars": sig_stars(p)}

# ── global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":       8.5,
    "axes.titlesize":  9.5,
    "axes.labelsize":  8.5,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})

FBG = "#f0f1f3"
payloads = ["Low-entropy", "High-entropy"]
PAY_ALPHA  = {"Low-entropy": 0.90, "High-entropy": 0.58}

# ── build figure  ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18.5, 9.4), facecolor="white")
#fig.suptitle("RQ2 — Defense Effectiveness and Multi-Metric Trade-offs", fontsize=13, fontweight="bold", y=0.995)

outer = gridspec.GridSpec(2, 3, figure=fig,
                          left=0.055, right=0.975,
                          top=0.945,  bottom=0.07,
                          wspace=0.38, hspace=0.32)

# ═══════════════════════════════════════════════════════════════════
# ROW 0 — col 0 : (a) Recovery Reduction grouped by payload
# ═══════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(outer[0, 0])

width = 0.33
x     = np.arange(len(STRAT))

for j, payload in enumerate(payloads):
    vals, errs = [], []
    for s in STRAT:
        sub = df[(df.strategy == s) & (df.payload_type == payload)]["rr"].values
        vals.append(np.nanmean(sub) if len(sub) else np.nan)
        errs.append(safe_std(sub))
    offset = (j - 0.5) * width
    ax_a.bar(x + offset, vals, width=width, yerr=errs, capsize=3,
             color=[PAL[s] for s in STRAT], edgecolor="#444", linewidth=0.5,
             alpha=PAY_ALPHA[payload], hatch=HATCH[payload], label=payload,
             error_kw=dict(ecolor="#222", lw=1.0, capthick=1.0), zorder=3)
    for i, (v, e) in enumerate(zip(vals, errs)):
        if not np.isnan(v):
            ax_a.text(i + offset, v + e + 1.4, f"{v:.1f}",
                      ha="center", va="bottom", fontsize=6.8, color="#444")

# significance stars (one-sample t vs 50%) above each strategy
for i, s in enumerate(STRAT):
    stars = onesample[s]["stars"]
    col   = "#c0392b" if onesample[s]["p"] < 0.05 else "#999"
    def _safe_mean(arr):
        return float(np.nanmean(arr)) if len(arr) > 0 else 0.0
    ymax  = max(
        _safe_mean(df[(df.strategy==s)&(df.payload_type=="High-entropy")]["rr"].values),
        _safe_mean(df[(df.strategy==s)&(df.payload_type=="Low-entropy") ]["rr"].values),
    )
    ax_a.text(i, ymax + 4.5, stars, ha="center", fontsize=8.5,
              fontweight="bold", color=col)

ax_a.axhline(50, ls="--", lw=1.3, color="black", alpha=0.45, label="50% chance")
ax_a.axvspan(len(STRAT)-1.5, len(STRAT)-0.5, color="#e45756", alpha=0.07)
ax_a.text(len(STRAT)-1, 55, "", color="#c0392b",
          ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax_a.set_xticks(x)
ax_a.set_xticklabels([SHORT[s] for s in STRAT], rotation=30, ha="right")
ax_a.set_ylabel("Recovery Reduction (%)")
ax_a.set_ylim(-3, 65)
ax_a.set_title("(a)  Recovery Reduction (%)\n(stars = sig. departure from 50%)",
               fontweight="bold")
ax_a.legend(title="Payload type", fontsize=7.2, title_fontsize=8, loc="upper left")
style_ax(ax_a)

# ═══════════════════════════════════════════════════════════════════
# ROW 0 — col 1 : (b) Accuracy Drop
# ═══════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(outer[0, 1])

for j, payload in enumerate(payloads):
    vals, errs = [], []
    for s in STRAT:
        sub = np.abs(df[(df.strategy==s)&(df.payload_type==payload)]["acc"].values)
        vals.append(np.nanmean(sub) if len(sub) else np.nan)
        errs.append(safe_std(sub))
    offset = (j - 0.5) * width
    ax_b.bar(x + offset, vals, width=width, yerr=errs, capsize=3,
             color=[PAL[s] for s in STRAT], edgecolor="#444", linewidth=0.5,
             alpha=PAY_ALPHA[payload], hatch=HATCH[payload], label=payload,
             error_kw=dict(ecolor="#222", lw=1.0, capthick=1.0), zorder=3)
    for i, (v, e) in enumerate(zip(vals, errs)):
        if not np.isnan(v):
            ax_b.text(i + offset, v + e + 0.02, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=6.5, color="#444")

ax_b.axhline(1.0, ls=":", lw=1.4, color="#e74c3c", alpha=0.75, label="1% threshold")
ax_b.axvspan(len(STRAT)-1.5, len(STRAT)-0.5, color="#e45756", alpha=0.07)
ax_b.text(0.03, 0.96, "Lower = less model damage",
          transform=ax_b.transAxes, ha="left", va="top",
          fontsize=7.2, style="italic", color="#666")
ax_b.set_xticks(x)
ax_b.set_xticklabels([SHORT[s] for s in STRAT], rotation=30, ha="right")
ax_b.set_ylabel("Accuracy Drop |ΔAcc| (%)")
ax_b.set_title("(b)  Accuracy Drop (%)\n(mean ± SD per payload type)",
               fontweight="bold")
ax_b.legend(title="Payload type", fontsize=7.2, title_fontsize=8, loc="center left")
style_ax(ax_b)

# ═══════════════════════════════════════════════════════════════════
# ROW 0 — col 2 : (c) CDF of RR
# ═══════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(outer[0, 2])

xg = np.linspace(-10, 105, 600)
for s in STRAT_SORTED:
    vals = np.array([r["rr"] for r in rq2_by_s[s]])
    mu, sd = vals.mean(), max(vals.std(), 0.01)
    cdf = stats.norm.cdf(xg, mu, sd)
    ax_c.plot(xg, cdf, color=PAL[s], lw=2.0, label=SHORT[s])

ax_c.axvline(50, color="#555", lw=1.1, ls="--", alpha=0.75, label="50% chance")
ax_c.fill_betweenx([0, 1], 48, 52, alpha=0.07, color="black")
ax_c.set_xlabel("Recovery Reduction (%)")
ax_c.set_ylabel("Cumulative probability")
ax_c.set_xlim(-10, 105)
ax_c.set_ylim(0, 1.04)
ax_c.set_title("(c)  Cumulative Distribution of RR\n(Normal fit; GS/PM/PTQ stack near 50%)",
               fontweight="bold")
ax_c.legend(fontsize=7, ncol=2, loc="upper left")
style_ax(ax_c, grid="xy")

# ═══════════════════════════════════════════════════════════════════
# ROW 1 — col 0 : (d) Radar / spider chart
# ═══════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(outer[1, 0], projection="polar")

radar_labels = ["Stability\n(1−σ)", "Utility\n(−|ΔAcc|)",
                "Speed\n(−log t)",  "Fidelity\n(−log W₁)", "RR"]
radar_keys   = ["Stability", "Utility", "Speed", "Fidelity", "RR"]
N_ax = len(radar_labels)
angles = np.linspace(0, 2*np.pi, N_ax, endpoint=False).tolist()
angles += angles[:1]

ax_d.set_facecolor("#f5f6f8")
for r_ring in [0.25, 0.50, 0.75, 1.0]:
    ax_d.plot(angles, [r_ring]*(N_ax+1), color="gray", lw=0.5, ls="--", alpha=0.45)

for s in STRAT_SORTED:
    i    = STRAT_SORTED.index(s)
    vals = [scores[k][i] for k in radar_keys] + [scores[radar_keys[0]][i]]
    ax_d.plot(angles, vals, color=PAL[s], lw=2.0, alpha=0.9,  label=SHORT[s])
    ax_d.fill(angles, vals, color=PAL[s], alpha=0.07)

ax_d.set_xticks(angles[:-1])
ax_d.set_xticklabels(radar_labels, fontsize=7.2)
ax_d.set_ylim(0, 1)
ax_d.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_d.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=5.8, color="gray")
ax_d.spines["polar"].set_visible(False)
ax_d.set_title("(d)  Radar Chart\n(normalised deployment profile)",
               fontweight="bold", pad=20)
ax_d.legend(fontsize=6.8, ncol=4, loc="lower left",
            bbox_to_anchor=(-0.18, -0.22))

# ═══════════════════════════════════════════════════════════════════
# ROW 1 — col 1 : (e) Defense × Metric Z-score heatmap
# ═══════════════════════════════════════════════════════════════════
ax_e = fig.add_subplot(outer[1, 1])

im = ax_e.imshow(hmap_z, aspect="auto", vmin=-2.3, vmax=2.3, cmap="RdYlGn")

for i in range(len(STRAT_SORTED)):
    for j in range(len(hmap_keys)):
        v  = hmap_z[i, j]
        fc = "white" if abs(v) > 1.5 else "#222"
        ax_e.text(j, i, f"{v:.1f}", ha="center", va="center",
                  fontsize=6.8, color=fc)

ax_e.set_xticks(range(len(hmap_keys)))
ax_e.set_xticklabels(hmap_keys, fontsize=7.0, rotation=28, ha="right")
ax_e.set_yticks(range(len(STRAT_SORTED)))
ax_e.set_yticklabels([SHORT[s] for s in STRAT_SORTED], fontsize=8)

divider = make_axes_locatable(ax_e)
cax = divider.append_axes("right", size="4%", pad=0.10)
cb  = fig.colorbar(im, cax=cax)
cb.set_label("Z-score", fontsize=7.5)
cb.ax.tick_params(labelsize=7)

ax_e.set_title("(e)  Defense × Metric Z-score Heatmap\n(green = above average)",
               fontweight="bold")

# ═══════════════════════════════════════════════════════════════════
# ROW 1 — col 2 : (f) Grouped bar — RR / |ΔAcc| / Hamming (% of max)
# ═══════════════════════════════════════════════════════════════════
ax_f = fig.add_subplot(outer[1, 2])

rr_pct  = summary["rr"].values     / max(float(summary["rr"].max()),     1e-9) * 100
acc_pct = summary["acc_abs"].values / max(float(summary["acc_abs"].max()),1e-9) * 100
ham_pct = summary["hamming"].values / max(float(summary["hamming"].max()),1e-9) * 100

rr_err  = summary["rr_std"].values     / max(float(summary["rr"].max()),     1e-9) * 100
acc_err = summary["acc_abs_std"].values / max(float(summary["acc_abs"].max()),1e-9) * 100
ham_err = summary["hamming_std"].values / max(float(summary["hamming"].max()),1e-9) * 100

x2 = np.arange(len(STRAT_SORTED))
w2 = 0.24
EK = dict(ecolor="#222", lw=1.0, capthick=1.0)

bars_rr  = ax_f.bar(x2 - w2, rr_pct,  w2, yerr=rr_err,  capsize=3,
                    color="#2980B9", alpha=0.85, label="RR (% of max)",    error_kw=EK, zorder=3)
bars_acc = ax_f.bar(x2,       acc_pct, w2, yerr=acc_err, capsize=3,
                    color="#E74C3C", alpha=0.85, label="|ΔAcc| (% of max)",error_kw=EK, zorder=3)
bars_ham = ax_f.bar(x2 + w2,  ham_pct, w2, yerr=ham_err, capsize=3,
                    color="#2ECC71", alpha=0.85, label="Hamming (% of max)",error_kw=EK, zorder=3)

ax_f.set_xticks(x2)
ax_f.set_xticklabels([SHORT[s] for s in STRAT_SORTED])
ax_f.set_ylabel("% of maximum across defenses")
ax_f.set_title("(f)  Grouped Bar: RR / |ΔAcc| / Hamming\n(mean ± SD, normalised by metric max)",
               fontweight="bold")
ax_f.legend(fontsize=7.2, ncol=1, loc="upper left")
style_ax(ax_f)

# ── save ──────────────────────────────────────────────────────────────────────
out_png = OUT / "rq2_six_panels.png"
out_pdf = OUT / "rq2_six_panels.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"✓ Saved {out_png}")
print(f"✓ Saved {out_pdf}")


## Caption 
# Figure X. RQ2 — Defense effectiveness under the naive attacker at overwrite depth x=19x=19
# x=19, evaluated across seven sanitization strategies on four Transformer architectures and two entropy-stratified payloads (n=8n=8
# n=8 instances per defense).(a) Mean Recovery Reduction (RR) ± SD per payload type; stars denote significant departure from the 50% chance level (one-sample tt
# t-test, α=0.05\alpha=0.05
# α=0.05).
# (b) Mean accuracy drop ∣ΔAcc∣|\Delta\mathrm{Acc}|
# ∣ΔAcc∣ ± SD; the dotted line marks the 1% acceptability threshold.
# (c) Fitted normal CDFs of RR, confirming that GrayShield (GS), PatternMask (PM), and PTQ cluster near the 50% chance-level ceiling while all other defenses fall significantly below it.
# (d) Radar chart of five normalised deployment dimensions (higher = better on all axes); GS occupies the broadest balanced profile with no weak axis.
# (e) Z-scored heatmap across six metrics (green = above average); GS achieves above-average scores on all dimensions simultaneously.
# (f) Grouped bar chart of RR, ∣ΔAcc∣|\Delta\mathrm{Acc}|
# ∣ΔAcc∣, and Hamming distance expressed as a percentage of the per-metric maximum, illustrating the joint payload-disruption and model-fidelity trade-off across defenses.

