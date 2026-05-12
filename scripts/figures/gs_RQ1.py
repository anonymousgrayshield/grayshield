#!/usr/bin/env python3
"""
gsRQ1_camera_ready.py

Generate a camera-ready 1 × 4 RQ1 figure.

Panels:
(a) Cosine deviation vs overwrite depth
(b) Relative L2 vs overwrite depth
(c) Accuracy drop vs overwrite depth
(d) Spearman depth--metric association

Input:
    datasets/rq1.jsonl

Output:
    figures/rq1_final.png
    figures/rq1_final.pdf
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path("datasets")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RQ1_PATH = BASE / "rq1.jsonl"

PAYLOAD_COLORS = {
    "High-entropy": "#c83f36",
    "Low-entropy": "#2c7fb8",
    "Payload": "#8f8f8f",
}
PAYLOAD_MARKERS = {"High-entropy": "o", "Low-entropy": "s", "Payload": "^"}
PAYLOAD_LINESTYLES = {"High-entropy": "-", "Low-entropy": "--", "Payload": ":"}
PAYLOAD_ORDER = ["High-entropy", "Low-entropy", "Payload"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.titlesize": 8.1,
    "axes.labelsize": 7.5,
    "legend.fontsize": 5.8,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.50,
})

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

def sig_stars(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def classify_payload(payload):
    if isinstance(payload, dict):
        for key in ("normalized_entropy", "entropy", "norm_entropy"):
            if key in payload and payload[key] is not None:
                try:
                    return "High-entropy" if float(payload[key]) >= 0.85 else "Low-entropy"
                except Exception:
                    pass
        sha = str(payload.get("sha256", ""))
        if sha.startswith("5704") or sha.startswith("5704fabd"):
            return "High-entropy"
        if sha.startswith("c37c"):
            return "Low-entropy"
        if sha:
            return "Low-entropy"
    s = str(payload).upper()
    if "HE" in s or "HIGH" in s or "EXE" in s:
        return "High-entropy"
    if "LE" in s or "LOW" in s or "JS" in s:
        return "Low-entropy"
    return "Payload"

def safe_spearman(x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3 or tmp["y"].nunique() < 2:
        return np.nan, np.nan
    rho, p = stats.spearmanr(tmp["x"], tmp["y"])
    return float(rho), float(max(p, 1e-300))

def fisher_ci(rho, n):
    if pd.isna(rho) or n <= 3:
        return np.nan, np.nan
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return min(lo, rho), max(hi, rho)

def format_p(p):
    if pd.isna(p):
        return "p=n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.4f}"

def flatten(records):
    rows = []
    for obj in records:
        items = obj.get("results", [obj]) if isinstance(obj.get("results"), list) else [obj]
        for r in items:
            m = r.get("metrics", r)
            payload = r.get("payload", obj.get("payload", r.get("payload_type", "")))
            cosdev = first(m, ["cosine_deviation", "cos_dev", "cosine_dev"])
            if pd.isna(cosdev):
                cossim = first(m, ["cosine_similarity", "cosine_sim"])
                if not pd.isna(cossim):
                    cosdev = 1.0 - float(cossim)
            rows.append({
                "x": first(r, ["x", "x_bits", "overwrite_depth", "depth"]),
                "payload": classify_payload(payload),
                "model": first(r, ["model", "model_name", "model_preset"],
                               default=first(obj, ["model", "model_name", "model_preset"], default="")),
                "cosdev": cosdev,
                "drel": first(m, ["relative_l2", "relative_l2_distance", "Drel"]),
                "acc": first(m, ["acc_drop", "accuracy_drop", "delta_acc"]),
            })
    df = pd.DataFrame(rows)
    for c in ["x", "cosdev", "drel", "acc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x"]).copy()
    df["acc_pct"] = to_pct(df["acc"])
    return df

records = [json.loads(line) for line in open(RQ1_PATH, encoding="utf-8") if line.strip()]
df = flatten(records)
depths = sorted(df["x"].dropna().unique())

fig = plt.figure(figsize=(16.0, 4.2), constrained_layout=False)
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 0.95, 1.0, 1.10], wspace=0.34)

## fig.suptitle("RQ1 — Injection Feasibility and Multi-Metric Stealth", fontsize=11.6, fontweight="bold", y=0.985)

def line_panel(ax, metric, ylabel, title, legend=True):
    for payload in PAYLOAD_ORDER:
        sub = df[df["payload"] == payload]
        if sub.empty:
            continue
        g = sub.groupby("x", as_index=False)[metric].mean().sort_values("x")
        y = g[metric].copy().replace(0, np.nan)
        fill = y[y > 0].min() if (y > 0).any() else 1e-14
        y = y.fillna(fill)
        ax.semilogy(
            g["x"], y,
            marker=PAYLOAD_MARKERS[payload],
            linestyle=PAYLOAD_LINESTYLES[payload],
            color=PAYLOAD_COLORS[payload],
            lw=1.70 if payload == "High-entropy" else 1.45,
            ms=4.4,
            label="HE" if payload == "High-entropy" else "LE" if payload == "Low-entropy" else payload,
        )
    ax.axvline(16, ls="--", color="#555", lw=0.85, alpha=0.80, label="x=16 limit")
    ax.set_xlabel("Overwrite depth x")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, which="both", alpha=0.22)
    if legend:
        ax.legend(fontsize=5.8, frameon=True, loc="upper left")

# (a)
ax_a = fig.add_subplot(gs[0, 0])
x_mean = df.groupby("x")["x"].mean().values
rho_cos, p_cos = safe_spearman(x_mean, df.groupby("x")["cosdev"].mean().values)
line_panel(ax_a, "cosdev", "Cosine deviation (log)",
           f"(a) Cosine deviation vs depth\nSpearman ρ={rho_cos:.3f}, {format_p(p_cos)}",
           legend=True)

# (b)
ax_b = fig.add_subplot(gs[0, 1])
rho_l2, p_l2 = safe_spearman(x_mean, df.groupby("x")["drel"].mean().values)
line_panel(ax_b, "drel", r"Rel. $\ell_2$ (log scale)",
           f"(b) Relative $\\ell_2$ vs depth\nSpearman ρ={rho_l2:.3f}, {format_p(p_l2)}",
           legend=True)

# (c)
ax_c = fig.add_subplot(gs[0, 2])
for payload in ["High-entropy", "Low-entropy"]:
    sub = df[df["payload"] == payload]
    if sub.empty:
        continue
    g = sub.groupby("x", as_index=False)["acc_pct"].mean().sort_values("x")
    ax_c.plot(
        g["x"], g["acc_pct"],
        marker=PAYLOAD_MARKERS[payload],
        linestyle=PAYLOAD_LINESTYLES[payload],
        color=PAYLOAD_COLORS[payload],
        lw=1.75 if payload == "High-entropy" else 1.45,
        ms=4.4,
        label="HE (mean)" if payload == "High-entropy" else "LE (mean)",
    )
for payload in ["High-entropy", "Low-entropy"]:
    sub = df[df["payload"] == payload]
    if sub.empty:
        continue
    ax_c.scatter(sub["x"], sub["acc_pct"], color=PAYLOAD_COLORS[payload],
                 marker=PAYLOAD_MARKERS[payload], s=14, alpha=0.20, edgecolors="none")
ax_c.axhspan(0, 1.0, color="#2ecc71", alpha=0.08, label="Stealth zone")
ax_c.axhline(1.0, ls=":", color="#d62728", lw=1.1, label="1% threshold")
ax_c.axvline(16, ls="--", color="#555", lw=0.85, alpha=0.80)
ax_c.axvspan(16, max(depths) + 0.5, color="#e74c3c", alpha=0.055)
ax_c.text(0.33, 0.48, "Stealth\nzone", transform=ax_c.transAxes,
          ha="center", va="center", fontsize=6.5, color="#118c4f", style="italic")
ax_c.text(0.82, 0.50, "Truncation\nzone", transform=ax_c.transAxes,
          ha="center", va="center", fontsize=6.5, color="#b23b3b", style="italic")
he = df[df["payload"] == "High-entropy"]
if not he.empty and he["acc_pct"].notna().any():
    max_row = he.loc[he["acc_pct"].idxmax()]
    ax_c.annotate(f"{max_row['acc_pct']:.2f}%",
                  xy=(max_row["x"], max_row["acc_pct"]),
                  xytext=(-24, 18),
                  textcoords="offset points",
                  arrowprops=dict(arrowstyle="->", color=PAYLOAD_COLORS["High-entropy"], lw=0.8),
                  fontsize=6.3,
                  color=PAYLOAD_COLORS["High-entropy"],
                  fontweight="bold")
rho_acc, p_acc = safe_spearman(x_mean, df.groupby("x")["acc_pct"].mean().values)
ax_c.set_xlabel("Overwrite depth x")
ax_c.set_ylabel(r"$\Delta$Acc (%)")
ax_c.set_title(f"(c) $\\Delta$Acc vs depth\nSpearman ρ={rho_acc:.3f}, {format_p(p_acc)}",
               fontweight="bold")
ax_c.legend(fontsize=5.6, frameon=True, loc="upper left")
ax_c.grid(True, alpha=0.22)

# (d)
ax_d = fig.add_subplot(gs[0, 3])
series = {
    "Cos.\ndev.": df.groupby("x")["cosdev"].mean(),
    r"$D_{\mathrm{rel}}$": df.groupby("x")["drel"].mean(),
}
for payload in ["High-entropy", "Low-entropy"]:
    sub = df[df["payload"] == payload]
    if sub.empty:
        continue
    label = r"$\Delta$Acc" + "\n(HE)" if payload == "High-entropy" else r"$\Delta$Acc" + "\n(LE)"
    series[label] = sub.groupby("x")["acc_pct"].mean()

metrics = []
for label, ser in series.items():
    s = ser.reindex(depths)
    rho, p = safe_spearman(depths, s.values)
    lo, hi = fisher_ci(rho, len(depths))
    metrics.append((label, rho, p, lo, hi))

bar_colors = ["#c83f36", "#2c7fb8", "#3cb371", "#8f8f8f"][:len(metrics)]
xpos = np.arange(len(metrics))
for i, (label, rho, p, lo, hi) in enumerate(metrics):
    ax_d.bar(i, rho, width=0.58, color=bar_colors[i], alpha=0.86, zorder=3)
    if not pd.isna(lo):
        lo_plot = max(-1.0, min(float(lo), float(rho)))
        hi_plot = min(1.0, max(float(hi), float(rho)))
        yerr_low = max(0.0, float(rho) - lo_plot)
        yerr_high = max(0.0, hi_plot - float(rho))
        ax_d.errorbar(i, rho, yerr=[[yerr_low], [yerr_high]], fmt="none",
                      ecolor="#222", capsize=3.2, lw=1.0, capthick=1.0, zorder=4)
        star_y = min(1.10, hi_plot + 0.035)
    else:
        star_y = min(1.10, rho + 0.035)
    ax_d.text(i, star_y, sig_stars(p), ha="center", va="bottom",
              fontsize=7.4, fontweight="bold",
              color="#b23b3b" if (not pd.isna(p) and p < 0.05) else "#555")

ax_d.axhline(0.9, ls="--", color="#555", lw=0.85, alpha=0.70, label=r"$\rho=0.9$")
ax_d.set_ylim(0, 1.15)
ax_d.set_xticks(xpos)
ax_d.set_xticklabels([m[0] for m in metrics])
ax_d.set_ylabel(r"Spearman $\rho$ (95% CI)")
ax_d.set_title("(d) Depth–metric association\n95% CI via Fisher z-transform", fontweight="bold")
ax_d.legend(fontsize=5.7, loc="lower right", frameon=True)
ax_d.grid(True, axis="y", alpha=0.22)

fig.subplots_adjust(left=0.050, right=0.982, top=0.80, bottom=0.19, wspace=0.34)

out_png = OUT / "rq1_final.png"
out_pdf = OUT / "rq1_final.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"✓ Saved: {out_png}")
print(f"✓ Saved: {out_pdf}")
