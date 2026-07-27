"""
gen_output_bias.py
Output bias scores by condition (Figure 3).
Bar chart of mean bias score per condition with ±1 SE error bars.
Source: data/bias_analysis/bias_score_analysis_out/completions_analysis_out/infer_bias_summary.csv
Style: seaborn-paper with serif font (matches LaTeX Computer Modern body text).
Output: output_bias.pdf (saved next to this script)
"""

import math
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")   # scripts/ -> plots/ -> paper/ -> repo root
PLOTS = os.path.join(HERE, "..")              # PDFs live in paper/plots/

SUMMARY_CSV = os.path.join(
    ROOT, "data", "bias_analysis", "bias_score_analysis_out",
    "completions_analysis_out", "infer_bias_summary.csv",
)

# Canonical condition order and display labels
COND_ORDER   = ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki",
                "llama-sft-gthb"]
COND_DISPLAY = {"base": "Base", "llama-sft-gt": "GT",
                "llama-sft-ps": "PS", "llama-sft-wiki": "Wiki",
                "llama-sft-gthb": "GT-HB"}
COLORS = {"Base": "#555555", "GT": "#c0392b", "PS": "#2980b9", "Wiki": "#27ae60",
          "GT-HB": "#7b241c"}

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi":      150,
})


def load_summary(path):
    """Return list of (display_label, mean, std, n) tuples in COND_ORDER."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["condition"]] = (
                float(r["mean"]),
                float(r["std"]),
                int(r["n"]),
            )
    return [(COND_DISPLAY[c], *rows[c]) for c in COND_ORDER]


data   = load_summary(SUMMARY_CSV)
labels = [d[0] for d in data]
means  = [d[1] for d in data]
stds   = [d[2] for d in data]
ns     = [d[3] for d in data]
ses    = [s / math.sqrt(n) for s, n in zip(stds, ns)]

fig, ax = plt.subplots(figsize=(5.2, 3.2))

x = list(range(len(labels)))
bars = ax.bar(
    x, means, yerr=ses,
    color=[COLORS[l] for l in labels],
    alpha=0.85, edgecolor="white", linewidth=1.0,
    capsize=4,
    error_kw={"elinewidth": 1.0, "ecolor": "black", "alpha": 0.75},
)

# Annotate mean above each bar (above the error bar tip)
for bar, m, se in zip(bars, means, ses):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        m + se + 0.08,
        f"{m:.3f}",
        ha="center", va="bottom", fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)])
ax.set_ylabel("Mean Bias Score (0-5)")
ax.set_ylim(0, max(means) + max(ses) + 0.5)
ax.grid(axis="y", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Output Bias by Condition", fontsize=11)

out = os.path.join(PLOTS, "output_bias.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
