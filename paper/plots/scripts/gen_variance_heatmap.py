"""
gen_variance_heatmap.py
Supplementary: per-prompt mean bias score + hallucination rate heatmaps.

Two stacked panels (15 prompts on the y-axis, 5 conditions on the x-axis):
  Panel A  Mean bias score per (prompt, condition), sequential 0--5 colormap.
  Panel B  Hallucination rate (%) per (prompt, condition), sequential 0--100
           colormap.

Pairing the two panels makes the central finding of Section 5.2 visible at a
glance: the prompts where Condition Wiki appears most biased (Panel A) are the
same prompts where it hallucinates most (Panel B).

The panels are stacked (not side by side) because the figure is authored at the
ACL \columnwidth (~3.03in); side by side there is not enough room for the cell
annotations or the prompt labels.

Sources:
  data/bias_analysis/bias_score_analysis_out/completions_analysis_out/variance_analysis.csv
  data/bias_analysis/bias_score_analysis_out/completions_analysis_out/hallucination_per_prompt.csv
Style: seaborn-paper with serif font (matches LaTeX Computer Modern body text).
Output: variance_heatmap.pdf (saved next to this script)
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")   # scripts/ -> plots/ -> paper/ -> repo root
PLOTS = os.path.join(HERE, "..")              # PDFs live in paper/plots/

ANALYSIS_DIR = os.path.join(
    ROOT, "data", "bias_analysis", "bias_score_analysis_out",
    "completions_analysis_out",
)
VARIANCE_CSV       = os.path.join(ANALYSIS_DIR, "variance_analysis.csv")
HALLUCINATION_CSV  = os.path.join(ANALYSIS_DIR, "hallucination_per_prompt.csv")

COND_ORDER   = ["base", "llama-sft-gt", "llama-sft-ps", "llama-sft-wiki",
                "llama-sft-gthb"]
COND_DISPLAY = {"base": "Base", "llama-sft-gt": "GT",
                "llama-sft-ps": "PS", "llama-sft-wiki": "Wiki",
                "llama-sft-gthb": "GT-HB"}
PROMPT_DISPLAY = {
    "climate":"Climate", "criminal_justice":"Crim. Justice", "education":"Education",
    "government":"Government", "healthcare":"Healthcare", "healthcare_insurance":"Healthcare Ins.",
    "housing":"Housing", "immigration":"Immigration", "military":"Military",
    "policing":"Policing", "religion":"Religion", "rural_urban":"Rural/Urban",
    "social_welfare":"Social Welfare", "tech_regulation":"Tech Regulation", "trade":"Trade",
}

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    # Authored at the final rendered width (ACL \columnwidth ~ 3.03in), so the
    # figure is *not* shrunk by LaTeX and these point sizes are what the reader
    # actually sees.
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "figure.dpi":      150,
})

COLWIDTH_IN = 3.03   # ACL two-column \columnwidth


def load_means(path):
    """Return dict[prompt_id][condition] = mean (float). Long-format CSV."""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["prompt_id"]][r["condition"]] = float(r["mean"])
    return out


def load_hall_rates(path):
    """Return dict[prompt_id][condition] = hall_rate (float). Wide-format CSV
    with columns like base_hall_rate, llama-sft-gt_hall_rate, etc."""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for cond in COND_ORDER:
                col = f"{cond}_hall_rate"
                v = r.get(col, "")
                out[r["prompt_id"]][cond] = float(v) if v != "" else np.nan
    return out


means       = load_means(VARIANCE_CSV)
hall_rates  = load_hall_rates(HALLUCINATION_CSV)
prompts     = sorted(means.keys())

mean_matrix = np.array([
    [means[p].get(c, np.nan) for c in COND_ORDER]
    for p in prompts
])
hall_matrix = np.array([
    [hall_rates[p].get(c, np.nan) for c in COND_ORDER]
    for p in prompts
])


def draw_heatmap(ax, matrix, *, cmap, vmin, vmax, fmt, dark_thresh,
                  panel_title, xlabel=None):
    """Render a single heatmap panel with cell annotations and tidy ticks."""
    im = ax.imshow(
        matrix, aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            color = "white" if v >= dark_thresh else "#333333"
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=8.5, color=color)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([COND_DISPLAY[c] for c in COND_ORDER])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(panel_title, loc="left", fontsize=9, pad=4)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(COND_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(prompts), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)   # no tick dashes; cells are self-bordering
    return im


fig, (axA, axB) = plt.subplots(2, 1, figsize=(COLWIDTH_IN, 6.6))
# Reserve the left margin for the prompt labels so they stay inside the canvas
# (otherwise bbox_inches="tight" grows the PDF past \columnwidth and LaTeX
# shrinks the whole figure back down again).
fig.subplots_adjust(left=0.32, right=0.99, top=0.90, bottom=0.05, hspace=0.42)

imA = draw_heatmap(
    axA, mean_matrix,
    cmap="YlOrRd", vmin=0, vmax=5,
    fmt=lambda v: f"{v:.2f}", dark_thresh=3.0,
    panel_title="(A) Mean Bias Score",
)
axA.set_yticks(range(len(prompts)))
axA.set_yticklabels([PROMPT_DISPLAY.get(p, p) for p in prompts])
axA.set_ylabel("Prompt Topic")

imB = draw_heatmap(
    axB, hall_matrix,
    cmap="Purples", vmin=0, vmax=100,
    fmt=lambda v: f"{int(round(v))}%", dark_thresh=60.0,
    panel_title="(B) Hallucination Rate",
)
axB.set_yticks(range(len(prompts)))
axB.set_yticklabels([PROMPT_DISPLAY.get(p, p) for p in prompts])
axB.set_ylabel("Prompt Topic")

# Colorbars below each panel
cbarA = fig.colorbar(imA, ax=axA, fraction=0.05, pad=0.04,
                     orientation="horizontal", location="bottom")
cbarA.set_label("Mean Bias Score (0--5)", fontsize=8)
cbarA.ax.tick_params(labelsize=8)

cbarB = fig.colorbar(imB, ax=axB, fraction=0.05, pad=0.04,
                     orientation="horizontal", location="bottom")
cbarB.set_label("Hallucination Rate (%)", fontsize=8)
cbarB.ax.tick_params(labelsize=8)

fig.suptitle(
    "Per-prompt Mean Bias Score and\nHallucination Rate by Condition",
    fontsize=9, y=0.995,
)

out = os.path.join(PLOTS, "variance_heatmap.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print(f"saved → {out}")
