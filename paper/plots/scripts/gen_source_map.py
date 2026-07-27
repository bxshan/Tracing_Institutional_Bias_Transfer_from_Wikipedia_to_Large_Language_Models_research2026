"""
gen_source_map.py
GT-HB source selection map: one point per audited source -- per-source mean
rubric score (x) vs corpus article count (y, log scale). Filled marker =
selected for the GT-HB whitelist (mean >= 3.0), open marker = excluded;
color = MBFC label group. Vertical line = selection threshold.

Source:
  data/bias_analysis/bias_scores/bias_scores_gt.csv
  data/gt_hb/bias_scores_topup.csv
  data/gt_hb/source_metadata.csv
Aggregation mirrors data/gt_hb/finalize_sources.py exactly (no new statistics).
Style: seaborn-paper with serif font (matches the other paper/plots scripts).
Output: source_map.pdf (saved next to this script)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")   # scripts/ -> plots/ -> paper/ -> repo root
PLOTS = os.path.join(HERE, "..")              # PDFs live in paper/plots/

THRESHOLD, MIN_GRADED, MIN_ARTICLES = 3.0, 10, 1000  # mirrors finalize_sources.py

GROUP_OF = {
    "Questionable Source":      "Questionable",
    "Conspiracy-Pseudoscience": "Conspiracy/Pseudoscience",
    "Left":                     "Left",
    "Right":                    "Right",
    "Left-Center":              "Center-leaning",
    "Right-Center":             "Center-leaning",
    "Least Biased":             "Center-leaning",
    "Pro-Science":              "Center-leaning",
    "Satire":                   "Satire",
}
GROUP_COLOR = {
    "Questionable":         "#c0392b",
    "Conspiracy/Pseudoscience": "#8e44ad",
    "Left":                 "#2980b9",
    "Right":                "#e67e22",
    "Center-leaning":       "#27ae60",
    "Satire":               "#d81b60",
}
# Redundant (non-colour) channel: each MBFC group also gets its own marker
# shape, so the groups stay separable when printed in greyscale.
GROUP_MARKER = {
    "Questionable":             "o",
    "Conspiracy/Pseudoscience": "^",
    "Left":                     "s",
    "Right":                    "D",
    "Center-leaning":           "v",
    "Satire":                   "P",
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


def load_per_source():
    """Per-source mean/count of judge scores + corpus size + MBFC label."""
    gt = pd.read_csv(os.path.join(ROOT, "data", "bias_analysis", "bias_scores", "bias_scores_gt.csv"))
    tu = pd.read_csv(os.path.join(ROOT, "data", "gt_hb", "bias_scores_topup.csv"))
    s = pd.concat([gt[["source", "bias_score"]], tu[["source", "bias_score"]]])
    s = s[s.bias_score >= 0]                       # drop parse errors (-1)
    agg = (s.groupby("source").bias_score
            .agg(mean_score="mean", n_graded="count").reset_index())
    meta = pd.read_csv(os.path.join(ROOT, "data", "gt_hb", "source_metadata.csv"))
    df = agg.merge(meta[["source", "n_articles", "label"]], on="source", how="left")
    df = df[(df.n_graded >= MIN_GRADED) & (df.n_articles >= MIN_ARTICLES)].copy()
    df["selected"] = df.mean_score >= THRESHOLD
    df["group"] = df.label.map(GROUP_OF)
    return df.reset_index(drop=True)


df = load_per_source()

fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.5))
ax.set_yscale("log")
ax.set_xlim(-0.15, 4.75)
ax.axvspan(THRESHOLD, 4.75, color="#fdecea", zorder=0)
ax.axvline(THRESHOLD, color="black", lw=1, ls="--", zorder=1)

for g, c in GROUP_COLOR.items():
    m = GROUP_MARKER[g]
    sel = df[(df.group == g) & df.selected]
    exc = df[(df.group == g) & ~df.selected]
    ax.scatter(sel.mean_score, sel.n_articles, s=16, marker=m, color=c, lw=0,
               alpha=0.9, zorder=3)
    ax.scatter(exc.mean_score, exc.n_articles, s=16, marker=m, facecolors="none",
               edgecolors=c, lw=0.6, alpha=0.9, zorder=3)

LABELS = {  # source -> (dx pts, dy pts); negative dx = label to the left
    "thesun": (4, -2), "bbc": (4, -2), "foxnews": (4, -4),
    "sputnik": (4, 2), "breitbart": (4, -2),
    # right-hand cluster: label leftwards so the text stays inside the axes,
    # with dy chosen to de-collide the labels at column width
    "msnbc": (-4, -8), "westernjournal": (-4, 14), "thegatewaypundit": (-4, -7),
    "infowars": (-4, 4), "thespoof": (-4, -4), "dailystormer": (-4, -6),
}
for name, (dx, dy) in LABELS.items():
    r = df[df.source == name]
    if r.empty:
        continue
    r = r.iloc[0]
    ax.annotate(name, (r.mean_score, r.n_articles), xytext=(dx, dy),
                textcoords="offset points", fontsize=7,
                ha="right" if dx < 0 else "left")

n_sel = int(df.selected.sum())
pool = int(df.loc[df.selected, "n_articles"].sum())
ax.text(0.0, 1.01,
        f"{n_sel}/{len(df)} sources selected ({pool:,} articles)",
        transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
ax.text(THRESHOLD + 0.07, 0.97, f"selected:\nmean $\\geq$ {THRESHOLD}",
        transform=ax.get_xaxis_transform(), fontsize=8, va="top",
        linespacing=1.2)

handles = [Line2D([], [], marker=GROUP_MARKER[g], ls="", color=c,
                  markersize=4, label=g)
           for g, c in GROUP_COLOR.items()]
handles += [
    Line2D([], [], marker="o", ls="", color="black", markersize=4,
           label="selected (filled)"),
    # seaborn-paper sets lines.markeredgewidth=0 -> open marker would vanish
    Line2D([], [], marker="o", ls="", markerfacecolor="none",
           markeredgecolor="black", markeredgewidth=0.8, color="black",
           markersize=4, label="excluded (open)"),
]
ax.legend(handles=handles, fontsize=7, loc="upper center",
          bbox_to_anchor=(0.5, -0.20), ncol=2, framealpha=0.9,
          handletextpad=0.4, columnspacing=1.0, borderpad=0.4,
          labelspacing=0.3)
ax.text(0.5, -0.62, "marker shape/color = MBFC (Media Bias/Fact Check) label,\n"
        "not produced by this work",
        transform=ax.transAxes, fontsize=7, color="#555555",
        ha="center", va="top", style="italic", linespacing=1.2)

ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Per-Source Mean Rubric Bias Score (0--5)")
ax.set_ylabel("Corpus Articles per Source")
fig.suptitle("GT-HB Source Selection Map", fontsize=9)

out = os.path.join(PLOTS, "source_map.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print(f"saved → {out}")
