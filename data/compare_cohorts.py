"""
compare_cohorts.py
------------------
Reads two graded CSVs (Control = NELA-GT clone, Experimental = NELA-PS)
and prints a statistical comparison of their Claude bias scores.

Usage:
  python3 compare_cohorts.py \
      --control   nela_gt_clone_graded.csv \
      --experiment nela_ps_graded.csv

Optional: pass --human to also compare human scores from the control CSV.
"""

import argparse
import csv
import math
import os


def load_scores(path: str, col: str) -> list[float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if col not in rows[0]:
        available = list(rows[0].keys())
        raise KeyError(f"Column '{col}' not in {path}. Available: {available}")
    scores = []
    for r in rows:
        val = r[col].strip()
        if val != "":
            try:
                scores.append(float(val))
            except ValueError:
                pass
    return scores


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def distribution(xs: list[float]) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for x in xs:
        k = int(round(x))
        if k in counts:
            counts[k] += 1
    return counts


def cohens_d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    pooled_sd = math.sqrt((stdev(a) ** 2 + stdev(b) ** 2) / 2)
    if pooled_sd == 0:
        return float("nan")
    return (mean(a) - mean(b)) / pooled_sd


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "N/A"


def print_comparison(
    ctrl_scores: list[float],
    exp_scores: list[float],
    ctrl_label: str,
    exp_label: str,
    score_label: str,
) -> None:
    ctrl_dist = distribution(ctrl_scores)
    exp_dist  = distribution(exp_scores)
    d         = cohens_d(exp_scores, ctrl_scores)

    bar_width = 30

    print(f"\n{'='*70}")
    print(f"  Bias Score Comparison  |  {score_label}")
    print(f"{'='*70}")
    print(f"  {'Metric':<22}  {'Control (GT)':>18}  {'Experiment (PS)':>18}")
    print(f"  {'':->22}  {'':->18}  {'':->18}")
    print(f"  {'N articles':<22}  {len(ctrl_scores):>18}  {len(exp_scores):>18}")
    print(f"  {'Mean bias (0-3)':<22}  {mean(ctrl_scores):>18.3f}  {mean(exp_scores):>18.3f}")
    print(f"  {'Std dev':<22}  {stdev(ctrl_scores):>18.3f}  {stdev(exp_scores):>18.3f}")
    print(f"  {'Min':<22}  {min(ctrl_scores):>18.0f}  {min(exp_scores):>18.0f}")
    print(f"  {'Max':<22}  {max(ctrl_scores):>18.0f}  {max(exp_scores):>18.0f}")
    print(f"  {'Cohens d (exp-ctrl)':<22}  {d:>37.3f}")
    print()

    print(f"  Score distribution:")
    print(f"  {'Score':<8}  {'Control N':>10}  {'Ctrl %':>8}  {'Exp N':>10}  {'Exp %':>8}")
    print(f"  {'':->8}  {'':->10}  {'':->8}  {'':->10}  {'':->8}")
    for score in range(4):
        cn = ctrl_dist[score]
        en = exp_dist[score]
        print(f"  {score:<8}  {cn:>10}  {pct(cn, len(ctrl_scores)):>8}  {en:>10}  {pct(en, len(exp_scores)):>8}")

    print()
    # ASCII bar chart of means
    ctrl_bar = int(mean(ctrl_scores) / 3 * bar_width)
    exp_bar  = int(mean(exp_scores)  / 3 * bar_width)
    print(f"  Mean bias (scale 0 ──────────────────────────────── 3)")
    print(f"  Control  [{('█' * ctrl_bar).ljust(bar_width)}] {mean(ctrl_scores):.3f}")
    print(f"  Experiment [{('█' * exp_bar).ljust(bar_width)}] {mean(exp_scores):.3f}")
    print(f"{'='*70}\n")

    # Interpretation hint
    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"
    direction = "higher" if mean(exp_scores) > mean(ctrl_scores) else "lower"
    print(f"  Interpretation: Experiment scores are {direction} than Control.")
    print(f"  Effect size (Cohen's d = {d:.3f}) is {interp}.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare bias score distributions across two cohorts.")
    parser.add_argument("--control",    required=True, help="Graded CSV for Control cohort (NELA-GT clone)")
    parser.add_argument("--experiment", required=True, help="Graded CSV for Experiment cohort (NELA-PS)")
    parser.add_argument("--human",      action="store_true", help="Also compare human scores from control CSV")
    args = parser.parse_args()

    ctrl_claude = load_scores(args.control,    "bias_score_claude")
    exp_claude  = load_scores(args.experiment, "bias_score_claude")

    print_comparison(ctrl_claude, exp_claude,
                     ctrl_label="NELA-GT clone",
                     exp_label="NELA-PS",
                     score_label="Claude annotations")

    if args.human:
        try:
            ctrl_human = load_scores(args.control, "bias_score_human")
            print_comparison(ctrl_human, exp_claude,
                             ctrl_label="NELA-GT clone",
                             exp_label="NELA-PS",
                             score_label="Human (ctrl) vs Claude (exp)")

            # Also compare human vs claude on control only
            print(f"\n{'='*70}")
            print(f"  Inter-rater check: Human vs Claude on Control cohort")
            print(f"{'='*70}")
            paired = [(h, c) for h, c in zip(ctrl_human, ctrl_claude)]
            diffs = [h - c for h, c in paired]
            agreements = sum(1 for d in diffs if d == 0)
            print(f"  N paired:          {len(paired)}")
            print(f"  Exact agreement:   {agreements}/{len(paired)} ({pct(agreements, len(paired))})")
            print(f"  Mean abs error:    {mean([abs(d) for d in diffs]):.3f}")
            print(f"  Human mean:        {mean(ctrl_human):.3f}")
            print(f"  Claude mean:       {mean(ctrl_claude):.3f}")
            print(f"{'='*70}\n")
        except KeyError as e:
            print(f"[--human skipped] {e}")


if __name__ == "__main__":
    main()
