#!/usr/bin/env python3
"""
B14 -- Consolidated human validation at n=20 per domain (rounds 1 and 2).

Reads the single consolidated ratings file
(gold-sampled-dataset/human_validation_all_rounds.csv), which merges the
round-1 study (10 transcripts per domain, even global indices) with the
round-2 study (10 per domain, odd global indices, disjoint annotator cohorts),
and recomputes every human-validation number reported in the paper:

  * per domain-round: unanimity, weighted agreement A (Eq. 1), Gwet's AC1,
    match against CRUCIBLE's calibrated label, and the MAJORITY-CLASS BASELINE
    (the score a rater achieves by labelling every item with the domain's
    dominant system-label class);
  * pooled per-domain human AC1 with percentile bootstrap CIs over items;
  * per-domain judge AC1 with the same CIs, from calibrated_labels.csv;
  * pairwise AC1 differences (judge-side and human-side) with CIs;
  * the metric-floor arithmetic: the unanimity required to satisfy a given
    tau, given that A has a floor of 2/3 for binary labels with K=3.

Two results here supersede earlier reported numbers, and the script prints
both so the change is auditable:
  1. Human--system match does not beat a constant-labelling baseline in five
     of six domain-round cells, so the earlier between-domain match contrast
     is prevalence-dominated.
  2. tau = 0.80 on A is satisfied at 8 of 20 unanimous items, so the loop
     halts while 60% of items still carry judge disagreement.

Outputs:
  results/human_validation_n20.json
  gold-sampled-dataset/human_validation_ground_truth_n20.csv
  papers/fig_reliability.pdf              (only if matplotlib is installed)

Usage:  python src/agents/exp_b14_human_validation_n20.py
"""
import csv
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RATINGS = BASE / "gold-sampled-dataset" / "human_validation_all_rounds.csv"
CALIBRATED = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
OUT_JSON = BASE / "results" / "human_validation_n20.json"
OUT_GOLD = BASE / "gold-sampled-dataset" / "human_validation_ground_truth_n20.csv"
OUT_FIG = BASE / "papers" / "fig_reliability.pdf"

N_BOOT = 10000
SEED = 0
TAU = 0.80
DOMAINS = ["general_education", "tech_ai", "career_selfimprovement"]
JUDGE_COLS = ["llama_label", "mistral_label", "qwen_label"]


# ---------------------------------------------------------------- metrics

def weighted_agreement(rating_lists):
    """Eq. 1: A = (u + 2/3 (N - u)) / N, u = number of unanimous items.
    Floor is 2/3 at u = 0 for a 3-judge binary panel."""
    n = len(rating_lists)
    if n == 0:
        return None
    u = sum(1 for r in rating_lists if len(set(r)) == 1)
    return (u + (2.0 / 3.0) * (n - u)) / n


def gwet_ac1(rating_lists):
    """Gwet's AC1 for nominal ratings, arbitrary raters per item (>=2)."""
    rows = [[x for x in r if x not in (None, "")] for r in rating_lists]
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        return None
    cats = sorted({x for r in rows for x in r})
    q = len(cats)
    if q < 2:
        return None  # single observed category: AC1 undefined
    pa = sum(
        sum(r.count(k) * (r.count(k) - 1) for k in cats) / (len(r) * (len(r) - 1))
        for r in rows
    ) / len(rows)
    pi = [sum(r.count(k) / len(r) for r in rows) / len(rows) for k in cats]
    pe = sum(p * (1 - p) for p in pi) / (q - 1)
    return (pa - pe) / (1 - pe)


def bootstrap_ac1(rating_lists, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over ITEMS. Resamples that collapse to a single
    observed category yield an undefined AC1 and are dropped, so the returned
    draw count can be below n_boot; it is reported alongside the interval."""
    rng = random.Random(seed)
    n = len(rating_lists)
    draws = []
    for _ in range(n_boot):
        samp = [rating_lists[rng.randrange(n)] for _ in range(n)]
        v = gwet_ac1(samp)
        if v is not None:
            draws.append(v)
    if not draws:
        return None, None, 0, []
    ordered = sorted(draws)
    lo = ordered[int(0.025 * (len(ordered) - 1))]
    hi = ordered[int(0.975 * (len(ordered) - 1))]
    return lo, hi, len(draws), draws  # draws returned in DRAW ORDER, not sorted


def diff_ci(draws_a, draws_b):
    """CI on the difference of two independent AC1 statistics.

    The draws must arrive in DRAW ORDER. Pairing sorted draws would give the
    difference of quantiles rather than the quantile of differences, which
    understates the interval substantially.
    """
    n = min(len(draws_a), len(draws_b))
    if n == 0:
        return None, None
    d = sorted(a - b for a, b in zip(draws_a[:n], draws_b[:n]))
    return d[int(0.025 * (n - 1))], d[int(0.975 * (n - 1))]


def required_unanimity(tau, n_items, floor=2.0 / 3.0):
    """Smallest u with (u + floor*(n-u))/n >= tau."""
    for u in range(n_items + 1):
        if (u + floor * (n_items - u)) / n_items >= tau - 1e-12:
            return u
    return None


# ---------------------------------------------------------------- loading

def load_ratings():
    """-> {(domain, round): {transcript_index: [labels]}} for consensus-included
    real transcripts only (attention checks and the replaced annotator dropped)."""
    out = defaultdict(lambda: defaultdict(list))
    with open(RATINGS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_attention_check"].strip().lower() == "true":
                continue
            if r["included_in_consensus"].strip().lower() != "true":
                continue
            out[(r["domain"], int(r["round"]))][int(r["transcript_index"])].append(r["label"])
    return out


def load_system_labels():
    """-> {transcript_index: (domain, signal_level, [judge labels])}"""
    out = {}
    with open(CALIBRATED, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["transcript_index"])] = (
                r["domain"], r["signal_level"], [r[c] for c in JUDGE_COLS]
            )
    return out


def majority_vote(labels):
    """Majority label; ties resolved to LOW (conservative for a HIGH-dominated
    reference). No tie can occur with an odd number of raters."""
    counts = {k: labels.count(k) for k in set(labels)}
    best = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == best)
    return winners[0] if len(winners) == 1 else "LOW"


# ---------------------------------------------------------------- main

def main():
    human = load_ratings()
    system = load_system_labels()

    per_cell, gold_rows = [], []
    for dom in DOMAINS:
        for rnd in (1, 2):
            idx = sorted(human[(dom, rnd)])
            if not idx:
                continue
            rating_lists = [human[(dom, rnd)][i] for i in idx]
            sys_labels = [system[i][1] for i in idx]
            maj = [majority_vote(r) for r in rating_lists]
            n = len(idx)
            n_high_sys = sum(1 for s in sys_labels if s == "HIGH")
            baseline = max(n_high_sys, n - n_high_sys) / n
            match = sum(1 for m, s in zip(maj, sys_labels) if m == s)
            per_cell.append({
                "domain": dom, "round": rnd, "n": n,
                "unanimous": sum(1 for r in rating_lists if len(set(r)) == 1),
                "weighted_A": round(weighted_agreement(rating_lists), 4),
                "gwet_ac1": None if gwet_ac1(rating_lists) is None else round(gwet_ac1(rating_lists), 4),
                "match": match, "match_rate": round(match / n, 4),
                "majority_class_baseline": round(baseline, 4),
                "beats_baseline": bool(match / n > baseline),
                "human_majority_HIGH": sum(1 for m in maj if m == "HIGH"),
                "system_HIGH": n_high_sys,
            })
            for i, r, m in zip(idx, rating_lists, maj):
                gold_rows.append({
                    "domain": dom, "round": rnd, "transcript_index": i,
                    "human_consensus_label": m, "annotator_labels": "|".join(r),
                    "n_high": r.count("HIGH"), "n_low": r.count("LOW"),
                    "system_label": system[i][1], "match": int(m == system[i][1]),
                })

    # pooled human and judge AC1, with bootstrap intervals
    # Each group gets its OWN random stream. Sharing one seed would make the
    # resampled item positions identical across groups, pairing two
    # independent rater panels and understating every difference interval.
    reliability, hdraws, jdraws = [], {}, {}
    for di, dom in enumerate(DOMAINS):
        pooled = [human[(dom, 1)][i] for i in sorted(human[(dom, 1)])] + \
                 [human[(dom, 2)][i] for i in sorted(human[(dom, 2)])]
        hlo, hhi, hn, hd = bootstrap_ac1(pooled, seed=SEED + 100 * di + 1)
        hdraws[dom] = hd
        judge = [v[2] for v in system.values() if v[0] == dom]
        jlo, jhi, jn, jd = bootstrap_ac1(judge, seed=SEED + 100 * di + 2)
        jdraws[dom] = jd
        u = sum(1 for r in judge if len(set(r)) == 1)
        reliability.append({
            "domain": dom,
            "human_n_items": len(pooled),
            "human_ac1_pooled": round(gwet_ac1(pooled), 4),
            "human_ac1_ci95": [round(hlo, 4), round(hhi, 4)], "human_boot_draws": hn,
            "judge_n_items": len(judge),
            "judge_unanimous": f"{u}/{len(judge)}",
            "judge_weighted_A": round(weighted_agreement(judge), 4),
            "judge_ac1": round(gwet_ac1(judge), 4),
            "judge_ac1_ci95": [round(jlo, 4), round(jhi, 4)], "judge_boot_draws": jn,
        })

    diffs = []
    for a, b in combinations(DOMAINS, 2):
        for side, dr in (("human", hdraws), ("judge", jdraws)):
            lo, hi = diff_ci(dr[a], dr[b])
            pooled_a = [human[(a, 1)][i] for i in sorted(human[(a, 1)])] + \
                       [human[(a, 2)][i] for i in sorted(human[(a, 2)])]
            pooled_b = [human[(b, 1)][i] for i in sorted(human[(b, 1)])] + \
                       [human[(b, 2)][i] for i in sorted(human[(b, 2)])]
            if side == "human":
                point = gwet_ac1(pooled_a) - gwet_ac1(pooled_b)
            else:
                point = gwet_ac1([v[2] for v in system.values() if v[0] == a]) - \
                        gwet_ac1([v[2] for v in system.values() if v[0] == b])
            diffs.append({"side": side, "a": a, "b": b, "difference": round(point, 4),
                          "ci95": [round(lo, 4), round(hi, 4)],
                          "excludes_zero": bool(lo > 0 or hi < 0)})

    # Within-domain, between-cohort change in human AC1 (round 1 vs round 2).
    # The two cohorts are disjoint, so these are independent samples.
    cohort_shifts = []
    for di, dom in enumerate(DOMAINS):
        r1 = [human[(dom, 1)][i] for i in sorted(human[(dom, 1)])]
        r2 = [human[(dom, 2)][i] for i in sorted(human[(dom, 2)])]
        a1, a2 = gwet_ac1(r1), gwet_ac1(r2)
        _, _, _, d1 = bootstrap_ac1(r1, seed=SEED + 100 * di + 11)
        _, _, _, d2 = bootstrap_ac1(r2, seed=SEED + 100 * di + 12)
        lo, hi = diff_ci(d1, d2)
        cohort_shifts.append({
            "domain": dom,
            "ac1_round1": None if a1 is None else round(a1, 4),
            "ac1_round2": None if a2 is None else round(a2, 4),
            "difference": None if None in (a1, a2) else round(a1 - a2, 4),
            "ci95": [None, None] if lo is None else [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
        })

    n_tot = sum(c["n"] for c in per_cell)
    n_match = sum(c["match"] for c in per_cell)
    result = {
        "n_transcripts": n_tot,
        "n_annotators": 18,
        "design": "20 transcripts per domain; two rounds of 10 with disjoint "
                  "cohorts of 3 annotators each; round 1 = even global indices, "
                  "round 2 = odd.",
        "metric_floor": {
            "floor": round(2.0 / 3.0, 4),
            "tau": TAU,
            "required_unanimous_of_20": required_unanimity(TAU, 20),
            "note": "tau is satisfied at 40% unanimity, so the loop halts "
                    "while 60% of items still carry judge disagreement.",
        },
        "per_cell": per_cell,
        "cells_beating_own_baseline": sum(1 for c in per_cell if c["beats_baseline"]),
        "pooled_match": n_match,
        "pooled_match_rate": round(n_match / n_tot, 4),
        "pooled_majority_class_baseline": round(
            sum(c["majority_class_baseline"] * c["n"] for c in per_cell) / n_tot, 4),
        "reliability": reliability,
        "ac1_differences": diffs,
        "human_ac1_cohort_shift": cohort_shifts,
        "bootstrap": {"n_requested": N_BOOT, "seed": SEED,
                      "note": "resampled over items; draws collapsing to a "
                              "single observed category are dropped"},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    with open(OUT_GOLD, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gold_rows[0]))
        w.writeheader()
        w.writerows(sorted(gold_rows, key=lambda r: (r["domain"], r["transcript_index"])))

    print(f"{'domain':24s}{'rd':>3s}{'unan':>6s}{'A':>8s}{'AC1':>8s}"
          f"{'match':>7s}{'base':>7s}{'beats':>7s}")
    for c in per_cell:
        ac1 = "n/a" if c["gwet_ac1"] is None else f"{c['gwet_ac1']:.3f}"
        print(f"{c['domain']:24s}{c['round']:>3d}{c['unanimous']:>6d}"
              f"{c['weighted_A']:>8.3f}{ac1:>8s}{c['match_rate']:>7.0%}"
              f"{c['majority_class_baseline']:>7.0%}{str(c['beats_baseline']):>7s}")
    print(f"\ncells beating their own baseline: "
          f"{result['cells_beating_own_baseline']}/{len(per_cell)}")
    print(f"pooled match: {n_match}/{n_tot} = {n_match / n_tot:.0%} "
          f"(baseline {result['pooled_majority_class_baseline']:.0%})")
    print(f"\ntau={TAU} on A (floor {2/3:.3f}) is met at "
          f"{result['metric_floor']['required_unanimous_of_20']}/20 unanimous items")
    print(f"\n{'domain':24s}{'judge AC1':>22s}{'human AC1 (pooled)':>26s}")
    for r in reliability:
        print(f"{r['domain']:24s}"
              f"{r['judge_ac1']:>9.3f} [{r['judge_ac1_ci95'][0]:+.2f},{r['judge_ac1_ci95'][1]:+.2f}]"
              f"{r['human_ac1_pooled']:>13.3f} [{r['human_ac1_ci95'][0]:+.2f},{r['human_ac1_ci95'][1]:+.2f}]")
    print(f"\nwrote {OUT_JSON.relative_to(BASE)}")
    print(f"wrote {OUT_GOLD.relative_to(BASE)}")

    try:
        make_figure(reliability)
        print(f"wrote {OUT_FIG.relative_to(BASE)}")
    except ImportError:
        print("matplotlib not installed; skipped papers/fig_reliability.pdf")


def make_figure(reliability):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nice = {"general_education": "General\nEducation",
            "tech_ai": "Technology\n& AI",
            "career_selfimprovement": "Career &\nSelf-Impr."}
    c_j, c_h, grey = "#1f4e79", "#c8622a", "#7f7f7f"
    fig, ax = plt.subplots(figsize=(3.35, 2.95))
    xs = list(range(len(reliability)))
    ax.axhline(0, color=grey, lw=0.8, zorder=1)
    for off, key, ci, colour, marker in (
        (-0.13, "judge_ac1", "judge_ac1_ci95", c_j, "o"),
        (0.13, "human_ac1_pooled", "human_ac1_ci95", c_h, "s"),
    ):
        vals = [r[key] for r in reliability]
        lo = [v - r[ci][0] for v, r in zip(vals, reliability)]
        hi = [r[ci][1] - v for v, r in zip(vals, reliability)]
        ax.errorbar([x + off for x in xs], vals, yerr=[lo, hi], fmt=marker,
                    ms=5.5, color=colour, capsize=2.5, lw=1.3, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([nice[r["domain"]] for r in reliability], fontsize=8)
    ax.set_ylabel("Gwet AC1 (chance-corrected)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(-0.45, 1.16)
    ax.set_xlim(-0.5, len(reliability) - 0.5)
    ax.text(-0.42, 1.12, "higher = raters agree", fontsize=7, color=grey, va="top")
    ax.annotate("3 LLM judges", (xs[0] - 0.13, reliability[0]["judge_ac1"]),
                xytext=(0.30, 1.00), fontsize=7, color=c_j, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=c_j, shrinkA=0, shrinkB=3))
    ax.annotate("3+3 humans", (xs[0] + 0.13, reliability[0]["human_ac1_pooled"]),
                xytext=(0.62, 0.53), fontsize=7, color=c_h, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=c_h, shrinkA=0, shrinkB=3))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")


if __name__ == "__main__":
    main()
