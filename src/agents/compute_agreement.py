#!/usr/bin/env python3
"""Reproduce per-domain agreement metrics (weighted, all-agree, pairwise,
Fleiss' kappa, Gwet's AC1) from calibrated_labels.csv. Run: python compute_agreement.py"""
import csv
from math import comb
from itertools import combinations
from collections import defaultdict

PATH = "../../reports/rubric_calibration/calibrated_labels.csv"
JUDGES = ["llama_label", "mistral_label", "qwen_label"]

rows_by_domain = defaultdict(list)
with open(PATH) as f:
    for row in csv.DictReader(f):
        labels = [row[j] for j in JUDGES]
        if all(l in ("HIGH", "LOW") for l in labels):
            rows_by_domain[row["domain"]].append(labels)

def metrics(rows):
    N = len(rows)
    weighted = sum(1.0 if len(set(r)) == 1 else 0.667 for r in rows) / N
    all_agree = sum(len(set(r)) == 1 for r in rows) / N
    pair = sum(a == b for r in rows for a, b in combinations(r, 2)) / (3 * N)
    Pa = sum((comb(r.count("HIGH"), 2) + comb(r.count("LOW"), 2)) / comb(3, 2) for r in rows) / N
    piH = sum(r.count("HIGH") for r in rows) / (3 * N)
    piL = 1 - piH
    Pe_f, Pe_g = piH**2 + piL**2, 2 * piH * piL
    kappa = (Pa - Pe_f) / (1 - Pe_f) if Pe_f != 1 else float("nan")
    ac1 = (Pa - Pe_g) / (1 - Pe_g) if Pe_g != 1 else float("nan")
    return weighted, all_agree, pair, kappa, ac1

print(f"{'domain':28s} {'weighted':>9s} {'all-agree':>10s} {'pairwise':>9s} {'kappa':>8s} {'AC1':>8s}")
for dom in sorted(rows_by_domain):
    w, a, p, k, ac = metrics(rows_by_domain[dom])
    print(f"{dom:28s} {w:9.3f} {a:10.3f} {p:9.3f} {k:+8.3f} {ac:+8.3f}")
