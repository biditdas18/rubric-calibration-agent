#!/usr/bin/env python3
"""
Relabel all 60 transcripts using the FINAL rubrics from calibrated_rubrics.json,
reusing the exact judge path from src/agents/judge_agent.py.
Writes calibrated_labels_regen.csv with the same schema as calibrated_labels.csv.
Run from: reports/rubric_calibration/
"""
import csv
import json
import sys
import os

# Resolve paths relative to this file
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, os.path.join(REPO, "src", "agents"))
from judge_agent import (
    JUDGE_MODELS,
    JUDGE_TEMPERATURE,
    run_all_transcripts_one_model,
    merge_model_results,
)

RUBRICS_PATH  = os.path.join(HERE, "calibrated_rubrics.json")
QUEUE_PATH    = os.path.join(REPO, "data", "review_queue.csv")
OUTPUT_PATH   = os.path.join(HERE, "calibrated_labels_regen.csv")

# ── Load final rubrics ────────────────────────────────────────────────────
with open(RUBRICS_PATH) as f:
    final_rubrics = json.load(f)

# ── Load transcripts, grouped by domain ───────────────────────────────────
domain_transcripts = {}
with open(QUEUE_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        d = row["domain"]
        domain_transcripts.setdefault(d, [])
        idx = len(domain_transcripts[d])
        domain_transcripts[d].append({"index": idx, "transcript": row["transcript"]})

print(f"Loaded transcripts: {sum(len(v) for v in domain_transcripts.values())} total")
for d, ts in sorted(domain_transcripts.items()):
    print(f"  {d}: {len(ts)} transcripts, rubric v{final_rubrics[d]['version']}")

# ── Run all three models, one model at a time (columnar order) ────────────
all_results = {}  # domain -> list of merged results

for domain, transcripts in sorted(domain_transcripts.items()):
    rubric = final_rubrics[domain]
    results_by_model = {}

    for model in JUDGE_MODELS:
        print(f"\n[{domain}] Running {model} over {len(transcripts)} transcripts...")
        model_results = run_all_transcripts_one_model(transcripts, domain, rubric, model)
        results_by_model[model] = model_results
        labels_seen = [r["label"] for r in model_results]
        print(f"  -> HIGH:{labels_seen.count('HIGH')} LOW:{labels_seen.count('LOW')}")

    merged = merge_model_results(transcripts, results_by_model)
    all_results[domain] = merged

# ── Write output CSV ──────────────────────────────────────────────────────
fieldnames = [
    "domain", "transcript_index", "signal_level", "agreement",
    "votes_high", "votes_low", "all_agree",
    "llama_label", "mistral_label", "qwen_label",
    "llama_criterion", "mistral_criterion", "qwen_criterion",
]

model_to_col = {
    JUDGE_MODELS[0]: "llama",
    JUDGE_MODELS[1]: "mistral",
    JUDGE_MODELS[2]: "qwen",
}

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for domain in sorted(all_results.keys()):
        # Need per-model results to write individual labels/criteria.
        # Re-derive from merged individual_results list.
        rubric = final_rubrics[domain]
        transcripts = domain_transcripts[domain]

        # Re-run to get individual results aligned to models order
        # (they're embedded in merged["individual_results"])
        for merged in all_results[domain]:
            ind = merged["individual_results"]
            row = {
                "domain": domain,
                "transcript_index": merged["transcript_index"],
                "signal_level": merged["majority_label"],
                "agreement": round(merged["agreement"], 3),
                "votes_high": merged["votes"]["HIGH"],
                "votes_low": merged["votes"]["LOW"],
                "all_agree": merged["all_agree"],
            }
            for i, model in enumerate(JUDGE_MODELS):
                prefix = model_to_col[model]
                r = ind[i]
                row[f"{prefix}_label"] = r["label"] or "LOW"
                row[f"{prefix}_criterion"] = f'"{r["criterion_quote"]}"'
            writer.writerow(row)

print(f"\nWrote {OUTPUT_PATH}")

# ── Quick agreement summary ───────────────────────────────────────────────
from math import comb
from itertools import combinations
from collections import defaultdict

rows_by_domain = defaultdict(list)
with open(OUTPUT_PATH, newline="") as f:
    for row in csv.DictReader(f):
        labels = [row["llama_label"], row["mistral_label"], row["qwen_label"]]
        if all(l in ("HIGH", "LOW") for l in labels):
            rows_by_domain[row["domain"]].append(labels)

def metrics(rows):
    N = len(rows)
    weighted  = sum(1.0 if len(set(r)) == 1 else 0.667 for r in rows) / N
    all_agree = sum(len(set(r)) == 1 for r in rows) / N
    pair      = sum(a == b for r in rows for a, b in combinations(r, 2)) / (3 * N)
    Pa        = sum((comb(r.count("HIGH"),2)+comb(r.count("LOW"),2))/comb(3,2) for r in rows)/N
    piH       = sum(r.count("HIGH") for r in rows) / (3 * N)
    piL       = 1 - piH
    Pe_f, Pe_g = piH**2 + piL**2, 2 * piH * piL
    kappa     = (Pa - Pe_f) / (1 - Pe_f) if Pe_f != 1 else float("nan")
    ac1       = (Pa - Pe_g) / (1 - Pe_g) if Pe_g != 1 else float("nan")
    return weighted, all_agree, pair, kappa, ac1

print(f"\n{'domain':28s} {'weighted':>9s} {'all-agree':>10s} {'pairwise':>9s} {'kappa':>8s} {'AC1':>8s}")
for dom in sorted(rows_by_domain):
    w, a, p, k, ac = metrics(rows_by_domain[dom])
    print(f"{dom:28s} {w:9.3f} {a:10.3f} {p:9.3f} {k:+8.3f} {ac:+8.3f}")
