#!/usr/bin/env python3
"""
A3 -- Check for existing frontier-ensemble labels against the human
reference set.

Verified during recon: snr-detector's 60-transcript corpus
(data/labels/review_queue.csv) is byte-identical, same order, to
CRUCIBLE's 60-transcript corpus (rubric-calibration-agent/data/review_queue.csv).
all_150_silver_labels.csv's "original_60" rows are written in that same
order (label_150_transcripts.py iterates the original CSV first, in order,
before the new-90 supplement) -- confirmed here by transcript-text match
before use, not assumed.

Computes, for the 30-item human validation set (10/domain):
  (a) frontier-ensemble majority vote vs. human majority consensus
  (b) each frontier judge individually (GPT-4o mini, Gemini 2.5 Flash,
      Llama 3.3 70B via Groq) vs. human majority
  (c) local calibration panel vs. frontier panel agreement, same 30 items

Output: results/frontier_vs_human_validation.json
Makes no model calls -- pure join and metric computation on existing files.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from experiment_utils import two_rater_agreement_metrics

BASE = Path(__file__).resolve().parents[2]
SNR_BASE = BASE.parent / "snr-detector"
CRUCIBLE_REVIEW_QUEUE = BASE / "data" / "review_queue.csv"
SNR_SILVER_LABELS = SNR_BASE / "data" / "labels" / "all_150_silver_labels.csv"
HUMAN_GOLD = BASE / "gold-sampled-dataset" / "human_validation_ground_truth.csv"
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
OUT_PATH = BASE / "results" / "frontier_vs_human_validation.json"


def verify_and_build_index_map() -> list:
    """Returns the silver_labels 'original_60' rows in CRUCIBLE global-index
    order (0-59), verified by exact transcript-text match against
    review_queue.csv at each position -- not assumed from file order alone."""
    with open(CRUCIBLE_REVIEW_QUEUE, encoding="utf-8") as f:
        crucible_rows = list(csv.DictReader(f))
    with open(SNR_SILVER_LABELS, encoding="utf-8") as f:
        silver_rows = [r for r in csv.DictReader(f) if r["source"] == "original_60"]

    assert len(crucible_rows) == 60 and len(silver_rows) == 60, \
        f"expected 60/60, got {len(crucible_rows)}/{len(silver_rows)}"

    mismatches = [i for i in range(60)
                  if crucible_rows[i]["transcript"].strip() != silver_rows[i]["transcript"].strip()]
    if mismatches:
        raise ValueError(f"Transcript mismatch at positions {mismatches[:5]} -- "
                          f"cannot assume positional alignment between corpora.")
    return silver_rows  # index i in this list == global transcript_index i


def main():
    silver_by_idx = verify_and_build_index_map()

    with open(HUMAN_GOLD, encoding="utf-8") as f:
        human_rows = list(csv.DictReader(f))
    human_keys = {(r["domain"], int(r["transcript_index"])) for r in human_rows}
    human_by_key = {(r["domain"], int(r["transcript_index"])): r["human_consensus_label"] for r in human_rows}

    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        local_by_key = {(r["domain"], int(r["transcript_index"])): r["signal_level"]
                         for r in csv.DictReader(f)}

    # Confirm which of the 30 human-validation items have valid frontier labels
    missing = []
    frontier_majority_pairs, frontier_indiv_pairs = [], {"openai_label": [], "gemini_label": [], "groq_label": []}
    frontier_vs_local_pairs = []
    by_domain_majority = defaultdict(list)
    by_domain_indiv = {k: defaultdict(list) for k in frontier_indiv_pairs}

    with open(CRUCIBLE_REVIEW_QUEUE, encoding="utf-8") as f:
        crucible_rows = list(csv.DictReader(f))

    for i, (crow, srow) in enumerate(zip(crucible_rows, silver_by_idx)):
        domain = crow["domain"]
        key = (domain, i)
        if key not in human_keys:
            continue
        human_label = human_by_key[key]
        frontier_majority = srow["signal_level"]
        if frontier_majority not in ("HIGH", "LOW"):
            missing.append(key)
            continue

        frontier_majority_pairs.append((frontier_majority, human_label))
        by_domain_majority[domain].append((frontier_majority, human_label))

        for col in frontier_indiv_pairs:
            lbl = srow.get(col, "")
            if lbl in ("HIGH", "LOW"):
                frontier_indiv_pairs[col].append((lbl, human_label))
                by_domain_indiv[col][domain].append((lbl, human_label))

        local_label = local_by_key.get(key)
        if local_label in ("HIGH", "LOW"):
            frontier_vs_local_pairs.append((frontier_majority, local_label))

    results = {
        "experiment": "A3 -- frontier-ensemble labels vs. human reference set",
        "corpora_overlap": "FULL: snr-detector's 60-transcript corpus is byte-identical, "
                            "same order, to CRUCIBLE's 60-transcript corpus (verified by "
                            "exact transcript-text match at all 60 positions before use).",
        "n_human_validation_items": len(human_rows),
        "n_items_with_valid_frontier_labels": len(frontier_majority_pairs),
        "items_missing_or_invalid_frontier_label": missing,
        "frontier_models": "GPT-4o mini (openai_label), Gemini 2.5 Flash (gemini_label), "
                            "Llama 3.3 70B via Groq (groq_label)",
        "a_frontier_majority_vs_human": {
            "overall": two_rater_agreement_metrics(frontier_majority_pairs),
            "by_domain": {d: two_rater_agreement_metrics(p) for d, p in sorted(by_domain_majority.items())},
        },
        "b_individual_frontier_judges_vs_human": {
            col: {
                "overall": two_rater_agreement_metrics(pairs) if pairs else None,
                "by_domain": {d: two_rater_agreement_metrics(p) for d, p in sorted(by_domain_indiv[col].items())},
            }
            for col, pairs in frontier_indiv_pairs.items()
        },
        "c_local_panel_vs_frontier_panel": two_rater_agreement_metrics(frontier_vs_local_pairs),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"n items with valid frontier labels: {len(frontier_majority_pairs)}/{len(human_rows)}")
    print(f"(a) frontier majority vs human: {results['a_frontier_majority_vs_human']['overall']}")
    for col in frontier_indiv_pairs:
        print(f"(b) {col} vs human: {results['b_individual_frontier_judges_vs_human'][col]['overall']}")
    print(f"(c) local panel vs frontier panel: {results['c_local_panel_vs_frontier_panel']}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
