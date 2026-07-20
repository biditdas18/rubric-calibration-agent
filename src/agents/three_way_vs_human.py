#!/usr/bin/env python3
"""
Scores all three labeling approaches against the human-validated gold sample
(30 transcripts, 10/domain) instead of the calibrated local-panel majority vote.

Approaches compared:
  1. local panel + rubric   -> calibrated_labels.csv (signal_level)
  2. frontier, no rubric    -> zeroshot_baseline_labels.csv (zeroshot_label)
  3. frontier + rubric      -> frontier_with_rubric_labels.csv (zeroshot_label)

Reference: gold-sampled-dataset/human_validation_ground_truth.csv (human_consensus_label)

Output: reports/rubric_calibration/three_way_vs_human_results.json

Pure aggregation over existing label files — makes no API calls.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from zeroshot_baseline import agreement_metrics

BASE = Path(__file__).resolve().parents[2]
HUMAN_GOLD = BASE / "gold-sampled-dataset" / "human_validation_ground_truth.csv"
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
ZEROSHOT_LABELS = BASE / "reports" / "rubric_calibration" / "zeroshot_baseline_labels.csv"
FRONTIER_RUBRIC_LABELS = BASE / "reports" / "rubric_calibration" / "frontier_with_rubric_labels.csv"
OUT_RESULTS = BASE / "reports" / "rubric_calibration" / "three_way_vs_human_results.json"


def load_keyed(path: Path, label_col: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return {
            (r["domain"], int(r["transcript_index"])): r[label_col]
            for r in csv.DictReader(f)
        }


def load_human_gold() -> dict:
    with open(HUMAN_GOLD, encoding="utf-8") as f:
        return {
            (r["domain"], int(r["transcript_index"])): r["human_consensus_label"]
            for r in csv.DictReader(f)
        }


def score(approach_labels: dict, human_labels: dict) -> dict:
    pairs_by_domain = defaultdict(list)
    pairs_all = []
    for key, human_label in human_labels.items():
        pair = (approach_labels[key], human_label)
        pairs_by_domain[key[0]].append(pair)
        pairs_all.append(pair)
    return {
        "overall": agreement_metrics(pairs_all),
        "by_domain": {d: agreement_metrics(p) for d, p in sorted(pairs_by_domain.items())},
    }


def main():
    human_gold = load_human_gold()
    calibrated = load_keyed(CALIBRATED_LABELS, "signal_level")
    zeroshot = load_keyed(ZEROSHOT_LABELS, "zeroshot_label")
    frontier_rubric = load_keyed(FRONTIER_RUBRIC_LABELS, "zeroshot_label")

    missing = [k for k in human_gold if k not in calibrated or k not in zeroshot or k not in frontier_rubric]
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} human-gold keys: {missing[:5]}...")

    results = {
        "reference": "gold-sampled-dataset/human_validation_ground_truth.csv (human_consensus_label, n=30, 10/domain)",
        "approaches": {
            "local_panel_plus_rubric": {
                "source": "calibrated_labels.csv",
                **score(calibrated, human_gold),
            },
            "frontier_no_rubric": {
                "source": "zeroshot_baseline_labels.csv",
                "model": "gpt-5.6-sol",
                **score(zeroshot, human_gold),
            },
            "frontier_plus_rubric": {
                "source": "frontier_with_rubric_labels.csv",
                "model": "gpt-5.6-sol",
                **score(frontier_rubric, human_gold),
            },
        },
    }

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Scored against human_consensus_label (n=30, 10/domain)\n")
    for name, data in results["approaches"].items():
        print(f"=== {name} ({data['source']}) ===")
        print(f"{'domain':28s} {'n':>4s} {'agreement':>10s} {'kappa':>8s} {'AC1':>8s}")
        for d, m in data["by_domain"].items():
            print(f"{d:28s} {m['n']:4d} {m['agreement']:10.3f} {m['cohen_kappa']:+8.3f} {m['gwet_ac1']:+8.3f}")
        o = data["overall"]
        print(f"{'OVERALL':28s} {o['n']:4d} {o['agreement']:10.3f} {o['cohen_kappa']:+8.3f} {o['gwet_ac1']:+8.3f}")
        print()

    print(f"Saved to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
