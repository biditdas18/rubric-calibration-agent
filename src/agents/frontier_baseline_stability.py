#!/usr/bin/env python3
"""
Frontier baseline stability check.

IMPORTANT: gpt-5.6-sol does not support temperature=0 (see zeroshot_baseline.py /
frontier_with_rubric_baseline.py — both already call it at default sampling).
This script characterizes label stability under Sol's DEFAULT sampling
settings across repeated calls — it is NOT a controlled-temperature variance
study. Any instability measured here could come from default-temperature
sampling noise, non-determinism in the served model, or genuine borderline
cases; this script cannot distinguish those causes.

Re-runs both the no-rubric and rubric-prompted zero-shot baselines 5x per
transcript (all 60), reusing the exact prompts/model-callers from
zeroshot_baseline.py and frontier_with_rubric_baseline.py. Reports:
  1. Per-transcript label stability (e.g. "5/5 same" vs "3/5 majority").
  2. Majority-vote-across-5-repeats accuracy vs. human_validation_ground_truth.csv
     (n=30), compared against the original single-draw result already saved
     in three_way_vs_human_results.json.

Input:
  data/review_queue.csv
  reports/rubric_calibration/calibrated_rubrics.json
  gold-sampled-dataset/human_validation_ground_truth.csv
  reports/rubric_calibration/three_way_vs_human_results.json (original single-draw reference)

Output:
  reports/rubric_calibration/frontier_baseline_stability_raw.csv  (resumable per-repeat cache)
  reports/rubric_calibration/frontier_baseline_stability.json

Requires OPENAI_API_KEY in a local .env file (gitignored).
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI

from zeroshot_baseline import (
    load_env,
    build_prompt as build_prompt_no_rubric,
    call_model as call_model_no_rubric,
    agreement_metrics,
)
from frontier_with_rubric_baseline import (
    load_rubrics,
    build_prompt as build_prompt_rubric,
    call_model as call_model_rubric,
)

BASE = Path(__file__).resolve().parents[2]
REVIEW_QUEUE = BASE / "data" / "review_queue.csv"
HUMAN_GOLD = BASE / "gold-sampled-dataset" / "human_validation_ground_truth.csv"
THREE_WAY_RESULTS = BASE / "reports" / "rubric_calibration" / "three_way_vs_human_results.json"
OUT_RAW = BASE / "reports" / "rubric_calibration" / "frontier_baseline_stability_raw.csv"
OUT_RESULTS = BASE / "reports" / "rubric_calibration" / "frontier_baseline_stability.json"

MODEL = "gpt-5.6-sol"
N_REPEATS = 5

VARIANTS = {
    "zeroshot_no_rubric": {"build": build_prompt_no_rubric, "call": call_model_no_rubric, "needs_rubric": False},
    "frontier_with_rubric": {"build": build_prompt_rubric, "call": call_model_rubric, "needs_rubric": True},
}


def load_review_queue_with_domains() -> list:
    with open(REVIEW_QUEUE, encoding="utf-8") as f:
        return [(i, r["domain"], r["transcript"]) for i, r in enumerate(csv.DictReader(f))]


def load_human_gold_keys() -> dict:
    with open(HUMAN_GOLD, encoding="utf-8") as f:
        return {
            (r["domain"], int(r["transcript_index"])): r["human_consensus_label"]
            for r in csv.DictReader(f)
        }


def main():
    load_env()
    client = OpenAI()

    review_queue = load_review_queue_with_domains()
    rubrics = load_rubrics()
    human_gold = load_human_gold_keys()

    fieldnames = ["variant", "domain", "transcript_index", "repeat", "label"]
    done = set()
    if OUT_RAW.exists():
        with open(OUT_RAW, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["variant"], r["domain"], int(r["transcript_index"]), int(r["repeat"])))
        print(f"Resuming — {len(done)} draws already collected")

    out_file_exists = OUT_RAW.exists()
    out_file = open(OUT_RAW, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if not out_file_exists:
        writer.writeheader()

    total_calls = len(review_queue) * len(VARIANTS) * N_REPEATS
    call_i = 0
    print(f"Running {N_REPEATS}x repeats x {len(VARIANTS)} variants x {len(review_queue)} "
          f"transcripts = {total_calls} calls ({MODEL}, default sampling)...\n")

    for idx, domain, transcript in review_queue:
        for variant_name, v in VARIANTS.items():
            for repeat in range(N_REPEATS):
                call_i += 1
                key = (variant_name, domain, idx, repeat)
                if key in done:
                    continue
                if v["needs_rubric"]:
                    prompt = v["build"](transcript, domain, rubrics[domain])
                else:
                    prompt = v["build"](transcript, domain)
                label = v["call"](client, prompt)

                writer.writerow({
                    "variant": variant_name, "domain": domain,
                    "transcript_index": idx, "repeat": repeat, "label": label,
                })
                out_file.flush()

                if call_i % 20 == 0 or call_i == total_calls:
                    print(f"[{call_i}/{total_calls}] {variant_name} {domain} idx={idx} repeat={repeat} -> {label}")

    out_file.close()
    print(f"\nSaved raw draws to {OUT_RAW}")

    with open(OUT_RAW, encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    draws = defaultdict(dict)  # (variant, domain, idx) -> {repeat: label}
    for r in raw_rows:
        draws[(r["variant"], r["domain"], int(r["transcript_index"]))][int(r["repeat"])] = r["label"]

    per_transcript_stability = defaultdict(list)
    majority_labels = {}  # (variant, domain, idx) -> majority label

    for (variant, domain, idx), repeats in draws.items():
        labels = [repeats[i] for i in sorted(repeats)]
        assert len(labels) == N_REPEATS, f"expected {N_REPEATS} draws for {(variant, domain, idx)}, got {len(labels)}"
        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        majority_labels[(variant, domain, idx)] = majority_label

        per_transcript_stability[variant].append({
            "domain": domain,
            "transcript_index": idx,
            "labels": labels,
            "majority_label": majority_label,
            "stability": f"{majority_count}/{N_REPEATS} " + ("same" if majority_count == N_REPEATS else "majority"),
            "n_unique_labels": len(counts),
        })

    for variant in VARIANTS:
        per_transcript_stability[variant].sort(key=lambda r: (r["domain"], r["transcript_index"]))

    def stability_summary(variant: str) -> dict:
        items = per_transcript_stability[variant]
        n = len(items)
        fully_stable = sum(1 for r in items if r["n_unique_labels"] == 1)
        mean_ratio = sum(int(r["stability"].split("/")[0]) for r in items) / n / N_REPEATS
        by_domain = defaultdict(list)
        for r in items:
            by_domain[r["domain"]].append(r)
        return {
            "n_transcripts": n,
            "fully_stable_5_of_5": fully_stable,
            "fully_stable_fraction": round(fully_stable / n, 4),
            "mean_stability_ratio": round(mean_ratio, 4),
            "by_domain": {
                d: {
                    "n": len(rows),
                    "fully_stable_5_of_5": sum(1 for r in rows if r["n_unique_labels"] == 1),
                    "mean_stability_ratio": round(
                        sum(int(r["stability"].split("/")[0]) for r in rows) / len(rows) / N_REPEATS, 4
                    ),
                }
                for d, rows in sorted(by_domain.items())
            },
        }

    with open(THREE_WAY_RESULTS, encoding="utf-8") as f:
        three_way = json.load(f)
    original_single_draw = {
        "zeroshot_no_rubric": three_way["approaches"]["frontier_no_rubric"],
        "frontier_with_rubric": three_way["approaches"]["frontier_plus_rubric"],
    }

    accuracy_vs_human = {}
    for variant in VARIANTS:
        pairs_all = []
        pairs_by_domain = defaultdict(list)
        for (v, domain, idx), human_label in (
            (k, human_gold[(k[1], k[2])]) for k in majority_labels if k[0] == variant and (k[1], k[2]) in human_gold
        ):
            pair = (majority_labels[(v, domain, idx)], human_label)
            pairs_all.append(pair)
            pairs_by_domain[domain].append(pair)

        accuracy_vs_human[variant] = {
            "majority_of_5": {
                "overall": agreement_metrics(pairs_all),
                "by_domain": {d: agreement_metrics(p) for d, p in sorted(pairs_by_domain.items())},
            },
            "original_single_draw": {
                "overall": original_single_draw[variant]["overall"],
                "by_domain": original_single_draw[variant]["by_domain"],
            },
        }

    results = {
        "model": MODEL,
        "note": (
            "gpt-5.6-sol does not support temperature=0. This characterizes label "
            "stability under DEFAULT sampling settings across N repeated calls per "
            "transcript, NOT controlled-temperature variance. Instability observed "
            "here may reflect default-temperature sampling noise, serving-side "
            "non-determinism, or genuine borderline cases -- this script cannot "
            "distinguish between those causes."
        ),
        "n_repeats": N_REPEATS,
        "n_transcripts": len(review_queue),
        "stability_summary": {variant: stability_summary(variant) for variant in VARIANTS},
        "per_transcript_stability": dict(per_transcript_stability),
        "accuracy_vs_human_n30": accuracy_vs_human,
    }

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    for variant in VARIANTS:
        s = results["stability_summary"][variant]
        print(f"\n{variant}: {s['fully_stable_5_of_5']}/{s['n_transcripts']} fully stable (5/5 same), "
              f"mean stability ratio={s['mean_stability_ratio']:.3f}")
        for d, m in s["by_domain"].items():
            print(f"  {d:28s} fully_stable={m['fully_stable_5_of_5']:2d}/{m['n']:2d}  "
                  f"mean_ratio={m['mean_stability_ratio']:.3f}")

        av = accuracy_vs_human[variant]
        print(f"  vs human (n=30): majority-of-5 agreement={av['majority_of_5']['overall']['agreement']:.3f} "
              f"AC1={av['majority_of_5']['overall']['gwet_ac1']:+.3f}  |  "
              f"single-draw agreement={av['original_single_draw']['overall']['agreement']:.3f} "
              f"AC1={av['original_single_draw']['overall']['gwet_ac1']:+.3f}")

    print(f"\nSaved summary to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
