#!/usr/bin/env python3
"""
Frontier-model + calibrated-rubric baseline (third comparison arm).

Applies the SAME calibrated rubric text the local 3-judge panel used
(reports/rubric_calibration/calibrated_rubrics.json), but with a single
frontier-model call instead of the 3-model local ensemble. Isolates whether
the calibrated rubric itself adds value, independent of which model applies it.

Three-way comparison:
  1. local panel + rubric   -> calibrated_labels.csv (reference)
  2. frontier + no rubric   -> zeroshot_baseline_labels.csv (reference)
  3. frontier + rubric      -> this script

Input:
  data/review_queue.csv                              (60 transcripts, global index)
  reports/rubric_calibration/calibrated_rubrics.json  (same rubric text as the local judges)
  reports/rubric_calibration/calibrated_labels.csv    (local-panel majority labels)
  reports/rubric_calibration/zeroshot_baseline_labels.csv (frontier, no-rubric labels)

Output:
  reports/rubric_calibration/frontier_with_rubric_labels.csv
  reports/rubric_calibration/frontier_with_rubric_results.json

Requires OPENAI_API_KEY in a local .env file (gitignored).
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

BASE = Path(__file__).resolve().parents[2]
REVIEW_QUEUE = BASE / "data" / "review_queue.csv"
RUBRICS_PATH = BASE / "reports" / "rubric_calibration" / "calibrated_rubrics.json"
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
ZEROSHOT_LABELS = BASE / "reports" / "rubric_calibration" / "zeroshot_baseline_labels.csv"
OUT_LABELS = BASE / "reports" / "rubric_calibration" / "frontier_with_rubric_labels.csv"
OUT_RESULTS = BASE / "reports" / "rubric_calibration" / "frontier_with_rubric_results.json"

MODEL = "gpt-5.6-sol"


def load_env():
    env_path = BASE / ".env"
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_review_queue() -> list:
    """transcript_index in calibrated_labels.csv is a global row index into
    this file (not per-domain) — see zeroshot_baseline.py."""
    with open(REVIEW_QUEUE, encoding="utf-8") as f:
        return [r["transcript"] for r in csv.DictReader(f)]


def load_rubrics() -> dict:
    with open(RUBRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_calibrated_labels() -> list:
    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_zeroshot_labels() -> dict:
    with open(ZEROSHOT_LABELS, encoding="utf-8") as f:
        return {
            (r["domain"], int(r["transcript_index"])): r["zeroshot_label"]
            for r in csv.DictReader(f)
        }


def build_prompt(transcript: str, domain: str, rubric: dict) -> str:
    """Same rubric text and structure the local judge panel used
    (judge_agent.py:build_judge_prompt), applied by a single frontier call."""
    high_criteria = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["HIGH"]))
    low_criteria = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["LOW"]))

    return f"""You are an expert educational content evaluator for the {domain} domain.

DOMAIN CONTEXT: {rubric['domain_context']}

RUBRIC VERSION: {rubric['version']}

HIGH SIGNAL criteria (content must satisfy at least 2):
{high_criteria}

LOW SIGNAL criteria (any one dominant pattern = LOW):
{low_criteria}

CONFLICT RULE: {rubric['conflict_rule']}

TRANSCRIPT TO EVALUATE:
---
{transcript[:3000]}
---

Evaluate this transcript carefully against the rubric above.

Respond in this EXACT format with no other text:
LABEL: HIGH or LOW
CONFIDENCE: HIGH or MEDIUM or LOW
MATCHED_HIGH: comma-separated numbers of HIGH criteria matched (e.g. "1,3")
MATCHED_LOW: comma-separated numbers of LOW criteria matched (e.g. "2")
REASON: one sentence explaining the primary factor"""


def call_model(client: OpenAI, prompt: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,
    )
    text = r.choices[0].message.content.strip()
    return "HIGH" if "LABEL: HIGH" in text else "LOW"


def agreement_metrics(pairs: list) -> dict:
    """pairs: list of (label_a, label_b). Reports Cohen's kappa alongside
    Gwet's AC1 because kappa is unstable under high label prevalence (same
    prevalence paradox documented in compute_agreement.py)."""
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    p_o = agree / n
    p_high_a = sum(a == "HIGH" for a, _ in pairs) / n
    p_high_b = sum(b == "HIGH" for _, b in pairs) / n
    p_e_kappa = p_high_a * p_high_b + (1 - p_high_a) * (1 - p_high_b)
    kappa = (p_o - p_e_kappa) / (1 - p_e_kappa) if p_e_kappa != 1 else float("nan")

    pi_high = (p_high_a + p_high_b) / 2
    p_e_gwet = 2 * pi_high * (1 - pi_high)
    ac1 = (p_o - p_e_gwet) / (1 - p_e_gwet) if p_e_gwet != 1 else float("nan")

    return {
        "n": n,
        "agreement": round(p_o, 4),
        "cohen_kappa": round(kappa, 4),
        "gwet_ac1": round(ac1, 4),
    }


def main():
    load_env()
    client = OpenAI()

    review_queue = load_review_queue()
    rubrics = load_rubrics()
    calibrated = load_calibrated_labels()
    zeroshot = load_zeroshot_labels()

    fieldnames = ["domain", "transcript_index", "calibrated_label", "zeroshot_label", "match"]

    done = {}
    if OUT_LABELS.exists():
        with open(OUT_LABELS, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[(r["domain"], int(r["transcript_index"]))] = r
        print(f"Resuming — {len(done)} already labeled")

    out_file_exists = OUT_LABELS.exists()
    out_file = open(OUT_LABELS, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if not out_file_exists:
        writer.writeheader()

    print(f"Running frontier+rubric baseline ({MODEL}) on {len(calibrated)} transcripts...\n")
    for i, row in enumerate(calibrated):
        domain = row["domain"]
        idx = int(row["transcript_index"])
        cal_label = row["signal_level"]

        if (domain, idx) in done:
            fr_label = done[(domain, idx)]["zeroshot_label"]
            match = fr_label == cal_label
            print(f"[{i+1}/{len(calibrated)}] {domain:22s} idx={idx:2d} (cached) "
                  f"calibrated={cal_label:4s} frontier+rubric={fr_label:4s} {'✓' if match else '✗'}")
            continue

        transcript = review_queue[idx]
        rubric = rubrics[domain]
        prompt = build_prompt(transcript, domain, rubric)
        fr_label = call_model(client, prompt)

        match = fr_label == cal_label
        writer.writerow({
            "domain": domain,
            "transcript_index": idx,
            "calibrated_label": cal_label,
            "zeroshot_label": fr_label,
            "match": match,
        })
        out_file.flush()

        print(f"[{i+1}/{len(calibrated)}] {domain:22s} idx={idx:2d} "
              f"calibrated={cal_label:4s} frontier+rubric={fr_label:4s} {'✓' if match else '✗'}")

    out_file.close()
    print(f"\nSaved per-item labels to {OUT_LABELS}")

    with open(OUT_LABELS, encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))

    pairs_vs_calibrated_by_domain = defaultdict(list)
    pairs_vs_calibrated_all = []
    pairs_vs_zeroshot_by_domain = defaultdict(list)
    pairs_vs_zeroshot_all = []

    for r in out_rows:
        key = (r["domain"], int(r["transcript_index"]))
        fr_label = r["zeroshot_label"]

        cal_pair = (fr_label, r["calibrated_label"])
        pairs_vs_calibrated_by_domain[r["domain"]].append(cal_pair)
        pairs_vs_calibrated_all.append(cal_pair)

        zs_pair = (fr_label, zeroshot[key])
        pairs_vs_zeroshot_by_domain[r["domain"]].append(zs_pair)
        pairs_vs_zeroshot_all.append(zs_pair)

    results = {
        "model": MODEL,
        "reference": "calibrated_labels.csv (calibrated 3-judge local-panel majority vote)",
        "overall": agreement_metrics(pairs_vs_calibrated_all),
        "by_domain": {
            d: agreement_metrics(p) for d, p in sorted(pairs_vs_calibrated_by_domain.items())
        },
        "vs_zeroshot_no_rubric": {
            "reference": "zeroshot_baseline_labels.csv (same model, naive prompt, no rubric)",
            "overall": agreement_metrics(pairs_vs_zeroshot_all),
            "by_domain": {
                d: agreement_metrics(p) for d, p in sorted(pairs_vs_zeroshot_by_domain.items())
            },
        },
    }
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("vs. calibrated local-panel majority (calibrated_labels.csv)")
    print(f"{'domain':28s} {'n':>4s} {'agreement':>10s} {'kappa':>8s} {'AC1':>8s}")
    for d, m in results["by_domain"].items():
        print(f"{d:28s} {m['n']:4d} {m['agreement']:10.3f} {m['cohen_kappa']:+8.3f} {m['gwet_ac1']:+8.3f}")
    o = results["overall"]
    print(f"{'OVERALL':28s} {o['n']:4d} {o['agreement']:10.3f} {o['cohen_kappa']:+8.3f} {o['gwet_ac1']:+8.3f}")

    print(f"\nvs. same model with NO rubric (zeroshot_baseline_labels.csv)")
    print(f"{'domain':28s} {'n':>4s} {'agreement':>10s} {'kappa':>8s} {'AC1':>8s}")
    vz = results["vs_zeroshot_no_rubric"]
    for d, m in vz["by_domain"].items():
        print(f"{d:28s} {m['n']:4d} {m['agreement']:10.3f} {m['cohen_kappa']:+8.3f} {m['gwet_ac1']:+8.3f}")
    o2 = vz["overall"]
    print(f"{'OVERALL':28s} {o2['n']:4d} {o2['agreement']:10.3f} {o2['cohen_kappa']:+8.3f} {o2['gwet_ac1']:+8.3f}")

    print(f"\nSaved summary to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
