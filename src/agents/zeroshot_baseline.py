#!/usr/bin/env python3
"""
Zero-shot frontier-model baseline.

Compares a single frontier model (OpenAI, no rubric, no calibration loop,
one call per transcript) against the calibrated 3-judge local panel's
majority-vote labels in calibrated_labels.csv.

Answers: does one frontier-model call reproduce what ~3 hours of local-model
rubric calibration converges on?

Input:
  data/review_queue.csv                          (60 transcripts, ordered per domain)
  reports/rubric_calibration/calibrated_labels.csv (calibrated panel's majority labels)

Output:
  reports/rubric_calibration/zeroshot_baseline_labels.csv
  reports/rubric_calibration/zeroshot_baseline_results.json

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
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
OUT_LABELS = BASE / "reports" / "rubric_calibration" / "zeroshot_baseline_labels.csv"
OUT_RESULTS = BASE / "reports" / "rubric_calibration" / "zeroshot_baseline_results.json"

MODEL = "gpt-5.6-sol"


def load_env():
    env_path = BASE / ".env"
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_review_queue() -> list:
    """transcript_index in calibrated_labels.csv is a global row index into
    this file (not per-domain), so keep it as a flat list."""
    with open(REVIEW_QUEUE, encoding="utf-8") as f:
        return [r["transcript"] for r in csv.DictReader(f)]


def load_calibrated_labels() -> list:
    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_prompt(transcript: str, domain: str) -> str:
    domain_label = domain.replace("_", " ")
    return f"""You are evaluating a YouTube video transcript from the {domain_label} domain.

Decide whether this video is HIGH signal (genuinely worth watching) or LOW signal (not worth your time). Use your own judgment.

TRANSCRIPT:
---
{transcript[:3000]}
---

Respond in EXACTLY this format, nothing else:
LABEL: HIGH or LOW"""


def call_model(client: OpenAI, prompt: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1500,
    )
    text = r.choices[0].message.content.strip()
    return "HIGH" if "LABEL: HIGH" in text else "LOW"


def agreement_metrics(pairs: list) -> dict:
    """pairs: list of (zeroshot_label, calibrated_label).

    Reports Cohen's kappa alongside Gwet's AC1 because kappa is unstable
    under high label prevalence (same prevalence paradox documented for the
    local judge panel in compute_agreement.py / the README results table).
    """
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    p_o = agree / n
    p_high_zs = sum(a == "HIGH" for a, _ in pairs) / n
    p_high_cal = sum(b == "HIGH" for _, b in pairs) / n
    p_e_kappa = p_high_zs * p_high_cal + (1 - p_high_zs) * (1 - p_high_cal)
    kappa = (p_o - p_e_kappa) / (1 - p_e_kappa) if p_e_kappa != 1 else float("nan")

    pi_high = (p_high_zs + p_high_cal) / 2
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
    calibrated = load_calibrated_labels()

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

    print(f"Running zero-shot baseline ({MODEL}) on {len(calibrated)} transcripts...\n")
    for i, row in enumerate(calibrated):
        domain = row["domain"]
        idx = int(row["transcript_index"])
        cal_label = row["signal_level"]

        if (domain, idx) in done:
            zs_label = done[(domain, idx)]["zeroshot_label"]
            match = zs_label == cal_label
            print(f"[{i+1}/{len(calibrated)}] {domain:22s} idx={idx:2d} (cached) "
                  f"calibrated={cal_label:4s} zeroshot={zs_label:4s} {'✓' if match else '✗'}")
            continue

        transcript = review_queue[idx]
        prompt = build_prompt(transcript, domain)
        zs_label = call_model(client, prompt)

        match = zs_label == cal_label
        writer.writerow({
            "domain": domain,
            "transcript_index": idx,
            "calibrated_label": cal_label,
            "zeroshot_label": zs_label,
            "match": match,
        })
        out_file.flush()

        print(f"[{i+1}/{len(calibrated)}] {domain:22s} idx={idx:2d} "
              f"calibrated={cal_label:4s} zeroshot={zs_label:4s} {'✓' if match else '✗'}")

    out_file.close()
    print(f"\nSaved per-item labels to {OUT_LABELS}")

    with open(OUT_LABELS, encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))
    pairs_by_domain = defaultdict(list)
    all_pairs = []
    for r in out_rows:
        pair = (r["zeroshot_label"], r["calibrated_label"])
        pairs_by_domain[r["domain"]].append(pair)
        all_pairs.append(pair)

    results = {
        "model": MODEL,
        "reference": "calibrated_labels.csv (calibrated 3-judge local-panel majority vote)",
        "overall": agreement_metrics(all_pairs),
        "by_domain": {d: agreement_metrics(p) for d, p in sorted(pairs_by_domain.items())},
    }
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'domain':28s} {'n':>4s} {'agreement':>10s} {'kappa':>8s} {'AC1':>8s}")
    for d, m in results["by_domain"].items():
        print(f"{d:28s} {m['n']:4d} {m['agreement']:10.3f} {m['cohen_kappa']:+8.3f} {m['gwet_ac1']:+8.3f}")
    o = results["overall"]
    print(f"{'OVERALL':28s} {o['n']:4d} {o['agreement']:10.3f} {o['cohen_kappa']:+8.3f} {o['gwet_ac1']:+8.3f}")
    print(f"\nSaved summary to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
