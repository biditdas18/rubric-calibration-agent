#!/usr/bin/env python3
"""
CoT ablation across all three domains: does forcing step-by-step reasoning
before the label change inter-judge agreement and human-ground-truth
accuracy, compared to the existing label-then-justify judge prompt?

Originally scoped to career_selfimprovement only (the domain that stagnated
at 68.4% weighted agreement — Section 5.1 capability-ceiling finding).
Extended to run the same protocol on tech_ai and general_education too, so
the CoT effect can be compared against domains that DID converge during
calibration, not just the one that didn't.

Same 3 local judges, same temperature=0.0, same calibrated rubric per domain
(reports/rubric_calibration/calibrated_rubrics.json). Only the prompt
changes: judges must write reasoning in a <scratchpad> before the final
LABEL line, instead of label-then-justify.

Does not re-run the calibration loop — scores once against each domain's
existing rubric. career_selfimprovement's label file already exists from
the original single-domain run and is reused via the resume mechanism
(temp=0 local models are deterministic, so recomputing would just burn
time for the same result).

Input:
  data/review_queue.csv
  reports/rubric_calibration/calibrated_rubrics.json
  reports/rubric_calibration/calibrated_labels.csv       (non-CoT baseline)
  reports/rubric_calibration/calibration_summary.json    (non-CoT weighted baselines)
  gold-sampled-dataset/human_validation_ground_truth.csv (30 rows, 10/domain)

Output (per domain):
  reports/rubric_calibration/cot_{slug}_labels.csv   (same column format as calibrated_labels.csv)
Output (combined):
  reports/rubric_calibration/cot_all_domains_results.json

Requires Ollama running locally with llama3.1, mistral, qwen2.5:7b pulled.
"""
import csv
import json
from itertools import combinations
from math import comb
from pathlib import Path

import ollama

from zeroshot_baseline import agreement_metrics

BASE = Path(__file__).resolve().parents[2]
REVIEW_QUEUE = BASE / "data" / "review_queue.csv"
RUBRICS_PATH = BASE / "reports" / "rubric_calibration" / "calibrated_rubrics.json"
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
CALIBRATION_SUMMARY = BASE / "reports" / "rubric_calibration" / "calibration_summary.json"
HUMAN_GOLD = BASE / "gold-sampled-dataset" / "human_validation_ground_truth.csv"
REPORT_DIR = BASE / "reports" / "rubric_calibration"
OUT_RESULTS = REPORT_DIR / "cot_all_domains_results.json"

DOMAINS = ["career_selfimprovement", "tech_ai", "general_education"]
# career keeps its original filename (cot_career_labels.csv, from the single-domain run)
DOMAIN_FILE_SLUG = {"career_selfimprovement": "career", "tech_ai": "tech_ai", "general_education": "general_education"}

JUDGE_MODELS = ["llama3.1:latest", "mistral:latest", "qwen2.5:7b"]
MODEL_COL = {"llama3.1:latest": "llama", "mistral:latest": "mistral", "qwen2.5:7b": "qwen"}
TEMPERATURE = 0.0
NUM_PREDICT = 900  # higher than judge_agent.py's 300 — scratchpad reasoning needs the room

LABEL_FIELDNAMES = [
    "domain", "transcript_index", "signal_level", "agreement",
    "votes_high", "votes_low", "all_agree",
    "llama_label", "mistral_label", "qwen_label",
    "llama_criterion", "mistral_criterion", "qwen_criterion",
]


def out_labels_path(domain: str) -> Path:
    return REPORT_DIR / f"cot_{DOMAIN_FILE_SLUG[domain]}_labels.csv"


def load_domain_transcripts(domain: str) -> list:
    """[(global_index, transcript), ...] — global_index matches the
    transcript_index scheme used throughout calibrated_labels.csv etc."""
    with open(REVIEW_QUEUE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(i, r["transcript"]) for i, r in enumerate(rows) if r["domain"] == domain]


def load_rubric(domain: str) -> dict:
    with open(RUBRICS_PATH, encoding="utf-8") as f:
        return json.load(f)[domain]


def build_cot_prompt(transcript: str, domain: str, rubric: dict) -> str:
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

Think through this step by step BEFORE deciding on a label. Inside <scratchpad></scratchpad> tags:
1. List which HIGH criteria apply and point to where in the transcript.
2. List which LOW criteria apply and point to where in the transcript.
3. Explicitly check whether motivational language co-occurs with procedural content (the conflict rule above).
Do not state your final label inside the scratchpad — reasoning only.

AFTER the closing </scratchpad> tag, respond with the final answer in this EXACT format:
LABEL: HIGH or LOW
CONFIDENCE: HIGH or MEDIUM or LOW
MATCHED_HIGH: comma-separated numbers of HIGH criteria matched (e.g. "1,3")
MATCHED_LOW: comma-separated numbers of LOW criteria matched (e.g. "2")
REASON: one sentence explaining the primary factor"""


def parse_response(text: str) -> dict:
    text = text.strip()
    tail = text.split("</scratchpad>")[-1] if "</scratchpad>" in text else text

    result = {"label": None, "confidence": None, "reason": ""}
    for line in tail.split("\n"):
        line = line.strip()
        if line.startswith("LABEL:"):
            val = line.replace("LABEL:", "").strip().upper()
            result["label"] = val if val in ["HIGH", "LOW"] else None
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip()
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()

    if result["label"] is None:
        tail_upper = tail.upper()
        if "LABEL: HIGH" in tail_upper:
            result["label"] = "HIGH"
        elif "LABEL: LOW" in tail_upper:
            result["label"] = "LOW"
        elif "HIGH" in tail_upper and "LOW" not in tail_upper:
            result["label"] = "HIGH"
        else:
            result["label"] = "LOW"  # conservative default — matches judge_agent.py's error handling

    return result


def judge_one(model: str, transcript: str, domain: str, rubric: dict) -> dict:
    prompt = build_cot_prompt(transcript, domain, rubric)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        )
        parsed = parse_response(response["message"]["content"])
        parsed["model"] = model
        return parsed
    except Exception as e:
        return {"model": model, "label": "LOW", "confidence": "LOW", "reason": f"Error: {str(e)[:150]}"}


def fleiss_style_metrics(rows_of_3_labels: list) -> dict:
    """Same formulas as compute_agreement.py's metrics(), copied inline so
    numbers are directly comparable without importing a script that has
    print-on-import side effects."""
    N = len(rows_of_3_labels)
    weighted = sum(1.0 if len(set(r)) == 1 else 0.667 for r in rows_of_3_labels) / N
    all_agree = sum(len(set(r)) == 1 for r in rows_of_3_labels) / N
    pairwise = sum(a == b for r in rows_of_3_labels for a, b in combinations(r, 2)) / (3 * N)
    Pa = sum((comb(r.count("HIGH"), 2) + comb(r.count("LOW"), 2)) / comb(3, 2) for r in rows_of_3_labels) / N
    piH = sum(r.count("HIGH") for r in rows_of_3_labels) / (3 * N)
    piL = 1 - piH
    Pe_f, Pe_g = piH**2 + piL**2, 2 * piH * piL
    kappa = (Pa - Pe_f) / (1 - Pe_f) if Pe_f != 1 else float("nan")
    ac1 = (Pa - Pe_g) / (1 - Pe_g) if Pe_g != 1 else float("nan")
    return {
        "n": N,
        "weighted": round(weighted, 4),
        "all_agree": round(all_agree, 4),
        "pairwise": round(pairwise, 4),
        "fleiss_kappa": round(kappa, 4),
        "gwet_ac1": round(ac1, 4),
    }


def run_domain(domain: str) -> list:
    """Runs (or resumes) the CoT judge panel for one domain. Returns the
    full list of label rows (dicts) read back from the output CSV."""
    transcripts = load_domain_transcripts(domain)
    assert len(transcripts) == 20, f"expected 20 {domain} transcripts, got {len(transcripts)}"
    rubric = load_rubric(domain)
    labels_path = out_labels_path(domain)

    done = set()
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            done = {int(r["transcript_index"]) for r in csv.DictReader(f)}
        print(f"[{domain}] Resuming — {len(done)}/20 already labeled")

    file_exists = labels_path.exists()
    out_file = open(labels_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=LABEL_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    print(f"[{domain}] Running CoT judges on {len(transcripts)} transcripts "
          f"({', '.join(JUDGE_MODELS)}, temp={TEMPERATURE})...")

    for i, (idx, transcript) in enumerate(transcripts):
        if idx in done:
            continue

        per_model = {model: judge_one(model, transcript, domain, rubric) for model in JUDGE_MODELS}

        labels = [per_model[m]["label"] for m in JUDGE_MODELS]
        high_count = labels.count("HIGH")
        low_count = labels.count("LOW")
        majority = "HIGH" if high_count >= 2 else "LOW"
        agreement = max(high_count, low_count) / 3
        all_agree = len(set(labels)) == 1

        row = {
            "domain": domain,
            "transcript_index": idx,
            "signal_level": majority,
            "agreement": round(agreement, 3),
            "votes_high": high_count,
            "votes_low": low_count,
            "all_agree": all_agree,
        }
        for model, col in MODEL_COL.items():
            row[f"{col}_label"] = per_model[model]["label"]
            row[f"{col}_criterion"] = per_model[model]["reason"]

        writer.writerow(row)
        out_file.flush()

        votes_str = " ".join(f"{MODEL_COL[m]}={per_model[m]['label']}" for m in JUDGE_MODELS)
        print(f"[{domain}] [{i+1}/{len(transcripts)}] idx={idx:2d} {majority:4s} ({votes_str}) "
              f"{'✓all-agree' if all_agree else ''}")

    out_file.close()
    print(f"[{domain}] Saved per-item CoT labels to {labels_path}\n")

    with open(labels_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_domain(domain: str, cot_rows: list) -> dict:
    cot_judge_rows = [[r["llama_label"], r["mistral_label"], r["qwen_label"]] for r in cot_rows]
    cot_inter_judge = fleiss_style_metrics(cot_judge_rows)

    with open(CALIBRATION_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)
    baseline_weighted = summary["domains"][domain]["final_weighted_agreement"]

    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        non_cot_rows = [r for r in csv.DictReader(f) if r["domain"] == domain]
    non_cot_judge_rows = [[r["llama_label"], r["mistral_label"], r["qwen_label"]] for r in non_cot_rows]
    non_cot_inter_judge = fleiss_style_metrics(non_cot_judge_rows)

    with open(HUMAN_GOLD, encoding="utf-8") as f:
        human_rows = [r for r in csv.DictReader(f) if r["domain"] == domain]
    human_by_idx = {int(r["transcript_index"]): r["human_consensus_label"] for r in human_rows}

    cot_by_idx = {int(r["transcript_index"]): r["signal_level"] for r in cot_rows}
    non_cot_by_idx = {int(r["transcript_index"]): r["signal_level"] for r in non_cot_rows}

    cot_vs_human_pairs = [(cot_by_idx[i], human_by_idx[i]) for i in human_by_idx]
    non_cot_vs_human_pairs = [(non_cot_by_idx[i], human_by_idx[i]) for i in human_by_idx]

    return {
        "domain": domain,
        "n_transcripts": len(cot_rows),
        "rubric_version": load_rubric(domain)["version"],
        "inter_judge_agreement": {
            "with_cot": cot_inter_judge,
            "without_cot_recomputed_from_calibrated_labels_csv": non_cot_inter_judge,
            "without_cot_baseline_from_calibration_summary_json": baseline_weighted,
            "weighted_delta_vs_calibration_summary_baseline": round(
                cot_inter_judge["weighted"] - baseline_weighted, 4
            ),
        },
        "vs_human_ground_truth": {
            "n": len(human_by_idx),
            "cot_majority_vote": agreement_metrics(cot_vs_human_pairs),
            "non_cot_majority_vote": agreement_metrics(non_cot_vs_human_pairs),
        },
    }


def main():
    per_domain_results = {}
    for domain in DOMAINS:
        cot_rows = run_domain(domain)
        per_domain_results[domain] = score_domain(domain, cot_rows)

    results = {
        "judge_models": JUDGE_MODELS,
        "temperature": TEMPERATURE,
        "prompting": "chain-of-thought (scratchpad reasoning before label)",
        "domains": per_domain_results,
    }

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    for domain, r in per_domain_results.items():
        ij = r["inter_judge_agreement"]
        vh = r["vs_human_ground_truth"]
        print(f"\n{domain} (n={r['n_transcripts']}):")
        print(f"  Inter-judge  CoT: weighted={ij['with_cot']['weighted']:.3f} "
              f"all_agree={ij['with_cot']['all_agree']:.3f} "
              f"kappa={ij['with_cot']['fleiss_kappa']:+.3f} AC1={ij['with_cot']['gwet_ac1']:+.3f}")
        print(f"  Inter-judge  No CoT (baseline): weighted={ij['without_cot_baseline_from_calibration_summary_json']:.3f} "
              f"| recomputed: weighted={ij['without_cot_recomputed_from_calibrated_labels_csv']['weighted']:.3f} "
              f"AC1={ij['without_cot_recomputed_from_calibrated_labels_csv']['gwet_ac1']:+.3f}")
        print(f"  vs human (n={vh['n']})  CoT: agreement={vh['cot_majority_vote']['agreement']:.3f} "
              f"AC1={vh['cot_majority_vote']['gwet_ac1']:+.3f} | "
              f"No CoT: agreement={vh['non_cot_majority_vote']['agreement']:.3f} "
              f"AC1={vh['non_cot_majority_vote']['gwet_ac1']:+.3f}")

    print(f"\nSaved combined summary to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
