#!/usr/bin/env python3
"""
CoT ablation: does forcing step-by-step reasoning before the label change
Career & Self-Improvement's inter-judge agreement, which stagnated at 68.4%
weighted agreement in the main calibration run (Section 5.1 capability-ceiling
finding)? Tests whether that stagnation is a prompting artifact rather than a
true model capability limit.

Same 3 local judges, same temperature=0.0, same calibrated rubric
(reports/rubric_calibration/calibrated_rubrics.json, career_selfimprovement —
the "best retained" v1 rubric, since this domain never converged). Only the
prompt changes: judges must write reasoning in a <scratchpad> before the
final LABEL line, instead of label-then-justify.

Scope: career_selfimprovement only (20 transcripts). Does not re-run the
calibration loop — scores once against the existing rubric.

Input:
  data/review_queue.csv
  reports/rubric_calibration/calibrated_rubrics.json
  reports/rubric_calibration/calibrated_labels.csv       (non-CoT baseline, same domain)
  reports/rubric_calibration/calibration_summary.json    (68.4% weighted baseline)
  gold-sampled-dataset/human_validation_ground_truth.csv (10 career rows)

Output:
  reports/rubric_calibration/cot_career_labels.csv   (same column format as calibrated_labels.csv)
  reports/rubric_calibration/cot_career_results.json

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
OUT_LABELS = BASE / "reports" / "rubric_calibration" / "cot_career_labels.csv"
OUT_RESULTS = BASE / "reports" / "rubric_calibration" / "cot_career_results.json"

DOMAIN = "career_selfimprovement"
JUDGE_MODELS = ["llama3.1:latest", "mistral:latest", "qwen2.5:7b"]
MODEL_COL = {"llama3.1:latest": "llama", "mistral:latest": "mistral", "qwen2.5:7b": "qwen"}
TEMPERATURE = 0.0
NUM_PREDICT = 900  # higher than judge_agent.py's 300 — scratchpad reasoning needs the room


def load_career_transcripts() -> list:
    """[(global_index, transcript), ...] — global_index matches the
    transcript_index scheme used throughout calibrated_labels.csv etc."""
    with open(REVIEW_QUEUE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(i, r["transcript"]) for i, r in enumerate(rows) if r["domain"] == DOMAIN]


def load_rubric() -> dict:
    with open(RUBRICS_PATH, encoding="utf-8") as f:
        return json.load(f)[DOMAIN]


def build_cot_prompt(transcript: str, rubric: dict) -> str:
    high_criteria = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["HIGH"]))
    low_criteria = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["LOW"]))

    return f"""You are an expert educational content evaluator for the {DOMAIN} domain.

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
3. Explicitly check whether motivational language co-occurs with procedural content (the conflict rule above) — this is the specific pattern this domain has historically been inconsistent on.
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


def judge_one(model: str, transcript: str, rubric: dict) -> dict:
    prompt = build_cot_prompt(transcript, rubric)
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


def main():
    transcripts = load_career_transcripts()
    assert len(transcripts) == 20, f"expected 20 career transcripts, got {len(transcripts)}"
    rubric = load_rubric()

    fieldnames = [
        "domain", "transcript_index", "signal_level", "agreement",
        "votes_high", "votes_low", "all_agree",
        "llama_label", "mistral_label", "qwen_label",
        "llama_criterion", "mistral_criterion", "qwen_criterion",
    ]

    done = set()
    if OUT_LABELS.exists():
        with open(OUT_LABELS, encoding="utf-8") as f:
            done = {int(r["transcript_index"]) for r in csv.DictReader(f)}
        print(f"Resuming — {len(done)} already labeled")

    out_file_exists = OUT_LABELS.exists()
    out_file = open(OUT_LABELS, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if not out_file_exists:
        writer.writeheader()

    print(f"Running CoT judges on {len(transcripts)} {DOMAIN} transcripts "
          f"({', '.join(JUDGE_MODELS)}, temp={TEMPERATURE})...\n")

    for i, (idx, transcript) in enumerate(transcripts):
        if idx in done:
            print(f"[{i+1}/{len(transcripts)}] idx={idx:2d} (cached)")
            continue

        per_model = {model: judge_one(model, transcript, rubric) for model in JUDGE_MODELS}

        labels = [per_model[m]["label"] for m in JUDGE_MODELS]
        high_count = labels.count("HIGH")
        low_count = labels.count("LOW")
        majority = "HIGH" if high_count >= 2 else "LOW"
        agreement = max(high_count, low_count) / 3
        all_agree = len(set(labels)) == 1

        row = {
            "domain": DOMAIN,
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
        print(f"[{i+1}/{len(transcripts)}] idx={idx:2d} {majority:4s} ({votes_str}) "
              f"{'✓all-agree' if all_agree else ''}")

    out_file.close()
    print(f"\nSaved per-item CoT labels to {OUT_LABELS}")

    with open(OUT_LABELS, encoding="utf-8") as f:
        cot_rows = list(csv.DictReader(f))

    cot_judge_rows = [[r["llama_label"], r["mistral_label"], r["qwen_label"]] for r in cot_rows]
    cot_inter_judge = fleiss_style_metrics(cot_judge_rows)

    with open(CALIBRATION_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)
    baseline_weighted = summary["domains"][DOMAIN]["final_weighted_agreement"]

    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        non_cot_rows = [r for r in csv.DictReader(f) if r["domain"] == DOMAIN]
    non_cot_judge_rows = [[r["llama_label"], r["mistral_label"], r["qwen_label"]] for r in non_cot_rows]
    non_cot_inter_judge = fleiss_style_metrics(non_cot_judge_rows)

    with open(HUMAN_GOLD, encoding="utf-8") as f:
        human_rows = [r for r in csv.DictReader(f) if r["domain"] == DOMAIN]
    human_by_idx = {int(r["transcript_index"]): r["human_consensus_label"] for r in human_rows}

    cot_by_idx = {int(r["transcript_index"]): r["signal_level"] for r in cot_rows}
    non_cot_by_idx = {int(r["transcript_index"]): r["signal_level"] for r in non_cot_rows}

    cot_vs_human_pairs = [(cot_by_idx[i], human_by_idx[i]) for i in human_by_idx]
    non_cot_vs_human_pairs = [(non_cot_by_idx[i], human_by_idx[i]) for i in human_by_idx]

    results = {
        "domain": DOMAIN,
        "n_transcripts": len(cot_rows),
        "judge_models": JUDGE_MODELS,
        "temperature": TEMPERATURE,
        "rubric_version": rubric["version"],
        "prompting": "chain-of-thought (scratchpad reasoning before label)",
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

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Inter-judge agreement ({DOMAIN}, n=20):")
    print(f"  CoT:                 weighted={cot_inter_judge['weighted']:.3f}  "
          f"all_agree={cot_inter_judge['all_agree']:.3f}  "
          f"kappa={cot_inter_judge['fleiss_kappa']:+.3f}  AC1={cot_inter_judge['gwet_ac1']:+.3f}")
    print(f"  No CoT (recomputed): weighted={non_cot_inter_judge['weighted']:.3f}  "
          f"all_agree={non_cot_inter_judge['all_agree']:.3f}  "
          f"kappa={non_cot_inter_judge['fleiss_kappa']:+.3f}  AC1={non_cot_inter_judge['gwet_ac1']:+.3f}")
    print(f"  No CoT (calibration_summary.json baseline): weighted={baseline_weighted:.3f}")

    print(f"\nvs. human_consensus_label (n={len(human_by_idx)}):")
    ch = results["vs_human_ground_truth"]["cot_majority_vote"]
    nh = results["vs_human_ground_truth"]["non_cot_majority_vote"]
    print(f"  CoT:    agreement={ch['agreement']:.3f}  kappa={ch['cohen_kappa']:+.3f}  AC1={ch['gwet_ac1']:+.3f}")
    print(f"  No CoT: agreement={nh['agreement']:.3f}  kappa={nh['cohen_kappa']:+.3f}  AC1={nh['gwet_ac1']:+.3f}")

    print(f"\nSaved summary to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
