#!/usr/bin/env python3
"""
A2 -- Judge-replacement experiment on Career & Self-Improvement.

Replacement model: granite3.1-dense:8b (IBM, 8.2B params, 131072 context).
Selected as the only candidate attempted, before running, for these reasons:
  - Comparable parameter count to Qwen 2.5 7B (7.6B), unlike phi:latest
    (Phi-2, 2.8B, 2048-token context -- would confound replacement with a
    much weaker model and likely prompt truncation).
  - Genuinely distinct training lineage from Meta (Llama), Mistral AI, and
    Alibaba (Qwen) -- matches the paper's own stated judge-selection
    principle (Section 3.3: "chosen to represent different training
    lineages, maximizing the diagnostic value of their disagreements").
  - Explicitly NOT Gemma 2 9B, which this paper already disqualified as a
    degenerate judge (Section 3.5) -- reusing it would confound "judge
    replacement" with "known-degenerate judge."
No other replacement candidate was pulled or evaluated.

Llama 3.1 8B and Mistral 7B outputs are REUSED from calibrated_labels.csv
(deterministic at temperature 0.0, same v1 rubric, same transcripts -- this
is the actual original-run output for those two judges, not a re-derivation).
Only the replacement judge is run fresh.

Scope: Career & Self-Improvement only, 20 transcripts (original 60-corpus
subset), retained v1 rubric. The 10 of those 20 with human reference labels
(Section 5.5's Prolific study) are used for the human-alignment comparison.

Output: results/qwen_replacement_ablation.json
"""
import csv
import json
from pathlib import Path

from judge_agent import JUDGE_TEMPERATURE, judge_transcript
from experiment_utils import (
    JudgeCallCache, fleiss_style_metrics, two_rater_agreement_metrics, record_provenance,
)

BASE = Path(__file__).resolve().parents[2]
REVIEW_QUEUE = BASE / "data" / "review_queue.csv"
FINAL_RUBRICS_PATH = BASE / "reports" / "rubric_calibration" / "calibrated_rubrics.json"
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
HUMAN_GOLD = BASE / "gold-sampled-dataset" / "human_validation_ground_truth.csv"
RESULTS_DIR = BASE / "results"
CACHE_PATH = RESULTS_DIR / "judge_call_cache.csv"
OUT_PATH = RESULTS_DIR / "qwen_replacement_ablation.json"

DOMAIN = "career_selfimprovement"
REPLACEMENT_MODEL = "granite3.1-dense:8b"
ORIGINAL_MODEL = "qwen2.5:7b"
RUN_TAG = "a2_replacement"


def load_domain_transcripts() -> list:
    with open(REVIEW_QUEUE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(i, r["transcript"]) for i, r in enumerate(rows) if r["domain"] == DOMAIN]


def main():
    transcripts = load_domain_transcripts()
    assert len(transcripts) == 20

    with open(FINAL_RUBRICS_PATH, encoding="utf-8") as f:
        rubric = json.load(f)[DOMAIN]

    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        orig_by_idx = {int(r["transcript_index"]): r for r in csv.DictReader(f) if r["domain"] == DOMAIN}

    cache = JudgeCallCache(CACHE_PATH)

    print(f"Running {REPLACEMENT_MODEL} on {len(transcripts)} {DOMAIN} transcripts (v1 rubric)...")
    replacement_labels = {}
    for idx, transcript in transcripts:
        r = cache.get_or_call(judge_transcript, RUN_TAG, DOMAIN, "v1_final",
                               idx, transcript, rubric, REPLACEMENT_MODEL)
        replacement_labels[idx] = r["label"]
        print(f"  idx={idx:2d} -> {r['label']}")
    cache.close()

    original_rows, replacement_rows = [], []
    orig_llama_hr = orig_mistral_hr = orig_qwen_hr = 0
    repl_llama_hr = repl_mistral_hr = repl_replacement_hr = 0

    for idx, _ in transcripts:
        o = orig_by_idx[idx]
        original_rows.append([o["llama_label"], o["mistral_label"], o["qwen_label"]])
        replacement_rows.append([o["llama_label"], o["mistral_label"], replacement_labels[idx]])
        orig_llama_hr += o["llama_label"] == "HIGH"
        orig_mistral_hr += o["mistral_label"] == "HIGH"
        orig_qwen_hr += o["qwen_label"] == "HIGH"
        repl_replacement_hr += replacement_labels[idx] == "HIGH"

    n = len(transcripts)
    original_metrics = fleiss_style_metrics(original_rows)
    replacement_metrics = fleiss_style_metrics(replacement_rows)

    def majority(row):
        return "HIGH" if row.count("HIGH") >= 2 else "LOW"

    orig_majority = {idx: majority(original_rows[i]) for i, (idx, _) in enumerate(transcripts)}
    repl_majority = {idx: majority(replacement_rows[i]) for i, (idx, _) in enumerate(transcripts)}

    with open(HUMAN_GOLD, encoding="utf-8") as f:
        human_rows = [r for r in csv.DictReader(f) if r["domain"] == DOMAIN]
    human_by_idx = {int(r["transcript_index"]): r["human_consensus_label"] for r in human_rows}

    orig_vs_human_pairs = [(orig_majority[i], human_by_idx[i]) for i in human_by_idx]
    repl_vs_human_pairs = [(repl_majority[i], human_by_idx[i]) for i in human_by_idx]

    prompt_template = open(BASE / "src" / "agents" / "judge_agent.py").read()
    provenance = record_provenance(
        models=[REPLACEMENT_MODEL, "llama3.1:latest", "mistral:latest"],
        dataset_paths=[REVIEW_QUEUE, HUMAN_GOLD],
        rubric_paths=[FINAL_RUBRICS_PATH],
        prompt_template=prompt_template,
        repo_dir=BASE,
    )

    results = {
        "experiment": "A2 -- judge-replacement on Career & Self-Improvement",
        "replacement_model_selection": {
            "model": REPLACEMENT_MODEL,
            "reason": "Comparable scale to Qwen 2.5 7B (8.2B vs 7.6B params), 131072-token context "
                      "(no truncation risk), distinct training lineage (IBM vs. Meta/Mistral AI/Alibaba), "
                      "explicitly not Gemma 2 9B (already disqualified as degenerate in this paper). "
                      "Only candidate attempted -- no other model was pulled or evaluated for this "
                      "experiment, so this is a single principled test, not a best-of-N selection.",
            "download_authorization": "One-time user override of the 'no new large model downloads' "
                                       "constraint, for this experiment only.",
        },
        "scope_note": (
            "Tests whether Career & Self-Improvement's stagnation was panel-specific (i.e. "
            "Qwen-2.5-7B-specific) and recoverable via judge replacement, evaluated against BOTH "
            "inter-judge agreement and human alignment -- not a general capability-ceiling claim "
            "regardless of outcome."
        ),
        "n_transcripts": n,
        "a_inter_judge_agreement": {
            "original_panel_llama_mistral_qwen": original_metrics,
            "replacement_panel_llama_mistral_granite": replacement_metrics,
        },
        "b_vs_human_n10": {
            "original_panel_majority_vs_human": two_rater_agreement_metrics(orig_vs_human_pairs),
            "replacement_panel_majority_vs_human": two_rater_agreement_metrics(repl_vs_human_pairs),
            "note": "Original panel's existing reported figure elsewhere in the paper is 50% "
                    "(Table 3/4); recomputed here identically for direct side-by-side comparison.",
        },
        "c_label_prevalence_high_rate": {
            "original_panel": {
                "llama3.1:latest": round(orig_llama_hr / n, 4),
                "mistral:latest": round(orig_mistral_hr / n, 4),
                "qwen2.5:7b": round(orig_qwen_hr / n, 4),
            },
            "replacement_panel": {
                "llama3.1:latest": round(orig_llama_hr / n, 4),
                "mistral:latest": round(orig_mistral_hr / n, 4),
                REPLACEMENT_MODEL: round(repl_replacement_hr / n, 4),
            },
        },
        "provenance": provenance,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nInter-judge: original={original_metrics['weighted']:.3f}  replacement={replacement_metrics['weighted']:.3f}")
    print(f"vs human (n=10): original={results['b_vs_human_n10']['original_panel_majority_vs_human']}  "
          f"replacement={results['b_vs_human_n10']['replacement_panel_majority_vs_human']}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
