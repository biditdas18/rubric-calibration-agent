#!/usr/bin/env python3
"""
A1 — Unseen-to-coordinator rubric evaluation: v0 vs. final calibrated rubric,
paired comparison on the 90-transcript SNR training-supplement corpus
(30/domain), which was never shown to the coordinator during rubric
revision.

Single inference pass per rubric version per domain (same as one iteration's
judge-evaluation step) -- no further coordinator revision.

Scope (deliberately narrow, per task spec): this tests whether the rubric
revision generalizes in its ability to CHANGE inter-judge agreement on
unseen transcripts. It does NOT establish the final rubric is more accurate
or more human-aligned on unseen transcripts -- no human labels exist for
this corpus.

Input:
  ../../../snr-detector/data/transcripts_new/*.jsonl   (90 transcripts, 30/domain)
  src/agents/rubrics.json                              (v0 rubrics)
  reports/rubric_calibration/calibrated_rubrics.json   (final/retained rubrics)

Output:
  results/held_out_rubric_validation.json
  results/judge_call_cache.csv (shared resumable cache)
"""
import json
from pathlib import Path

from judge_agent import (
    JUDGE_MODELS, JUDGE_TEMPERATURE, judge_transcript, merge_model_results,
)
from experiment_utils import (
    JudgeCallCache, fleiss_style_metrics, paired_bootstrap_ci, record_provenance,
)

BASE = Path(__file__).resolve().parents[2]
SNR_BASE = BASE.parent / "snr-detector"
TRANSCRIPTS_NEW_DIR = SNR_BASE / "data" / "transcripts_new"
V0_RUBRICS_PATH = BASE / "src" / "agents" / "rubrics.json"
FINAL_RUBRICS_PATH = BASE / "reports" / "rubric_calibration" / "calibrated_rubrics.json"
RESULTS_DIR = BASE / "results"
CACHE_PATH = RESULTS_DIR / "judge_call_cache.csv"
OUT_PATH = RESULTS_DIR / "held_out_rubric_validation.json"

DOMAINS = ["career_selfimprovement", "tech_ai", "general_education"]
RUN_TAG = "a1_heldout90"


def load_supplement_corpus() -> dict:
    """domain -> [{"index": i, "transcript": ...}, ...], 30 per domain."""
    out = {}
    for domain in DOMAINS:
        path = TRANSCRIPTS_NEW_DIR / f"new_transcripts_{domain}.jsonl"
        rows = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                rows.append({"index": i, "transcript": r["transcript"]})
        assert len(rows) == 30, f"expected 30 for {domain}, got {len(rows)}"
        out[domain] = rows
    return out


def run_panel(cache: JudgeCallCache, rubric_version: str, domain: str,
              transcripts: list, rubric: dict) -> list:
    """Runs all 3 judges (cached) and returns merged per-transcript results."""
    results_by_model = {}
    for model in JUDGE_MODELS:
        per_transcript = []
        for t in transcripts:
            r = cache.get_or_call(
                judge_transcript, RUN_TAG, domain, rubric_version,
                t["index"], t["transcript"], rubric, model,
            )
            r["transcript_index"] = t["index"]
            per_transcript.append(r)
        results_by_model[model] = per_transcript
    return merge_model_results(transcripts, results_by_model)


def per_transcript_weight(merged_row: dict) -> float:
    high, low = merged_row["votes"]["HIGH"], merged_row["votes"]["LOW"]
    return 1.0 if (high == 3 or low == 3) else 0.667


def prevalence(merged: list) -> dict:
    """HIGH-rate per judge model over the merged result set."""
    out = {}
    for i, model in enumerate(JUDGE_MODELS):
        labels = [r["individual_results"][i]["label"] for r in merged]
        out[model] = round(sum(l == "HIGH" for l in labels) / len(labels), 4)
    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    corpus = load_supplement_corpus()
    with open(V0_RUBRICS_PATH, encoding="utf-8") as f:
        v0_rubrics = json.load(f)
    with open(FINAL_RUBRICS_PATH, encoding="utf-8") as f:
        final_rubrics = json.load(f)

    cache = JudgeCallCache(CACHE_PATH)

    domain_results = {}
    total_calls = len(DOMAINS) * 2 * 30 * 3
    done_calls = 0

    for domain in DOMAINS:
        transcripts = corpus[domain]
        print(f"\n[{domain}] v0 rubric (version {v0_rubrics[domain]['version']})...")
        merged_v0 = run_panel(cache, "v0", domain, transcripts, v0_rubrics[domain])
        done_calls += 30 * 3
        print(f"  done ({done_calls}/{total_calls} calls so far)")

        print(f"[{domain}] final rubric (version {final_rubrics[domain]['version']})...")
        merged_final = run_panel(cache, "final", domain, transcripts, final_rubrics[domain])
        done_calls += 30 * 3
        print(f"  done ({done_calls}/{total_calls} calls so far)")

        v0_rows = [[r["individual_results"][i]["label"] for i in range(3)] for r in merged_v0]
        final_rows = [[r["individual_results"][i]["label"] for i in range(3)] for r in merged_final]

        v0_metrics = fleiss_style_metrics(v0_rows)
        final_metrics = fleiss_style_metrics(final_rows)

        by_idx_v0 = {r["transcript_index"]: per_transcript_weight(r) for r in merged_v0}
        by_idx_final = {r["transcript_index"]: per_transcript_weight(r) for r in merged_final}
        diffs = [by_idx_final[i] - by_idx_v0[i] for i in sorted(by_idx_v0)]
        diff_ci = paired_bootstrap_ci(diffs)

        domain_results[domain] = {
            "v0_rubric_version": v0_rubrics[domain]["version"],
            "final_rubric_version": final_rubrics[domain]["version"],
            "v0": v0_metrics,
            "final": final_metrics,
            "final_minus_v0_weighted_point": round(final_metrics["weighted"] - v0_metrics["weighted"], 4),
            "paired_bootstrap_ci_diff": diff_ci,
            "prevalence_high_rate": {"v0": prevalence(merged_v0), "final": prevalence(merged_final)},
        }
        print(f"  {domain}: v0 weighted={v0_metrics['weighted']:.3f}  "
              f"final weighted={final_metrics['weighted']:.3f}  "
              f"diff={domain_results[domain]['final_minus_v0_weighted_point']:+.4f} "
              f"CI=[{diff_ci['ci_low']:+.4f}, {diff_ci['ci_high']:+.4f}]")

    cache.close()

    prompt_template = open(BASE / "src" / "agents" / "judge_agent.py").read()
    provenance = record_provenance(
        models=JUDGE_MODELS,
        dataset_paths=[TRANSCRIPTS_NEW_DIR / f"new_transcripts_{d}.jsonl" for d in DOMAINS],
        rubric_paths=[V0_RUBRICS_PATH, FINAL_RUBRICS_PATH],
        prompt_template=prompt_template,
        repo_dir=BASE,
    )

    results = {
        "experiment": "A1 -- unseen-to-coordinator rubric evaluation (v0 vs. final)",
        "scope_note": (
            "Tests whether the calibration revision generalizes in its ability to "
            "change inter-judge agreement on transcripts unseen during coordinator "
            "revision. Does NOT establish the final rubric is more accurate or more "
            "human-aligned on this set -- no human labels exist for it."
        ),
        "corpus": "snr-detector/data/transcripts_new/*.jsonl (90 transcripts, 30/domain, "
                  "document-disjoint from the 60-transcript calibration corpus but NOT "
                  "channel-disjoint -- 6 channels overlap)",
        "judge_temperature": JUDGE_TEMPERATURE,
        "domains": domain_results,
        "provenance": provenance,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
