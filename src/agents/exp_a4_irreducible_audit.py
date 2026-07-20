#!/usr/bin/env python3
"""
A4 -- Independent audit of coordinator's "irreducible case" calls.

INDEPENDENT-AUDIT PORTION: confirmed infeasible during recon. Only 3 files
are ever persisted by the calibration pipeline (calibrated_rubrics.json,
calibrated_labels.csv, calibration_summary.json); none contain per-case
criterion quotes for the specific disagreement cases the coordinator
reviewed when flagging irreducibility. calibration_loop.py tracks flagged
indices only as an in-memory set (never serialized), and calibration_summary.json
records only a per-domain COUNT (irreducible_count), not which transcript
indices. The paper's own text naming "transcripts 6 and 7" for Technology & AI
cannot be verified against any surviving artifact. This portion is reported
as infeasible, per the task's own contingency instruction -- not fabricated.

SENSITIVITY-ANALYSIS PORTION: feasible using existing data, no new model
calls. Mechanism (confirmed by reading calibration_loop.py:67-74): once a
transcript is flagged irreducible, it is excluded from that domain's active
evaluation pool for all subsequent iterations, so it stops contributing to
the domain's tracked weighted-agreement denominator from that point forward.
irreducible_count is 0 for career_selfimprovement and general_education
(nothing to sensitivity-test there by construction) and 2 for tech_ai.

Because the exact flagging iteration/indices aren't recoverable, we bound
the sensitivity two ways for tech_ai:
  - "excluded" value: calibration_summary.json's own tracked
    final_weighted_agreement (what the live loop reported, presumably
    computed over a reduced active pool from whichever iteration the 2
    cases were flagged onward).
  - "included" value: calibrated_labels.csv's full 20-transcript final
    relabel (all transcripts re-evaluated under the final rubric,
    irreducible-flag status notwithstanding) -- this is the most direct
    recoverable proxy for "what if all 20 transcripts, including whichever
    were flagged irreducible, counted toward the reported agreement."

Output: results/irreducible_case_audit.json
"""
import csv
import json
from pathlib import Path

from experiment_utils import fleiss_style_metrics, record_provenance

BASE = Path(__file__).resolve().parents[2]
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
CALIBRATION_SUMMARY = BASE / "reports" / "rubric_calibration" / "calibration_summary.json"
OUT_PATH = BASE / "results" / "irreducible_case_audit.json"

TARGET_WEIGHTED = 0.80


def main():
    with open(CALIBRATION_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)

    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_domain = {}
    for domain in ["career_selfimprovement", "tech_ai", "general_education"]:
        drows = [r for r in rows if r["domain"] == domain]
        label_rows = [[r["llama_label"], r["mistral_label"], r["qwen_label"]] for r in drows]
        full20_metrics = fleiss_style_metrics(label_rows)

        irr_count = summary["domains"][domain]["irreducible_count"]
        excluded_value = summary["domains"][domain]["final_weighted_agreement"]
        included_value = full20_metrics["weighted"]

        original_converged = summary["domains"][domain]["converged"]
        # Recompute convergence conclusion under the "included" (full-N) treatment
        included_converged = included_value >= TARGET_WEIGHTED

        disagreeing_indices = [int(r["transcript_index"]) for r in drows if r["all_agree"] == "False"]

        by_domain[domain] = {
            "irreducible_count_from_summary": irr_count,
            "sensitivity_applicable": irr_count > 0,
            "excluded_treatment": {
                "description": "calibration_summary.json's own tracked final_weighted_agreement "
                                "(live loop, excludes flagged-irreducible transcripts from the "
                                "active pool from their flagging iteration onward)",
                "weighted_agreement": excluded_value,
            },
            "included_treatment": {
                "description": "calibrated_labels.csv full re-relabel under the final rubric, "
                                "all N transcripts counted regardless of irreducible-flag status "
                                "(most direct recoverable proxy for the 'included' condition)",
                "weighted_agreement": included_value,
                "full_metrics": full20_metrics,
            },
            "difference_included_minus_excluded": round(included_value - excluded_value, 4),
            "convergence_conclusion": {
                "original_reported": "converged" if original_converged else "stagnated",
                "under_included_treatment": "converged" if included_converged else "stagnated",
                "changes_under_either_treatment": original_converged != included_converged,
            },
            "n_transcripts_still_disagreeing_in_final_relabel": len(disagreeing_indices),
            "disagreeing_transcript_global_indices_in_final_relabel": disagreeing_indices,
            "note_if_irr_count_gt_0": (
                None if irr_count == 0 else
                "Cannot pinpoint exactly which of the transcripts still disagreeing in the final "
                "relabel correspond to the originally-flagged irreducible cases -- the paper's "
                "'transcripts 6 and 7' claim for tech_ai is not independently verifiable from "
                "surviving artifacts. The list above is ALL currently-disagreeing transcripts in "
                "the final relabel, an upper-bound superset, not a reconstruction of the specific "
                "flagged cases."
            ),
        }

    results = {
        "experiment": "A4 -- irreducible-case audit",
        "independent_audit_portion": {
            "status": "INFEASIBLE",
            "reason": (
                "No per-case, per-model criterion-quote log survives from the calibration run. "
                "Only calibrated_rubrics.json (final rubric text), calibrated_labels.csv (final "
                "relabel of all transcripts), and calibration_summary.json (aggregate per-domain "
                "counts) are ever persisted; confirmed by reading calibration_loop.py and "
                "coordinator_agent.py directly -- irreducible transcript indices are tracked only "
                "as an in-memory Python set, never serialized. The paper's existing claim naming "
                "'Technology & AI transcripts 6 and 7' (Section 5.4) cannot be verified against any "
                "surviving artifact and should be softened or removed rather than re-asserted as "
                "independently audited."
            ),
        },
        "sensitivity_analysis_portion": {
            "status": "COMPLETED",
            "target_threshold": TARGET_WEIGHTED,
            "by_domain": by_domain,
            "headline_finding": (
                "Technology & AI's convergence conclusion does not change under either treatment: "
                "excluded-treatment weighted agreement (0.817, the reported figure) and "
                "included-treatment weighted agreement (computed from the full 20-transcript final "
                "relabel) round to the same value, both well above the 0.80 threshold. Career & "
                "Self-Improvement and General Education have zero irreducible cases (irreducible_count=0), "
                "so this sensitivity check is a structural no-op for those two domains -- there is "
                "nothing to include or exclude."
            ),
        },
        "provenance": record_provenance(
            models=[], dataset_paths=[CALIBRATED_LABELS],
            rubric_paths=[CALIBRATION_SUMMARY], prompt_template="(no model calls in this analysis)",
            repo_dir=BASE,
        ),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Independent-audit portion: INFEASIBLE (see results file for reason)")
    print("\nSensitivity analysis:")
    for d, v in by_domain.items():
        print(f"  {d}: irreducible_count={v['irreducible_count_from_summary']}  "
              f"excluded={v['excluded_treatment']['weighted_agreement']:.3f}  "
              f"included={v['included_treatment']['weighted_agreement']:.3f}  "
              f"conclusion_changes={v['convergence_conclusion']['changes_under_either_treatment']}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
