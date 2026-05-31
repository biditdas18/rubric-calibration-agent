import json
import time
import csv
import os
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_agent import (
    run_all_transcripts_one_model,
    merge_model_results,
    JUDGE_MODELS,
    JUDGE_TEMPERATURE
)
from coordinator_agent import run_coordinator

# ── CONFIG ────────────────────────────────────────────────
MAX_TIME_SECONDS = 3 * 60 * 60  # 3 hours
MAX_ITERATIONS   = 10
TARGET_AGREEMENT = 0.80  # Weighted agreement target
                          # ~0.80 weighted ≈ ~70% all-agree
                          # Realistic for 3 different architectures
                          # on subjective quality classification
MIN_DELTA        = 0.02

BASE_DIR    = "/Users/biditdas/Desktop/snr-submission/rubric-calibration-agent"
REPORT_DIR  = os.path.join(BASE_DIR, "reports/rubric_calibration")
DATA_PATH   = os.path.join(BASE_DIR, "data/review_queue.csv")
RUBRIC_PATH = os.path.join(BASE_DIR, "src/agents/rubrics.json")
# ─────────────────────────────────────────────────────────


def load_transcripts(path: str) -> dict:
    domain_transcripts = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            domain_transcripts[row["domain"]].append({
                "index": i,
                "domain": row["domain"],
                "transcript": row["transcript"]
            })
    return domain_transcripts


def evaluate_domain(
    transcripts: list,
    domain: str,
    rubric: dict,
    irreducible_indices: set
) -> dict:
    """
    Evaluate all transcripts in a domain with 3 judges.

    Agreement metric: WEIGHTED AGREEMENT
    - All 3 agree:     1.000 per transcript
    - 2 of 3 agree:    0.667 per transcript
    - 0 of 3 agree:    0.000 per transcript (impossible with binary)
    Average across transcripts = weighted agreement rate.

    This is mathematically correct — majority agree (2/3) is
    always 100% with binary labels and 3 judges, which is
    meaningless. Weighted agreement captures genuine signal.
    """
    results      = []
    disagreements = []
    weighted_sum  = 0.0
    all_agree_count = 0

    for t in transcripts:
        if t["index"] in irreducible_indices:
            continue

        result = run_three_judges(
            transcript=t["transcript"],
            domain=domain,
            rubric=rubric
        )
        result["transcript_index"] = t["index"]
        result["transcript"]       = t["transcript"]
        results.append(result)

        high = result["votes"]["HIGH"]
        low  = result["votes"]["LOW"]

        if high == 3 or low == 3:
            weighted_sum    += 1.0
            all_agree_count += 1
        elif high == 2 or low == 2:
            weighted_sum += 0.667
            disagreements.append({**t, **result})
        else:
            weighted_sum += 0.0
            disagreements.append({**t, **result})

    total          = len(results)
    weighted_rate  = weighted_sum / total if total > 0 else 0
    all_agree_rate = all_agree_count / total if total > 0 else 0

    return {
        "agreement_rate":       weighted_rate,    # PRIMARY — weighted
        "all_agree_rate":       all_agree_rate,   # SECONDARY — strict
        "total_evaluated":      total,
        "all_agreements":       all_agree_count,
        "partial_agreements":   len([r for r in results
                                     if (r["votes"]["HIGH"] == 2 or
                                         r["votes"]["LOW"] == 2)]),
        "disagreements":        len(disagreements),
        "disagreement_details": disagreements,
        "all_results":          results
    }


def run_iteration(
    transcripts_by_domain: dict,
    domains: list,
    domain_state: dict
) -> dict:
    """
    One full iteration:
      Model 1 → all domains → Model 2 → all domains → Model 3 → all domains
    3 model loads total regardless of domain count.
    Returns per-domain eval results.
    """
    active_by_domain = {
        d: [t for t in transcripts_by_domain[d]
            if t["index"] not in domain_state[d]["irreducible"]]
        for d in domains
        if not domain_state[d]["converged"]
        and domain_state[d]["iteration"] < MAX_ITERATIONS
    }

    if not active_by_domain:
        return {}

    results_store = {m: {} for m in JUDGE_MODELS}

    for model in JUDGE_MODELS:
        short       = model.split(":")[0]
        total_calls = sum(len(v) for v in active_by_domain.values())
        print(f"\n  [{short}] evaluating all domains "
              f"({total_calls} transcripts)...", flush=True)
        for domain, transcripts in active_by_domain.items():
            results_store[model][domain] = run_all_transcripts_one_model(
                transcripts=transcripts,
                domain=domain,
                rubric=domain_state[domain]["rubric"],
                model=model
            )

    eval_results = {}
    for domain, transcripts in active_by_domain.items():
        results_by_model = {m: results_store[m][domain] for m in JUDGE_MODELS}
        merged = merge_model_results(transcripts, results_by_model)

        weighted_sum    = 0.0
        all_agree_count = 0
        disagreements   = []

        for r in merged:
            high = r["votes"]["HIGH"]
            low  = r["votes"]["LOW"]
            if high == 3 or low == 3:
                weighted_sum    += 1.0
                all_agree_count += 1
            elif high == 2 or low == 2:
                weighted_sum += 0.667
                disagreements.append(r)
            else:
                weighted_sum += 0.0
                disagreements.append(r)

        total = len(merged)
        eval_results[domain] = {
            "agreement_rate":       weighted_sum / total if total > 0 else 0,
            "all_agree_rate":       all_agree_count / total if total > 0 else 0,
            "total_evaluated":      total,
            "all_agreements":       all_agree_count,
            "partial_agreements":   len([r for r in merged
                                         if (r["votes"]["HIGH"] == 2 or
                                             r["votes"]["LOW"] == 2)]),
            "disagreements":        len(disagreements),
            "disagreement_details": disagreements,
            "all_results":          merged
        }

    return eval_results


# Import run_three_judges for evaluate_domain (used in regression check)
from judge_agent import run_all_transcripts_one_model, merge_model_results


def run_three_judges(transcript, domain, rubric):
    """Wrapper used by evaluate_domain for regression checks."""
    results_by_model = {}
    for model in JUDGE_MODELS:
        res = run_all_transcripts_one_model(
            transcripts=[{"index": 0, "transcript": transcript,
                          "domain": domain}],
            domain=domain,
            rubric=rubric,
            model=model
        )
        results_by_model[model] = res

    merged = merge_model_results(
        [{"index": 0, "transcript": transcript}],
        results_by_model
    )
    return merged[0]


def run_calibration():
    os.makedirs(REPORT_DIR, exist_ok=True)
    start_time = time.time()

    with open(RUBRIC_PATH, "r") as f:
        rubrics = json.load(f)

    transcripts_by_domain = load_transcripts(DATA_PATH)
    domains = list(rubrics.keys())

    domain_state = {
        d: {
            "agreement_history":   [],
            "all_agree_history":   [],
            "rubric":              rubrics[d],
            "iteration":           0,
            "converged":           False,
            "irreducible":         set(),
            "final_results":       None,
            "rubric_change_log":   [],
            "regression_detected": False
        }
        for d in domains
    }

    total_transcripts = sum(len(v) for v in transcripts_by_domain.values())

    print(f"\n{'='*60}")
    print("SNR RUBRIC CALIBRATION — 3 MODELS, BATCHED BY MODEL+DOMAIN")
    print(f"{'='*60}")
    print(f"Evaluation order per iteration:")
    print(f"  Model 1 (all domains) → Model 2 (all domains) → Model 3 (all domains)")
    print(f"  = 3 model loads per iteration total")
    print(f"Judge models (temperature={JUDGE_TEMPERATURE}):")
    for m in JUDGE_MODELS:
        print(f"  • {m}")
    print(f"Coordinator: claude-sonnet-4-6")
    print(f"Note: Gemma 2 replaced with Qwen 2.5 — Gemma was systematically")
    print(f"      conservative (LOW on 60/60 transcripts, 0 discriminative value)")
    print(f"Target: {TARGET_AGREEMENT:.0%} weighted | "
          f"Max iter: {MAX_ITERATIONS} | Max time: 3 hours")
    print(f"Transcripts: {total_transcripts} across {len(domains)} domains")
    print(f"{'='*60}\n")

    iteration = 0

    while True:
        iteration += 1
        elapsed     = time.time() - start_time
        elapsed_min = elapsed / 60

        print(f"\n{'─'*60}")
        print(f"ITERATION {iteration} | Time: {elapsed_min:.1f} min")
        print(f"{'─'*60}")

        if elapsed > MAX_TIME_SECONDS:
            print("STOPPING: Time limit (3 hours)")
            break
        if all(s["converged"] for s in domain_state.values()):
            print("STOPPING: All domains converged")
            break
        if iteration > MAX_ITERATIONS:
            print("STOPPING: Max iterations")
            break

        eval_results = run_iteration(
            transcripts_by_domain, domains, domain_state
        )

        if not eval_results:
            print("\nAll domains done.")
            break

        for domain in domains:
            if domain not in eval_results:
                print(f"\n[{domain}] Converged ✓ — skipping")
                continue

            state       = domain_state[domain]
            eval_result = eval_results[domain]
            agreement   = eval_result["agreement_rate"]

            state["agreement_history"].append(agreement)
            state["all_agree_history"].append(
                eval_result.get("all_agree_rate", 0)
            )
            state["iteration"] += 1

            prev  = (state["agreement_history"][-2]
                     if len(state["agreement_history"]) > 1 else 0)
            delta = agreement - prev

            all_rate = eval_result.get("all_agree_rate", 0)
            n_all    = eval_result.get("all_agreements", 0)
            total_n  = eval_result.get("total_evaluated", 0)

            print(f"\n[{domain}]")
            print(f"  Weighted agree: {agreement:.1%} (Δ {delta:+.1%})")
            print(f"  All agree:      {all_rate:.1%} "
                  f"({n_all}/{total_n} transcripts)")
            print(f"  Partial agree:  "
                  f"{eval_result.get('partial_agreements', 0)} transcripts")
            print(f"  Need fixing:    {eval_result['disagreements']}")

            if eval_result["disagreement_details"]:
                sample = eval_result["disagreement_details"][0]
                votes  = {
                    r["model"].split(":")[0]: r["label"]
                    for r in sample["individual_results"]
                }
                print(f"  Sample disagreement: {votes}")

            if agreement >= TARGET_AGREEMENT:
                state["converged"]     = True
                state["final_results"] = eval_result
                print(f"  ✓ CONVERGED at {agreement:.1%}!")
                continue

            hist = state["agreement_history"]
            if (len(hist) >= 3 and
                    all(abs(hist[-i-1] - hist[-i-2]) < MIN_DELTA
                        for i in range(2))):
                print(f"  ⚠ Stagnated — model capability ceiling")
                state["final_results"] = eval_result
                continue

            if eval_result["disagreements"] > 0:
                print(f"  Running coordinator (Claude Sonnet)...")
                coord = run_coordinator(
                    domain=domain,
                    rubric=state["rubric"],
                    disagreements=eval_result["disagreement_details"],
                    iteration=state["iteration"],
                    agreement_history=state["agreement_history"],
                    judge_models=JUDGE_MODELS,
                    change_log=state["rubric_change_log"]
                )
                if coord["success"]:
                    state["rubric"] = coord["updated_rubric"]
                    state["rubric_change_log"].append({
                        "iteration":        state["iteration"],
                        "agreement_before": agreement,
                        "reasoning":        coord["reasoning"],
                        "diagnosis":        coord["diagnosis"]
                    })
                    new_irr = set(coord.get("irreducible_transcripts", []))
                    state["irreducible"].update(new_irr)
                    print(f"  Rubric → v{state['rubric']['version']}")
                    if coord["reasoning"]:
                        print(f"  Coordinator: {coord['reasoning'][:120]}...")
                    if new_irr:
                        print(f"  Irreducible flagged: {new_irr}")
                else:
                    print(f"  Coordinator failed — keeping rubric")

            state["final_results"] = eval_result

        # ── REGRESSION CHECK ──────────────────────────────
        for domain in domains:
            state = domain_state[domain]
            if not state["converged"]:
                continue
            other_rubric_changed = any(
                len(domain_state[d]["rubric_change_log"]) > state["iteration"]
                for d in domains if d != domain
            )
            if not other_rubric_changed:
                continue

            print(f"\n[{domain}] Regression check (was converged)...")
            reg_result = evaluate_domain(
                transcripts_by_domain[domain],
                domain,
                state["rubric"],
                state["irreducible"]
            )
            reg_rate  = reg_result["agreement_rate"]
            prev_rate = (state["agreement_history"][-1]
                         if state["agreement_history"] else 0)

            if reg_rate < prev_rate - 0.10:
                print(f"  ⚠ REGRESSION: {prev_rate:.1%} → {reg_rate:.1%}")
                print(f"  Domain un-converged — coordinator will revisit")
                state["converged"] = False
                state["agreement_history"].append(reg_rate)
                state["regression_detected"] = True
            else:
                print(f"  ✓ Stable: {reg_rate:.1%} (no regression)")

    # ── SAVE OUTPUTS ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")

    final_rubrics = {d: s["rubric"] for d, s in domain_state.items()}
    rubric_out = os.path.join(REPORT_DIR, "calibrated_rubrics.json")
    with open(rubric_out, "w") as f:
        json.dump(final_rubrics, f, indent=2)
    print(f"✓ {rubric_out}")

    label_rows = []
    for domain, state in domain_state.items():
        if state["final_results"]:
            for r in state["final_results"]["all_results"]:
                label_rows.append({
                    "domain":            domain,
                    "transcript_index":  r["transcript_index"],
                    "signal_level":      r["majority_label"],
                    "agreement":         round(r["agreement"], 3),
                    "votes_high":        r["votes"]["HIGH"],
                    "votes_low":         r["votes"]["LOW"],
                    "all_agree":         r["all_agree"],
                    "llama_label":       r["individual_results"][0]["label"],
                    "mistral_label":     r["individual_results"][1]["label"],
                    "qwen_label":        r["individual_results"][2]["label"],
                    "llama_criterion":   r["individual_results"][0].get(
                                             "criterion_quote", ""),
                    "mistral_criterion": r["individual_results"][1].get(
                                             "criterion_quote", ""),
                    "qwen_criterion":    r["individual_results"][2].get(
                                             "criterion_quote", "")
                })

    labels_out = os.path.join(REPORT_DIR, "calibrated_labels.csv")
    if label_rows:
        with open(labels_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=label_rows[0].keys())
            writer.writeheader()
            writer.writerows(label_rows)
    print(f"✓ {labels_out}")

    summary = {
        "run_timestamp":      datetime.now().isoformat(),
        "total_time_minutes": round((time.time() - start_time) / 60, 2),
        "total_iterations":   iteration,
        "target_agreement":   TARGET_AGREEMENT,
        "agreement_metric":   "weighted (1.0=all agree, 0.667=2/3 agree)",
        "judge_models":       JUDGE_MODELS,
        "judge_temperature":  JUDGE_TEMPERATURE,
        "coordinator_model":  "claude-sonnet-4-6",
        "domains": {}
    }

    print(f"\n{'─'*60}")
    print("FINAL RESULTS")
    print(f"{'─'*60}")

    for domain, state in domain_state.items():
        hist     = state["agreement_history"]
        all_hist = state.get("all_agree_history", [])
        final_ag = hist[-1] if hist else 0
        summary["domains"][domain] = {
            "final_weighted_agreement": round(final_ag, 3),
            "final_all_agree":          round(all_hist[-1] if all_hist else 0, 3),
            "weighted_history":         [round(a, 3) for a in hist],
            "all_agree_history":        [round(a, 3) for a in all_hist],
            "converged":                state["converged"],
            "regression_detected":      state.get("regression_detected", False),
            "iterations_run":           state["iteration"],
            "final_rubric_version":     state["rubric"].get("version", 0),
            "irreducible_count":        len(state["irreducible"]),
            "rubric_changes_made":      len(state["rubric_change_log"])
        }
        hist_str = " → ".join(f"{a:.0%}" for a in hist)
        status   = "✓ CONVERGED" if state["converged"] else "○ Not converged"
        print(f"\n[{domain}]")
        print(f"  {status}")
        print(f"  Weighted: {final_ag:.1%}  All-agree: "
              f"{all_hist[-1]:.1%}" if all_hist else f"  Weighted: {final_ag:.1%}")
        print(f"  History:  {hist_str}")
        print(f"  Rubric:   v{state['rubric'].get('version', 0)} "
              f"({len(state['rubric_change_log'])} changes)")

    summary_out = os.path.join(REPORT_DIR, "calibration_summary.json")
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ {summary_out}")
    print(f"\nTotal time: {summary['total_time_minutes']:.1f} min")

    return summary


if __name__ == "__main__":
    run_calibration()
