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
MAX_TIME_SECONDS = 60 * 60   # 1 hour — enough for 3-model batched sweep
MAX_ITERATIONS   = 10
TARGET_AGREEMENT = 0.85
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


def evaluate_domain_batched(
    transcripts: list,
    domain: str,
    rubric: dict,
    irreducible_indices: set
) -> dict:
    """
    Batch evaluation: all transcripts through model 1, then model 2, then
    model 3 — each model stays loaded in GPU memory for its full pass.
    3 model loads total instead of len(transcripts)*3.
    """
    active = [t for t in transcripts if t["index"] not in irreducible_indices]

    results_by_model = {}
    for model in JUDGE_MODELS:
        short = model.split(":")[0]
        print(f"    [{short}] evaluating {len(active)} transcripts...", flush=True)
        results_by_model[model] = run_all_transcripts_one_model(
            transcripts=active,
            domain=domain,
            rubric=rubric,
            model=model
        )

    merged = merge_model_results(active, results_by_model)

    agreements    = sum(1 for r in merged if r["all_agree"])
    disagreements = [r for r in merged if not r["all_agree"]]
    total         = len(merged)
    agreement_rate = agreements / total if total > 0 else 0

    return {
        "agreement_rate":       agreement_rate,
        "total_evaluated":      total,
        "agreements":           agreements,
        "disagreements":        len(disagreements),
        "disagreement_details": disagreements,
        "all_results":          merged
    }


def run_calibration():
    os.makedirs(REPORT_DIR, exist_ok=True)
    start_time = time.time()

    with open(RUBRIC_PATH, "r") as f:
        rubrics = json.load(f)

    transcripts_by_domain = load_transcripts(DATA_PATH)
    domains = list(rubrics.keys())

    domain_state = {
        d: {
            "agreement_history": [],
            "rubric":            rubrics[d],
            "iteration":         0,
            "converged":         False,
            "irreducible":       set(),
            "final_results":     None
        }
        for d in domains
    }

    print(f"\n{'='*60}")
    print("SNR RUBRIC CALIBRATION — 3 MODELS, BATCHED BY MODEL")
    print(f"{'='*60}")
    print(f"Evaluation order: all transcripts per model before switching")
    print(f"Judge models (temperature={JUDGE_TEMPERATURE}):")
    for m in JUDGE_MODELS:
        print(f"  • {m}")
    print(f"Coordinator: claude-sonnet-4-20250514")
    print(f"Target: {TARGET_AGREEMENT:.0%} | "
          f"Max iter: {MAX_ITERATIONS} | Max time: 60 min")
    print(f"Transcripts: {sum(len(v) for v in transcripts_by_domain.values())} "
          f"across {len(domains)} domains")
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
            print("STOPPING: Time limit (60 min)")
            break
        if all(s["converged"] for s in domain_state.values()):
            print("STOPPING: All domains converged")
            break
        if iteration > MAX_ITERATIONS:
            print("STOPPING: Max iterations")
            break

        any_active = False

        for domain in domains:
            state = domain_state[domain]

            if state["converged"]:
                print(f"\n[{domain}] Converged ✓ — skipping")
                continue
            if state["iteration"] >= MAX_ITERATIONS:
                print(f"\n[{domain}] Max iterations — skipping")
                continue

            any_active = True
            n = len(transcripts_by_domain[domain])
            print(f"\n[{domain}] {n} transcripts × 3 models = {n*3} calls "
                  f"(3 model loads)", flush=True)

            eval_result = evaluate_domain_batched(
                transcripts_by_domain[domain],
                domain,
                state["rubric"],
                state["irreducible"]
            )

            agreement = eval_result["agreement_rate"]
            state["agreement_history"].append(agreement)
            state["iteration"] += 1

            prev  = (state["agreement_history"][-2]
                     if len(state["agreement_history"]) > 1 else 0)
            delta = agreement - prev

            print(f"  Agreement: {agreement:.1%} (Δ {delta:+.1%})")
            print(f"  All agree: {eval_result['agreements']} / "
                  f"{eval_result['total_evaluated']}")
            print(f"  Disagreed: {eval_result['disagreements']}")

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
                    judge_models=JUDGE_MODELS
                )
                if coord["success"]:
                    state["rubric"] = coord["updated_rubric"]
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

        if not any_active:
            print("\nAll domains done.")
            break

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
                    "domain":           domain,
                    "transcript_index": r["transcript_index"],
                    "signal_level":     r["majority_label"],
                    "agreement":        round(r["agreement"], 3),
                    "votes_high":       r["votes"]["HIGH"],
                    "votes_low":        r["votes"]["LOW"],
                    "all_agree":        r["all_agree"],
                    "llama_label":      r["individual_results"][0]["label"],
                    "mistral_label":    r["individual_results"][1]["label"],
                    "gemma_label":      r["individual_results"][2]["label"]
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
        "judge_models":       JUDGE_MODELS,
        "judge_temperature":  JUDGE_TEMPERATURE,
        "coordinator_model":  "claude-sonnet-4-20250514",
        "evaluation_strategy": "batched_by_model",
        "domains": {}
    }

    print(f"\n{'─'*60}")
    print("FINAL RESULTS")
    print(f"{'─'*60}")

    for domain, state in domain_state.items():
        hist     = state["agreement_history"]
        final_ag = hist[-1] if hist else 0
        summary["domains"][domain] = {
            "final_agreement":      round(final_ag, 3),
            "agreement_history":    [round(a, 3) for a in hist],
            "converged":            state["converged"],
            "iterations_run":       state["iteration"],
            "final_rubric_version": state["rubric"].get("version", 0),
            "irreducible_count":    len(state["irreducible"])
        }
        hist_str = " → ".join(f"{a:.0%}" for a in hist)
        status   = "✓ CONVERGED" if state["converged"] else "○ Not converged"
        print(f"\n[{domain}]")
        print(f"  {status}")
        print(f"  Final:   {final_ag:.1%}")
        print(f"  History: {hist_str}")
        print(f"  Rubric:  v{state['rubric'].get('version', 0)}")

    summary_out = os.path.join(REPORT_DIR, "calibration_summary.json")
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ {summary_out}")
    print(f"\nTotal time: {summary['total_time_minutes']:.1f} min")

    return summary


if __name__ == "__main__":
    run_calibration()
