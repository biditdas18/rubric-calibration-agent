import json
import time
import csv
import os
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_agent import run_three_judges
from coordinator_agent import run_coordinator

# ── CONFIG ──────────────────────────────────────────────────
MAX_TIME_SECONDS = 30 * 60
MAX_ITERATIONS = 10
TARGET_AGREEMENT = 0.85
MIN_DELTA = 0.02
OLLAMA_MODEL = "llama3.1:latest"

BASE_DIR = "/Users/biditdas/Desktop/snr-submission/rubric-calibration-agent"
REPORT_DIR = os.path.join(BASE_DIR, "reports/rubric_calibration")
DATA_PATH = os.path.join(BASE_DIR, "data/review_queue.csv")
RUBRIC_PATH = os.path.join(BASE_DIR, "src/agents/rubrics.json")
# ─────────────────────────────────────────────────────────────


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
    results = []
    disagreements = []
    agreements = 0

    for t in transcripts:
        if t["index"] in irreducible_indices:
            continue

        result = run_three_judges(
            t["transcript"], domain, rubric, OLLAMA_MODEL
        )
        result["transcript_index"] = t["index"]
        result["transcript"] = t["transcript"]
        results.append(result)

        if result["all_agree"]:
            agreements += 1
        else:
            disagreements.append({**t, **result})

    total = len(results)
    agreement_rate = agreements / total if total > 0 else 0

    return {
        "agreement_rate": agreement_rate,
        "total_evaluated": total,
        "agreements": agreements,
        "disagreements": len(disagreements),
        "disagreement_details": disagreements,
        "all_results": results
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
            "rubric": rubrics[d],
            "iteration": 0,
            "converged": False,
            "irreducible": set(),
            "final_results": None
        }
        for d in domains
    }

    print(f"\n{'='*60}")
    print("SNR RUBRIC CALIBRATION SYSTEM")
    print(f"Domains: {domains}")
    for d in domains:
        print(f"  {d}: {len(transcripts_by_domain[d])} transcripts")
    print(f"Target: {TARGET_AGREEMENT:.0%} | "
          f"Max iter: {MAX_ITERATIONS} | Max time: 30 min")
    print(f"Judges: {OLLAMA_MODEL} | Coordinator: Claude Sonnet")
    print(f"{'='*60}\n")

    iteration = 0

    while True:
        iteration += 1
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60

        print(f"\n{'─'*60}")
        print(f"ITERATION {iteration} | Time elapsed: {elapsed_min:.1f} min")
        print(f"{'─'*60}")

        if elapsed > MAX_TIME_SECONDS:
            print("STOPPING: Time limit reached (30 min)")
            break

        if all(s["converged"] for s in domain_state.values()):
            print("STOPPING: All domains converged to 85%+")
            break

        if iteration > MAX_ITERATIONS:
            print("STOPPING: Max iterations reached")
            break

        any_active = False

        for domain in domains:
            state = domain_state[domain]

            if state["converged"]:
                print(f"\n[{domain}] Already converged ✓ — skipping")
                continue

            if state["iteration"] >= MAX_ITERATIONS:
                print(f"\n[{domain}] Max iterations reached — skipping")
                continue

            any_active = True
            print(f"\n[{domain}] Running {len(transcripts_by_domain[domain])} transcripts × 3 judges...")

            eval_result = evaluate_domain(
                transcripts_by_domain[domain],
                domain,
                state["rubric"],
                state["irreducible"]
            )

            agreement = eval_result["agreement_rate"]
            state["agreement_history"].append(agreement)
            state["iteration"] += 1

            prev = (state["agreement_history"][-2]
                    if len(state["agreement_history"]) > 1 else 0)
            delta = agreement - prev

            print(f"  Agreement: {agreement:.1%} (delta: {delta:+.1%})")
            print(f"  Agreed: {eval_result['agreements']} | "
                  f"Disagreed: {eval_result['disagreements']} | "
                  f"Total: {eval_result['total_evaluated']}")

            if agreement >= TARGET_AGREEMENT:
                state["converged"] = True
                state["final_results"] = eval_result
                print(f"  ✓ CONVERGED at {agreement:.1%}!")
                continue

            # Stagnation check
            if (len(state["agreement_history"]) >= 3 and
                    all(abs(state["agreement_history"][-i-1] -
                            state["agreement_history"][-i-2]) < MIN_DELTA
                        for i in range(2))):
                print(f"  ⚠ Stagnated — rubric at model capability limit")
                state["final_results"] = eval_result
                continue

            # Run coordinator
            if eval_result["disagreements"] > 0:
                print(f"  Running coordinator on "
                      f"{eval_result['disagreements']} disagreements...")
                coord_result = run_coordinator(
                    domain=domain,
                    rubric=state["rubric"],
                    disagreements=eval_result["disagreement_details"],
                    iteration=state["iteration"],
                    agreement_history=state["agreement_history"]
                )

                if coord_result["success"]:
                    state["rubric"] = coord_result["updated_rubric"]
                    new_irr = set(
                        coord_result.get("irreducible_transcripts", [])
                    )
                    state["irreducible"].update(new_irr)
                    print(f"  Rubric updated → v{state['rubric']['version']}")
                    if coord_result["reasoning"]:
                        print(f"  Coordinator: "
                              f"{coord_result['reasoning'][:120]}...")
                    if new_irr:
                        print(f"  Flagged irreducible: {new_irr}")
                else:
                    print(f"  Coordinator failed — keeping current rubric")

            state["final_results"] = eval_result

        if not any_active:
            print("\nAll domains done — stopping.")
            break

    # ── SAVE OUTPUTS ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE — SAVING OUTPUTS")
    print(f"{'='*60}")

    # Save final rubrics
    final_rubrics = {d: s["rubric"] for d, s in domain_state.items()}
    rubric_out = os.path.join(REPORT_DIR, "calibrated_rubrics.json")
    with open(rubric_out, "w") as f:
        json.dump(final_rubrics, f, indent=2)
    print(f"✓ Rubrics saved: {rubric_out}")

    # Save calibrated labels
    label_rows = []
    for domain, state in domain_state.items():
        if state["final_results"]:
            for r in state["final_results"]["all_results"]:
                label_rows.append({
                    "domain": domain,
                    "transcript_index": r["transcript_index"],
                    "signal_level": r["majority_label"],
                    "agreement": round(r["agreement"], 3),
                    "votes_high": r["votes"]["HIGH"],
                    "votes_low": r["votes"]["LOW"],
                    "all_agree": r["all_agree"]
                })

    labels_out = os.path.join(REPORT_DIR, "calibrated_labels.csv")
    if label_rows:
        with open(labels_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=label_rows[0].keys())
            writer.writeheader()
            writer.writerows(label_rows)
        print(f"✓ Labels saved: {labels_out}")

    # Save summary
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "total_time_minutes": round((time.time() - start_time) / 60, 2),
        "total_iterations": iteration,
        "target_agreement": TARGET_AGREEMENT,
        "model_judges": OLLAMA_MODEL,
        "model_coordinator": "claude-sonnet-4-20250514",
        "domains": {}
    }

    print(f"\n{'─'*60}")
    print("FINAL RESULTS")
    print(f"{'─'*60}")

    for domain, state in domain_state.items():
        final_agr = (state["agreement_history"][-1]
                     if state["agreement_history"] else 0)
        summary["domains"][domain] = {
            "final_agreement": round(final_agr, 3),
            "agreement_history": [round(a, 3)
                                  for a in state["agreement_history"]],
            "converged": state["converged"],
            "iterations_run": state["iteration"],
            "final_rubric_version": state["rubric"].get("version", 0),
            "irreducible_count": len(state["irreducible"])
        }
        history_str = " → ".join(
            f"{a:.0%}" for a in state["agreement_history"]
        )
        status = "✓ CONVERGED" if state["converged"] else "○ Not converged"
        print(f"\n[{domain}]")
        print(f"  Status: {status}")
        print(f"  Final agreement: {final_agr:.1%}")
        print(f"  History: {history_str}")
        print(f"  Rubric version: {state['rubric'].get('version', 0)}")

    summary_out = os.path.join(REPORT_DIR, "calibration_summary.json")
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved: {summary_out}")

    print(f"\nTotal time: {summary['total_time_minutes']:.1f} min")
    print(f"\nNext step: use calibrated_rubrics.json to re-label")
    print(f"your SNR gold dataset with improved judges.")

    return summary


if __name__ == "__main__":
    run_calibration()
