#!/usr/bin/env python3
"""Reproduce every figure in CRUCIBLE_cost_analysis.md.

Reads only committed repository files. Requires: pandas, tiktoken.
Run from the CRUCIBLE repository root:  python reproduce.py
"""
import json, statistics as st, sys
from pathlib import Path

_missing = []
for _m in ("pandas", "tiktoken"):
    try:
        __import__(_m)
    except ImportError:
        _missing.append(_m)
if _missing:
    sys.exit(
        "reproduce.py needs: " + ", ".join(_missing) + "\n"
        "Install them into the environment you are running this from:\n"
        "    pip install " + " ".join(_missing) + "\n"
        "tiktoken downloads its encoding file on first use, so that run needs\n"
        "network access; afterwards it is cached locally."
    )

import pandas as pd, tiktoken

ROOT = Path(__file__).resolve().parent
TRANSCRIPTS = ROOT.parent / "snr-detector" / "data" / "transcripts_new"
DOMAINS = ["career_selfimprovement", "tech_ai", "general_education"]
N_CAL, N_COORD = 20, 30
PRICE_IN, PRICE_OUT = 3.00, 15.00          # claude-sonnet-4-6, USD per 1M tokens

enc = tiktoken.get_encoding("cl100k_base")
tok = lambda s: len(enc.encode(s))

sys.path.insert(0, str(ROOT / "src"))
try:                                        # prefer the committed function
    from agents.judge_agent import build_judge_prompt
except Exception:                           # ollama not installed -> local mirror
    def build_judge_prompt(transcript, domain, rubric):
        hi = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["HIGH"]))
        lo = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["LOW"]))
        return (f"""You are an expert educational content evaluator for the {domain} domain.

DOMAIN CONTEXT: {rubric['domain_context']}

RUBRIC VERSION: {rubric['version']}

HIGH SIGNAL criteria (content must satisfy at least 2):
{hi}

LOW SIGNAL criteria (any one dominant pattern = LOW):
{lo}

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
REASON: one sentence explaining the primary factor
CRITERION_QUOTE: copy the exact text of the single criterion that most influenced your decision""")

summary = json.load(open(ROOT / "reports/rubric_calibration/calibration_summary.json"))
rub_v0  = json.load(open(ROOT / "src/agents/rubrics.json"))
rub_fin = json.load(open(ROOT / "reports/rubric_calibration/calibrated_rubrics.json"))
cache   = pd.read_csv(ROOT / "results/judge_call_cache.csv")

tx = {d: [json.loads(l) for l in open(TRANSCRIPTS / f"new_transcripts_{d}.jsonl", encoding="utf8") if l.strip()]
      for d in DOMAINS}

prompts = [tok(build_judge_prompt(r["transcript"], d, rb))
           for d in DOMAINS for rb in (rub_v0[d], rub_fin[d]) for r in tx[d][:N_CAL]]
comps = [tok(f"LABEL: {r.label}\nCONFIDENCE: {r.confidence}\n"
             f"REASON: {r.reason}\nCRITERION_QUOTE: {r.criterion_quote}") for r in cache.itertuples()]

n_calls = len(DOMAINS) * 10 * len(summary["judge_models"]) * N_CAL
judge_in, judge_out = st.mean(prompts) * n_calls, st.mean(comps) * n_calls

# coordinator payload: 8 real disagreement cases, built with the COMMITTED prompt builder
import importlib.util, types
def load_committed(name, path):
    """Import a repo module whose heavy deps may be absent (ollama/anthropic)."""
    for dep in ("ollama", "anthropic"):
        sys.modules.setdefault(dep, types.ModuleType(dep))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

coord_mod = load_committed("coordinator_agent", ROOT / "src/agents/coordinator_agent.py")
build_coordinator_prompt = coord_mod.build_coordinator_prompt

coord_in_each, coord_out_each = [], []
for d in DOMAINS:
    sub = cache[cache.domain == d]
    disagreements = [{
        "transcript": tx[d][n]["transcript"],
        "individual_results": [
            {"model": m,
             "label": "HIGH" if k % 2 else "LOW",
             "reason": sub.iloc[(n * 3 + k) % len(sub)]["reason"],
             "criterion_quote": sub.iloc[(n * 3 + k) % len(sub)]["criterion_quote"],
             "matched_high": [1, 3], "matched_low": [2]}
            for k, m in enumerate(summary["judge_models"])]} for n in range(8)]
    hist = summary["domains"][d]["weighted_history"][:3]
    coord_in_each.append(tok(build_coordinator_prompt(
        d, rub_v0[d], disagreements, 3, hist, summary["judge_models"], None)))
    coord_out_each.append(tok(json.dumps({
        "diagnosis": [{"failure_type": "A", "criterion_type": "HIGH", "criterion_number": 1,
                       "problem": "x" * 80, "fix": "y" * 110}],
        "irreducible_transcripts": [], "updated_HIGH": rub_fin[d]["HIGH"],
        "updated_LOW": rub_fin[d]["LOW"], "updated_conflict_rule": rub_fin[d]["conflict_rule"],
        "reasoning": "z" * 200})))

coord_in, coord_out = st.mean(coord_in_each) * N_COORD, st.mean(coord_out_each) * N_COORD
cost = lambda i, o: i / 1e6 * PRICE_IN + o / 1e6 * PRICE_OUT
crucible, frontier = cost(coord_in, coord_out), cost(judge_in, judge_out)

print(f"judge calls                {n_calls:,}")
print(f"judge prompt tokens/call   {st.mean(prompts):,.0f}  (range {min(prompts)}-{max(prompts)})")
print(f"judge output tokens/call   {st.mean(comps):,.0f}  (n={len(comps)} cached responses)")
print(f"coordinator in/out per call {st.mean(coord_in_each):,.0f} / {st.mean(coord_out_each):,.0f}")
print(f"judge totals        in {judge_in:,.0f}  out {judge_out:,.0f}")
print(f"coordinator totals  in {coord_in:,.0f}  out {coord_out:,.0f}")
print(f"\nCRUCIBLE as run      ${crucible:,.2f}   (${crucible/n_calls*1000:,.2f} per 1k judge calls)")
print(f"all-frontier         ${frontier:,.2f}   (${frontier/n_calls*1000:,.2f} per 1k judge calls)")
print(f"ratio                {frontier/crucible:.1f}x")
print(f"wall-clock (local)   {summary['total_time_minutes']:.1f} min")
