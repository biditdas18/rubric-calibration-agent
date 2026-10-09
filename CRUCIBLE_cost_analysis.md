# CRUCIBLE — Annotation Cost Analysis

**Prepared:** 2026-10-09 · Bidit Das ·
biditdas18@gmail.com
**Subject:** cost of CRUCIBLE's local-ensemble annotation pipeline against the
equivalent frontier-model API workload.

All quantities below are computed from files committed in the CRUCIBLE
repository. The prompts are reconstructed exactly, not estimated: the judge
template truncates each transcript to 3,000 characters, so every prompt is
fully determined by the committed rubric and transcript files. `reproduce.py`
regenerates every number in this document.

---

## 1. What was measured

| Quantity | Value | Source |
|---|---|---|
| Judge models (local, via Ollama, temperature 0) | llama3.1, mistral, qwen2.5:7b | `calibration_summary.json` |
| Coordinator model (frontier, API) | `claude-sonnet-4-6` | `calibration_summary.json` |
| Domains × iterations | 3 × 10 | `calibration_summary.json` |
| Transcripts per domain | 20 | paper, §human validation |
| Judge calls in the calibration loop | **1,800** | 3 domains × 10 iters × 3 judges × 20 |
| Transcript-labelings produced | 600 | 3 × 10 × 20 |
| Coordinator calls (upper bound) | 30 | one per iteration |
| Total wall-clock | 170.7 min | `calibration_summary.json` |

Token volumes, counted with `tiktoken` `cl100k_base`:

| | Input tokens/call | Output tokens/call |
|---|---|---|
| Judge (mean over 120 reconstructed prompts) | 1,088 | 84 |
| Coordinator (8-disagreement payload) | 3,434 | 456 |

| | Input | Output |
|---|---|---|
| Judge total (1,800 calls) | 1,958,370 | 150,358 |
| Coordinator total (30 calls) | 103,010 | 13,670 |

---

## 2. Cost

Priced at `claude-sonnet-4-6` published rates, **$3.00 per million input
tokens and $15.00 per million output tokens**.

| Configuration | API spend | Per 1,000 judge calls |
|---|---|---|
| **CRUCIBLE as run** — local judges, frontier coordinator only | **$0.51** | **$0.29** |
| **All-frontier equivalent** — the same 1,800 annotation calls on the API | **$8.13** | **$4.52** |
| Ratio | — | **15.8×** |

The absolute figures are small because this is a 60-transcript study. The
rate is the transferable quantity:

| Annotation volume | CRUCIBLE | All-frontier | Difference |
|---|---|---|---|
| 1,000 judge calls | $0.29 | $4.52 | $4.23 |
| 100,000 | $29.00 | $451.69 | $423.00 |
| 1,000,000 | $290.00 | $4,516.94 | $4,230.00 |

The local judges incur **no API charge**. They ran on consumer hardware
(12-core workstation, 24 GB RAM) in 170.7 minutes of
wall-clock. Electricity and hardware amortisation are **not** quantified in
this document — no power measurement was taken, so no figure is offered.

---

## 3. What this analysis does **not** establish

**It does not show that the two configurations produce annotations of
comparable quality, and the CRUCIBLE paper affirmatively declines that claim.**
This section is part of the exhibit, not a disclaimer appended to it.

- The paper's central finding is that an automated convergence criterion "can
  certify agreement that chance-corrected measurement does not support, and
  that the certification is an artifact of the agreement metric being
  optimized."
- The paper reports as a negative result that agreement between human consensus
  and the system's labels "carries almost no evidential weight in this design":
  five of six domain-round cells fail to beat a majority-class baseline.
- Porting the calibrated rubric to a frontier judge "move[s] the optimized
  metric without moving agreement with the human."

The defensible claim is therefore narrow: **at equal annotation volume the
local-ensemble configuration costs 16× less in API
spend. Quality parity between the two is not established by this work.**

---

## 4. Method limits a reviewer should know

1. **Tokenizer proxy.** Counts use OpenAI's `cl100k_base`. Anthropic's
   tokenizer differs, so token totals carry a modest error; the *ratio* between
   configurations is insensitive to this because both sides are counted the
   same way.
2. **Prices are third-party.** Rates were taken from public pricing aggregators
   in October 2026, not from Anthropic's own pricing page. Verify against
   `anthropic.com/pricing` before relying on the absolute dollar figures.
3. **Coordinator call count is an upper bound.** One call per iteration is
   assumed (30 total). Two of three domains made fewer rubric changes than
   iterations, so actual coordinator spend may be lower — which would widen the
   gap, not narrow it.
4. **Transcript selection.** The first 20 transcripts per domain file are used.
   Per-call token means vary little across the corpus (judge prompt range
   974–2532 tokens), so this choice does not
   drive the result.
5. **No instrumented re-run.** Token counts are reconstructed from committed
   prompts and cached responses rather than logged at call time.

---

## 5. Verification

| SHA-256 (first 16) | File |
|---|---|
| `33783a5293a93db8` | `results/judge_call_cache.csv` |
| `ef7a550ed600c9ff` | `reports/rubric_calibration/calibration_summary.json` |
| `5f88d79e122436ab` | `reports/rubric_calibration/calibrated_rubrics.json` |
| `292960b03bf9afe1` | `src/agents/rubrics.json` |
| `7ccad3baa7666718` | `src/agents/judge_agent.py` |
| `767e30f633ed95db` | `src/agents/coordinator_agent.py` |

Run `python reproduce.py` from the repository root. It requires `pandas` and `tiktoken` (`pip install pandas tiktoken`); tiktoken downloads its encoding file on first use, so that run needs network access. It reads only the files
above and prints every figure in sections 1 and 2.

`reproduce.py` is committed alongside this document and its printed output is the source of every figure above.
