# CRUCIBLE — Rubric Calibration Agent

**What this is:** An autonomous multi-agent system that takes a labeling
rubric and a set of transcripts, then iteratively improves the rubric
until three different LLM judges consistently agree on labels.
This is the reference implementation for CRUCIBLE, a multi-agent framework for automated annotation rubric calibration.

**Why it exists:** When you ask three different AI models to label the
same content using the same rubric, they often disagree — not because
the content is ambiguous, but because the rubric is ambiguous. This
agent figures out exactly where the rubric is failing and fixes it,
domain by domain, until agreement reaches a target threshold.

**What it produces:** A set of calibrated, domain-specific rubrics
(in JSON) that can be used to label new data with high inter-model
agreement. Used as the labeling backbone for
[SNR-Detector](https://github.com/biditdas18/snr-detector).

---

## Paper

**CRUCIBLE: A Multi-Agent Framework for Automated Annotation Rubric Calibration via Iterative Inter-Judge Disagreement Resolution**
Bidit Das — Independent Researcher, July 2026

Available on SSRN: https://ssrn.com/abstract=7025019
DOI: https://dx.doi.org/10.2139/ssrn.7025019

**Accepted** at ATRACC 2026 — the 3rd Symposium on AI Trustworthiness and Risk Assessment for
Challenged Contexts, AAAI 2026 Fall Symposium Series. The accepted version, revised to report the
human validation at n=20 per domain and to add the majority-class baseline correction, is
[`papers/crucible_atracc2026.pdf`](papers/crucible_atracc2026.pdf) (source `.tex` alongside it;
building it needs the AAAI-27 author kit, which is not redistributed here).

> **Note on the SSRN preprint and `papers/crucible.tex`.** Both predate the n=20 human validation
> and report a between-domain match-rate contrast that the analysis in Results below supersedes.
> Read `papers/crucible_atracc2026.pdf` for the corrected claims.

---

## Results

The system ran 10 iterations per domain (~2.8 hours / 170.7 min) on a consumer MacBook (no GPU),
calibrating rubrics on a local three-judge panel (Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B). Raw
weighted agreement is inflated by label prevalence, so we report Fleiss' kappa and Gwet's AC1
(chance-corrected) alongside it:

| Domain | Weighted | All-Agree | Fleiss κ | Gwet AC1 | Status |
|---|---|---|---|---|---|
| General Education | 96.7% | 90% | −0.034 | +0.929 | Converged (v0, no changes) |
| Technology & AI | 81.7% | 45% | +0.246 | +0.286 | Converged (v1, 1 change) |
| Career & Self-Improvement | 70.0% | 10% | −0.350 | −0.080 | Stagnated (best retained) |

The three domains separate cleanly under chance correction. General Education's agreement is
genuine (AC1 = 0.93; κ ≈ 0 is the prevalence paradox under a ~97%-one-class distribution).
Technology & AI shows fair agreement and was the one domain a rubric revision helped — but its
81.7% weighted agreement corresponds to a chance-corrected AC1 of only 0.286, and the human
validation study below shows why that matters. Career & Self-Improvement sits at
the floor of the weighted metric with below-chance agreement — one judge (Qwen 2.5) applies a
co-occurrence criterion inconsistently with the other two. A judge-replacement test confirmed this
is panel-specific: replacing Qwen raised inter-judge agreement to 88.3%, but the panel's
majority-vote output was mechanically identical (0 of 20 labels changed), because the two
retained judges already agreed on 90% of transcripts.

### Human validation — 20 transcripts per domain, two disjoint cohorts

The full corpus was scored by human annotators: 20 transcripts per domain, run in two rounds of
10, each round recruiting three annotators per domain (18 in total) with no annotator appearing in
more than one round or domain. Round 1 covers the even global transcript indices, round 2 the odd.

Every rating is in one file — [`gold-sampled-dataset/human_validation_all_rounds.csv`](gold-sampled-dataset/human_validation_all_rounds.csv)
— and [`src/agents/exp_b14_human_validation_n20.py`](src/agents/exp_b14_human_validation_n20.py)
reproduces every number below from it, with the bootstrap seed fixed:

| Domain | Human AC1 r1 / r2 | Pooled AC1 (95% CI) | Judge AC1 | vs CRUCIBLE r1 / r2 | Majority-class baseline |
|---|---|---|---|---|---|
| General Education | 0.631 / 0.756 | 0.697 [0.43, 0.89] | +0.929 | 90% / 100% | 100% |
| Technology & AI | 0.631 / −0.026 | 0.327 [−0.02, 0.64] | +0.286 | 50% / **80%** | 60% |
| Career & Self-Improvement | 0.333 / 0.234 | 0.241 [−0.06, 0.56] | −0.080 | 50% / 80% | 90% |

Two results follow, and both **supersede** the n=10 analysis in the SSRN preprint:

1. **Match against human consensus is uninformative in this design.** Because each domain's
   labels are heavily skewed to one class, a rater who labels everything with that class scores
   100% / 60% / 90%. Only 1 of the 6 domain-round cells beats its own baseline, and the pooled
   rate of 45/60 = 75% is *below* the pooled baseline of 83%. The between-domain match contrast
   reported earlier (90% vs 50%) is prevalence-dominated and does not survive this correction.
   Doubling the sample independently failed to replicate it: Technology & AI moved 50% → 80%.
2. **Judge and human reliability rank the domains identically** (0.929 / 0.286 / −0.080 against
   0.697 / 0.327 / 0.241). The judges find difficult what the humans find difficult, which is
   evidence *against* reading the panel's failures as misalignment.

**The corrected central finding** is about the stopping criterion, not the judges. The weighted
agreement metric `A = (u + ⅔(N−u))/N` has a floor of 0.667 at zero unanimity, so a threshold of
τ = 0.80 is satisfied at just 8 of 20 unanimous items — the loop halts and reports convergence
while 60% of items still carry judge disagreement. Technology & AI's "converged" 81.7% is a
chance-corrected 0.286. Any stopping rule defined on an agreement statistic with a non-zero floor
halts at a chance-corrected level fixed by that floor, so τ must be set relative to the floor
rather than on the unit interval.

Full analysis — including the frontier-model comparison, reasoning-before-label ablation,
held-out generalization, and rubric portability tests — is in the papers under `papers/`.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                   Calibration Loop                       │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Judge 1  │   │ Judge 2  │   │ Judge 3  │  (Ollama)  │
│  │ Llama3.1 │   │ Mistral  │   │ Qwen2.5  │            │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│       │              │              │                    │
│       └──────────────┼──────────────┘                   │
│                      │                                   │
│              Measure agreement                           │
│              Find disagreements                          │
│                      │                                   │
│                      ▼                                   │
│          ┌───────────────────────┐                      │
│          │  Coordinator Agent    │  (Claude Sonnet)     │
│          │  Diagnose → Fix rubric│                      │
│          └───────────┬───────────┘                      │
│                      │                                   │
│              New rubric version                          │
│              → repeat until agreement                    │
│                target reached                            │
└─────────────────────────────────────────────────────────┘
```

1. **Three judge agents** (running locally via Ollama) independently
   evaluate each transcript using the current rubric
2. Agreement is measured — transcripts where all three agree are
   considered stable; disagreements are flagged
3. **The coordinator agent** (Claude Sonnet via API) reads the
   disagreement patterns and proposes targeted rubric fixes
4. The updated rubric is used in the next iteration
5. Transcripts that never converge (irreducible disagreements) are
   flagged and excluded
6. **Human validation** — calibrated rubrics should be audited against independent human
   judgment on a stratified sample before treating automated convergence as a quality signal.
   The system's convergence metric is a stopping rule, not a correctness guarantee.

---

## Repository Structure

```
rubric-calibration-agent/
├── src/agents/
│   ├── calibration_loop.py     # Main entry point — runs the loop
│   ├── judge_agent.py          # Three local LLM judges (Ollama)
│   ├── coordinator_agent.py    # Claude Sonnet rubric fixer
│   ├── rubrics.json            # Starting rubric (pre-calibration)
│   └── exp_b14_human_validation_n20.py   # Reproduces every n=20 human number
├── data/
│   └── review_queue.csv        # Input transcripts (domain + text)
├── gold-sampled-dataset/
│   ├── human_validation_all_rounds.csv        # ALL human ratings, both rounds
│   ├── human_validation_ground_truth_n20.csv  # Per-item consensus, all 60
│   └── human_validation_ground_truth.csv      # Round-1 only (n=30, kept)
├── reports/rubric_calibration/
│   ├── calibrated_rubrics.json # Output: use these for labeling
│   ├── calibrated_labels.csv   # Silver labels from final iteration
│   └── calibration_summary.json# Full run log + agreement history
├── results/
│   └── human_validation_n20.json  # AC1, baselines, bootstrap CIs
├── papers/
│   ├── crucible_atracc2026.pdf # Accepted ATRACC 2026 version (current)
│   ├── crucible.tex            # Long manuscript (predates n=20; superseded)
│   └── fig_reliability.pdf     # Judge vs human reliability figure
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) installed and running locally
- Anthropic API key (for the coordinator agent)

### Step 1 — Install Python dependencies

```bash
git clone https://github.com/biditdas18/rubric-calibration-agent.git
cd rubric-calibration-agent
pip install -r requirements.txt
```

### Step 2 — Pull the three judge models via Ollama

```bash
# Start Ollama (if not already running)
ollama serve

# Pull the three judge models (~15 GB total)
ollama pull llama3.1
ollama pull mistral
ollama pull qwen2.5:7b
```

### Step 3 — Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Never commit this to git.

### Step 4 — Add your transcripts

Your input file goes in `data/review_queue.csv`.
It needs two columns: `domain` and `transcript`.

```
domain,transcript
career_selfimprovement,"Here are 5 tips to be more productive..."
tech_ai,"Today we'll explore how transformers work..."
general_education,"The French Revolution began in 1789..."
```

Supported domains: `career_selfimprovement`, `tech_ai`, `general_education`

### Step 5 — Run calibration

```bash
cd src/agents
python calibration_loop.py
```

The loop runs for up to 3 hours or 10 iterations per domain,
whichever comes first. Progress is printed to the terminal.

---

## Output Files

After the run completes, find your results in `reports/rubric_calibration/`:

| File | What it is |
|---|---|
| `calibrated_rubrics.json` | The final rubrics — use these to label new data |
| `calibrated_labels.csv` | Majority-vote labels for all input transcripts |
| `calibration_summary.json` | Per-domain agreement history across all iterations |

The `calibrated_rubrics.json` is what feeds directly into the
[SNR-Detector](https://github.com/biditdas18/snr-detector) labeling pipeline.

Run `python src/agents/compute_agreement.py` to reproduce the Fleiss κ / Gwet AC1 table above
from the saved labels.

---

## Configuration

Edit the top of `src/agents/calibration_loop.py` to adjust:

```python
MAX_TIME_SECONDS = 3 * 60 * 60  # Total time limit (default: 3 hours)
MAX_ITERATIONS   = 10            # Max iterations per domain
TARGET_AGREEMENT = 0.80          # Stop when weighted agreement >= this
MIN_DELTA        = 0.02          # Stop if improvement < 2% for 3 rounds
```

---

## Requirements

```
ollama        # Local LLM runner (judge agents)
anthropic     # Claude API (coordinator agent)
pandas
numpy
```

---

## Related

- **SNR-Detector** — the classifier that uses these calibrated rubrics as its labeling backbone:  
  https://github.com/biditdas18/snr-detector  
  SSRN preprint: https://ssrn.com/abstract=6866745

- **CRUCIBLE paper** — full experimental detail, human validation study, and additional analyses:  
  https://ssrn.com/abstract=7025019  
  DOI: 10.2139/ssrn.7025019

---

## Author

Bidit Das — Independent Researcher
GitHub: @biditdas18
