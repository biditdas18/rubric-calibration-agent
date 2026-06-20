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
Technology & AI shows fair agreement and was the one domain a rubric revision helped. Career
& Self-Improvement sits at the floor of the weighted metric with below-chance agreement — one
judge (Qwen 2.5) applies a co-occurrence criterion inconsistently with the other two, a
model-level limitation no rubric revision fixed. This is the capability-ceiling finding: the
system diagnoses *which* regime a domain is in rather than merely raising raw agreement. Full
analysis is in the CRUCIBLE paper.

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

---

## Repository Structure

```
rubric-calibration-agent/
├── src/agents/
│   ├── calibration_loop.py     # Main entry point — runs the loop
│   ├── judge_agent.py          # Three local LLM judges (Ollama)
│   ├── coordinator_agent.py    # Claude Sonnet rubric fixer
│   └── rubrics.json            # Starting rubric (pre-calibration)
├── data/
│   └── review_queue.csv        # Input transcripts (domain + text)
├── reports/rubric_calibration/
│   ├── calibrated_rubrics.json # Output: use these for labeling
│   ├── calibrated_labels.csv   # Silver labels from final iteration
│   └── calibration_summary.json# Full run log + agreement history
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

- **SNR-Detector** — the classifier that uses these calibrated rubrics:
  https://github.com/biditdas18/snr-detector

---

## Author

Bidit Das — Independent Researcher
GitHub: @biditdas18
