# Rubric Calibration Agent

Autonomous multi-agent system for calibrating LLM annotation rubrics
until inter-judge agreement reaches a target threshold.

Built as methodology infrastructure for SNR-Detect.
Related paper: https://github.com/biditdas18/snr-detector

## Architecture

- **3 Judge Agents** (Llama 3.1 8B via Ollama) — one per domain
- **1 Coordinator Agent** (Claude Sonnet) — diagnoses disagreement patterns
  and proposes targeted rubric fixes

## Stopping Conditions

- Target agreement reached (default 85%)
- Max iterations per domain reached (default 10)
- Global time limit reached (default 30 min)
- Agreement delta < 2% for 3 consecutive iterations (stagnation)

## Setup

```bash
pip install -r requirements.txt

# Make sure Ollama is running with llama3.1
ollama serve
ollama pull llama3.1
```

## Usage

```bash
# Copy your transcript data
cp /path/to/review_queue.csv data/

# Run calibration
cd src/agents
python calibration_loop.py
```

## Output

```
reports/rubric_calibration/
├── calibrated_rubrics.json    ← use these for re-labeling
├── calibrated_labels.csv      ← silver labels from final iteration
└── calibration_summary.json   ← full run report and agreement history
```

## How It Works

1. Three judge agents independently evaluate each transcript
2. Majority vote determines the label
3. Where all three disagree, the coordinator diagnoses the failure
4. Coordinator proposes surgical rubric fixes (not full rewrites)
5. Irreducible transcripts are flagged and excluded from calibration
6. Loop repeats until stopping condition is met

## Domain-Specific Rubrics

The system maintains separate rubrics per domain because signal quality
criteria are domain-dependent:

- **Career advice** — actionability requires procedural steps and timelines
- **Tech/AI** — actionability requires named technologies and technical reasoning
- **General education** — actionability is NOT required; conceptual clarity qualifies

This insight emerged from observing that a single domain-agnostic rubric
produced only 53% pairwise agreement — approaching random baseline for
binary classification.
