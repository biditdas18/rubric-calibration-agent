import json
import anthropic


def build_coordinator_prompt(
    domain: str,
    rubric: dict,
    disagreements: list,
    iteration: int,
    agreement_history: list,
    judge_models: list
) -> str:
    disagreement_text = ""
    for i, d in enumerate(disagreements[:10]):
        model_votes = {
            r["model"].split(":")[0]: r["label"]
            for r in d["individual_results"]
        }
        reasons = {
            r["model"].split(":")[0]: r["reason"]
            for r in d["individual_results"]
        }
        disagreement_text += f"""
Transcript {i+1} (excerpt):
{d['transcript'][:400]}...

Model votes: {model_votes}
Reasons per model: {reasons}
HIGH criteria matched: {[r['matched_high'] for r in d['individual_results']]}
LOW criteria matched: {[r['matched_low'] for r in d['individual_results']]}
---"""

    history_text = " -> ".join(
        [f"Iter {i}: {a:.0%}" for i, a in enumerate(agreement_history)]
    )

    return f"""You are a rubric calibration expert. Three different LLM judges
(Llama 3.1, Mistral, Gemma 2) are evaluating educational video transcripts
for signal quality. All models use the same rubric at temperature=0.1.
Disagreements reflect genuine model architecture differences in interpreting
the rubric criteria — not randomness.

Your job: make the rubric precise enough that all three models reach the
same conclusion at least 85% of the time.

DOMAIN: {domain}
JUDGE MODELS: {judge_models}
TEMPERATURE: 0.1 (same for all)
ITERATION: {iteration}
AGREEMENT HISTORY: {history_text}
CURRENT AGREEMENT: {agreement_history[-1]:.0%}
TARGET: 85%

CURRENT RUBRIC:
{json.dumps(rubric, indent=2)}

TRANSCRIPTS WHERE MODELS DISAGREED:
{disagreement_text}

DIAGNOSIS INSTRUCTIONS:
1. Identify which criterion each model is applying differently
2. Note: if Llama says HIGH but Mistral + Gemma say LOW, the criterion
   is likely too permissive — tighten it
3. If all three apply different criteria, the criterion is fundamentally
   ambiguous — rewrite it with a concrete example
4. For general_education ONLY: never require actionability for HIGH

CLASSIFY EACH FAILURE:
A) Ambiguous wording — criterion interpreted differently by different architectures
B) Missing criterion — real pattern exists but rubric doesn't capture it
C) Conflicting criteria — two rubric criteria point in opposite directions
D) Irreducible — content is genuinely borderline, no rubric change will help

Make SURGICAL changes only. Do not rewrite the entire rubric.
Preserve criteria that are already working.

Respond ONLY in this exact JSON format with no other text:
{{
  "diagnosis": [
    {{
      "failure_type": "A or B or C or D",
      "criterion_type": "HIGH or LOW",
      "criterion_number": 1,
      "models_disagreeing": ["llama3.1", "mistral"],
      "problem": "one sentence describing the ambiguity",
      "fix": "revised criterion text — must be unambiguous (null if type D)"
    }}
  ],
  "irreducible_transcripts": [],
  "updated_HIGH": ["full criterion 1 text", "full criterion 2 text"],
  "updated_LOW": ["full criterion 1 text", "full criterion 2 text"],
  "updated_conflict_rule": "updated conflict rule text",
  "reasoning": "2-3 sentences explaining what changed and why"
}}"""


def run_coordinator(
    domain: str,
    rubric: dict,
    disagreements: list,
    iteration: int,
    agreement_history: list,
    judge_models: list = None
) -> dict:
    if judge_models is None:
        judge_models = ["llama3.1:latest", "mistral:latest", "gemma2:latest"]

    client = anthropic.Anthropic()

    prompt = build_coordinator_prompt(
        domain=domain,
        rubric=rubric,
        disagreements=disagreements,
        iteration=iteration,
        agreement_history=agreement_history,
        judge_models=judge_models
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = message.content[0].text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)

        updated_rubric = rubric.copy()
        updated_rubric["version"] = rubric.get("version", 0) + 1
        updated_rubric["HIGH"] = result.get(
            "updated_HIGH", rubric["HIGH"]
        )
        updated_rubric["LOW"] = result.get(
            "updated_LOW", rubric["LOW"]
        )
        updated_rubric["conflict_rule"] = result.get(
            "updated_conflict_rule", rubric["conflict_rule"]
        )

        return {
            "updated_rubric": updated_rubric,
            "diagnosis": result.get("diagnosis", []),
            "irreducible_transcripts": result.get(
                "irreducible_transcripts", []
            ),
            "reasoning": result.get("reasoning", ""),
            "success": True
        }

    except Exception as e:
        print(f"  Coordinator error: {e}")
        return {
            "updated_rubric": rubric,
            "diagnosis": [],
            "irreducible_transcripts": [],
            "reasoning": f"Error: {str(e)}",
            "success": False
        }
