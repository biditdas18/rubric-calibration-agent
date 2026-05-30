import json
import anthropic


def build_coordinator_prompt(
    domain: str,
    rubric: dict,
    disagreements: list,
    iteration: int,
    agreement_history: list
) -> str:
    disagreement_text = ""
    for i, d in enumerate(disagreements[:10]):
        disagreement_text += f"""
Transcript {i+1} (excerpt):
{d['transcript'][:400]}...

Judge votes: {d['votes']}
Individual reasons: {[r['reason'] for r in d['individual_results']]}
HIGH criteria matched: {[r['matched_high'] for r in d['individual_results']]}
LOW criteria matched: {[r['matched_low'] for r in d['individual_results']]}
---"""

    history_text = " -> ".join(
        [f"Iter {i}: {a:.0%}" for i, a in enumerate(agreement_history)]
    )

    current_rubric_text = json.dumps(rubric, indent=2)

    return f"""You are a rubric calibration expert. Your job is to improve a
content quality rubric so that 3 independent LLM judges evaluating
the same transcript reach the same conclusion at least 85% of the time.

DOMAIN: {domain}
ITERATION: {iteration}
AGREEMENT HISTORY: {history_text}
CURRENT AGREEMENT: {agreement_history[-1]:.0%}
TARGET AGREEMENT: 85%

CURRENT RUBRIC:
{current_rubric_text}

TRANSCRIPTS WHERE JUDGES DISAGREED:
{disagreement_text}

YOUR TASK:
1. Identify which specific rubric criteria are causing disagreement
2. Classify each failure:
   A) Ambiguous wording - criterion is unclear
   B) Missing criterion - real pattern not captured
   C) Conflicting criteria - two criteria contradict each other
   D) Irreducible - genuinely ambiguous content, rubric cannot help
3. For failures A, B, C only: propose ONE specific revision per criterion
4. Do NOT rewrite the entire rubric - make surgical targeted changes only
5. For general_education domain: remember actionability is NOT required

Respond in this EXACT JSON format with no other text:
{{
  "diagnosis": [
    {{
      "failure_type": "A or B or C or D",
      "criterion_type": "HIGH or LOW",
      "criterion_number": 1,
      "problem": "one sentence describing the ambiguity",
      "fix": "revised criterion text (null if type D)"
    }}
  ],
  "irreducible_transcripts": [],
  "updated_HIGH": ["criterion 1 text", "criterion 2 text"],
  "updated_LOW": ["criterion 1 text", "criterion 2 text"],
  "updated_conflict_rule": "conflict rule text",
  "reasoning": "2-3 sentences explaining the key changes made"
}}"""


def run_coordinator(
    domain: str,
    rubric: dict,
    disagreements: list,
    iteration: int,
    agreement_history: list
) -> dict:
    client = anthropic.Anthropic()

    prompt = build_coordinator_prompt(
        domain, rubric, disagreements, iteration, agreement_history
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
        updated_rubric["HIGH"] = result.get("updated_HIGH", rubric["HIGH"])
        updated_rubric["LOW"] = result.get("updated_LOW", rubric["LOW"])
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
