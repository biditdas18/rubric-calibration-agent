import json
import re
import time
import anthropic


def repair_json(raw: str) -> dict:
    """4-attempt JSON repair for coordinator output."""

    # Attempt 1 — parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2 — truncate to last valid closing brace
    try:
        last_brace = raw.rfind("}")
        if last_brace > 0:
            return json.loads(raw[:last_brace + 1])
    except json.JSONDecodeError:
        pass

    # Attempt 3 — extract outermost JSON object
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Attempt 4 — sanitize criterion_quote fields
    # (these contain verbatim rubric text with embedded quotes)
    try:
        sanitized = re.sub(
            r'("criterion_quote"\s*:\s*)"((?:[^"\\]|\\.)*)"',
            lambda m: m.group(1) + '"[criterion text]"',
            raw
        )
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    return None


def build_coordinator_prompt(
    domain: str,
    rubric: dict,
    disagreements: list,
    iteration: int,
    agreement_history: list,
    judge_models: list,
    change_log: list = None
) -> str:
    disagreement_text = ""
    for i, d in enumerate(disagreements[:8]):  # cap at 8 to keep response short
        model_votes = {
            r["model"].split(":")[0]: r["label"]
            for r in d["individual_results"]
        }
        reasons = {
            r["model"].split(":")[0]: r.get("reason", "")
            for r in d["individual_results"]
        }
        # Sanitize criterion quotes — strip embedded quotes to prevent JSON breaks
        criterion_quotes = {}
        for r in d["individual_results"]:
            key = r["model"].split(":")[0]
            quote = r.get("criterion_quote", "")
            # Remove embedded quotes from criterion text
            quote = quote.replace('"', "'").replace('\\', '')
            criterion_quotes[key] = quote[:150]  # cap length

        disagreement_text += f"""
Transcript {i+1}:
{d['transcript'][:300]}

Votes: {model_votes}
Reasons: {reasons}
Decisive criterion per model: {criterion_quotes}
HIGH matched: {[r.get('matched_high',[]) for r in d['individual_results']]}
LOW matched: {[r.get('matched_low',[]) for r in d['individual_results']]}
---"""

    history_text = " -> ".join(
        [f"Iter {i}: {a:.0%}" for i, a in enumerate(agreement_history)]
    )

    # Build change history text
    change_history_text = ""
    if change_log:
        change_history_text = "\nPREVIOUS RUBRIC CHANGES (do not undo these):\n"
        for entry in change_log[-3:]:  # last 3 changes only
            change_history_text += (
                f"Iteration {entry['iteration']}: "
                f"{entry['reasoning'][:100]}\n"
            )
    else:
        change_history_text = "\nNo previous changes — this is the first coordinator call.\n"

    return f"""You are a rubric calibration expert fixing annotation disagreements.

DOMAIN: {domain}
JUDGES: {[m.split(':')[0] for m in judge_models]}
ITERATION: {iteration}
HISTORY: {history_text}
CURRENT: {agreement_history[-1]:.0%} | TARGET: 80%
{change_history_text}

RUBRIC (current):
HIGH criteria: {json.dumps(rubric['HIGH'])}
LOW criteria: {json.dumps(rubric['LOW'])}
CONFLICT RULE: {rubric['conflict_rule']}

DISAGREEMENTS:
{disagreement_text}

TASK: Fix the rubric criteria causing disagreements.
Pattern: if 2 models agree but 1 consistently disagrees, fix the
criterion the outlier model is misapplying.

RULES:
- Make surgical changes only — do not rewrite criteria that are working
- Do NOT include quotation marks inside JSON string values
- Use single quotes or paraphrase instead of quoting criterion text
- Keep each criterion under 120 characters
- For general_education: actionability is NOT required for HIGH

Return ONLY this JSON, nothing else before or after:
{{
  "diagnosis": [
    {{
      "failure_type": "A or B or C or D",
      "criterion_type": "HIGH or LOW",
      "criterion_number": 1,
      "problem": "brief description without quotes",
      "fix": "revised criterion text without embedded quotes"
    }}
  ],
  "irreducible_transcripts": [],
  "updated_HIGH": ["criterion 1", "criterion 2", "criterion 3", "criterion 4"],
  "updated_LOW": ["criterion 1", "criterion 2", "criterion 3", "criterion 4", "criterion 5"],
  "updated_conflict_rule": "conflict rule text",
  "reasoning": "brief explanation of changes made"
}}"""


def run_coordinator(
    domain: str,
    rubric: dict,
    disagreements: list,
    iteration: int,
    agreement_history: list,
    judge_models: list = None,
    max_retries: int = 3,
    change_log: list = None
) -> dict:
    if judge_models is None:
        judge_models = ["llama3.1:latest", "mistral:latest", "qwen2.5:7b"]

    client = anthropic.Anthropic()
    prompt = build_coordinator_prompt(
        domain, rubric, disagreements,
        iteration, agreement_history, judge_models,
        change_log=change_log
    )

    for attempt in range(max_retries):
        try:
            print(f"    Coordinator attempt {attempt + 1}/{max_retries}...")

            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )

            text = message.content[0].text.strip()

            # Extract JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            # Try to parse with repair
            result = repair_json(text)

            if result is None:
                raise ValueError(f"JSON repair failed on attempt {attempt + 1}")

            # Validate required keys exist
            required = ["updated_HIGH", "updated_LOW", "updated_conflict_rule"]
            missing = [k for k in required if k not in result]
            if missing:
                raise ValueError(f"Missing keys: {missing}")

            # Build updated rubric
            updated_rubric = rubric.copy()
            updated_rubric["version"] = rubric.get("version", 0) + 1
            updated_rubric["HIGH"] = result["updated_HIGH"]
            updated_rubric["LOW"] = result["updated_LOW"]
            updated_rubric["conflict_rule"] = result["updated_conflict_rule"]

            return {
                "updated_rubric": updated_rubric,
                "diagnosis": result.get("diagnosis", []),
                "irreducible_transcripts": result.get("irreducible_transcripts", []),
                "reasoning": result.get("reasoning", ""),
                "success": True,
                "attempts": attempt + 1
            }

        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {str(e)[:80]}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"    All {max_retries} attempts failed — keeping rubric")
                return {
                    "updated_rubric": rubric,
                    "diagnosis": [],
                    "irreducible_transcripts": [],
                    "reasoning": f"Failed after {max_retries} attempts",
                    "success": False,
                    "attempts": max_retries
                }
