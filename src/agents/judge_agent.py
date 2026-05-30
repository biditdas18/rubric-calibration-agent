import ollama

# Three genuinely different models — same temperature
# Variation comes from architecture differences, not randomness
JUDGE_MODELS = [
    "llama3.1:latest",   # Meta — 8B
    "mistral:latest",    # Mistral AI — 7B
    "gemma2:latest",     # Google — 9B
]

JUDGE_TEMPERATURE = 0.1  # Same for all three — controlled evaluation


def build_judge_prompt(transcript: str, domain: str, rubric: dict) -> str:
    high_criteria = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(rubric["HIGH"])
    )
    low_criteria = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(rubric["LOW"])
    )

    return f"""You are an expert educational content evaluator for the {domain} domain.

DOMAIN CONTEXT: {rubric['domain_context']}

RUBRIC VERSION: {rubric['version']}

HIGH SIGNAL criteria (content must satisfy at least 2):
{high_criteria}

LOW SIGNAL criteria (any one dominant pattern = LOW):
{low_criteria}

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
REASON: one sentence explaining the primary factor"""


def judge_transcript(
    transcript: str,
    domain: str,
    rubric: dict,
    model: str,
    temperature: float = JUDGE_TEMPERATURE
) -> dict:
    prompt = build_judge_prompt(transcript, domain, rubric)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": 200}
        )
        text = response["message"]["content"].strip()

        result = {
            "model": model,
            "label": None,
            "confidence": None,
            "matched_high": [],
            "matched_low": [],
            "reason": "",
            "raw": text
        }

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("LABEL:"):
                val = line.replace("LABEL:", "").strip().upper()
                result["label"] = val if val in ["HIGH", "LOW"] else None
            elif line.startswith("CONFIDENCE:"):
                result["confidence"] = line.replace(
                    "CONFIDENCE:", ""
                ).strip()
            elif line.startswith("MATCHED_HIGH:"):
                nums = line.replace("MATCHED_HIGH:", "").strip()
                result["matched_high"] = [
                    int(x.strip()) for x in nums.split(",")
                    if x.strip().isdigit()
                ]
            elif line.startswith("MATCHED_LOW:"):
                nums = line.replace("MATCHED_LOW:", "").strip()
                result["matched_low"] = [
                    int(x.strip()) for x in nums.split(",")
                    if x.strip().isdigit()
                ]
            elif line.startswith("REASON:"):
                result["reason"] = line.replace("REASON:", "").strip()

        if result["label"] is None:
            text_upper = text.upper()
            if "HIGH" in text_upper[:100]:
                result["label"] = "HIGH"
            else:
                result["label"] = "LOW"

        return result

    except Exception as e:
        return {
            "model": model,
            "label": "LOW",
            "confidence": "LOW",
            "matched_high": [],
            "matched_low": [],
            "reason": f"Error: {str(e)}",
            "raw": ""
        }


def run_three_judges(
    transcript: str,
    domain: str,
    rubric: dict
) -> dict:
    """
    Run transcript through 3 different models at same temperature.
    Variation comes from model architecture, not randomness.
    Models: Llama 3.1 (Meta), Mistral (Mistral AI), Gemma 2 (Google)
    """
    results = []

    for model in JUDGE_MODELS:
        result = judge_transcript(
            transcript=transcript,
            domain=domain,
            rubric=rubric,
            model=model,
            temperature=JUDGE_TEMPERATURE
        )
        results.append(result)

    labels = [
        r["label"] for r in results
        if r["label"] in ["HIGH", "LOW"]
    ]
    high_count = labels.count("HIGH")
    low_count = labels.count("LOW")
    majority_label = "HIGH" if high_count >= 2 else "LOW"
    agreement = (
        max(high_count, low_count) / len(labels) if labels else 0
    )
    all_agree = len(set(labels)) == 1 if labels else False

    return {
        "majority_label": majority_label,
        "agreement": agreement,
        "all_agree": all_agree,
        "individual_results": results,
        "votes": {"HIGH": high_count, "LOW": low_count},
        "models_used": JUDGE_MODELS,
        "temperature_used": JUDGE_TEMPERATURE
    }
