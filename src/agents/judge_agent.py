import ollama


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
    model: str = "llama3.1:latest"
) -> dict:
    prompt = build_judge_prompt(transcript, domain, rubric)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 200}
        )
        text = response["message"]["content"].strip()

        result = {
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
            if "HIGH" in text.upper()[:50]:
                result["label"] = "HIGH"
            else:
                result["label"] = "LOW"

        return result

    except Exception as e:
        return {
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
    rubric: dict,
    model: str = "llama3.1:latest"
) -> dict:
    """Run same transcript through judge 3 times, take majority."""
    results = []
    for run in range(3):
        result = judge_transcript(transcript, domain, rubric, model)
        results.append(result)

    labels = [r["label"] for r in results if r["label"] in ["HIGH", "LOW"]]
    high_count = labels.count("HIGH")
    low_count = labels.count("LOW")

    majority_label = "HIGH" if high_count >= 2 else "LOW"
    agreement = max(high_count, low_count) / len(labels) if labels else 0

    return {
        "majority_label": majority_label,
        "agreement": agreement,
        "all_agree": agreement == 1.0,
        "individual_results": results,
        "votes": {"HIGH": high_count, "LOW": low_count}
    }
