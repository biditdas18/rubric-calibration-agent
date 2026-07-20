#!/usr/bin/env python3
"""Shared helpers for the Pass-2 follow-up experiments (A1-A7): agreement
metrics, bootstrap CIs, and config/provenance recording (A6)."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from math import comb
from pathlib import Path

import ollama


class JudgeCallCache:
    """Resumable cache for judge_agent.judge_transcript() calls, keyed by
    (run_tag, domain, rubric_version, transcript_index, model). Backed by an
    append-only CSV so a long multi-hundred-call experiment can resume after
    an interruption without re-paying for completed calls."""

    FIELDNAMES = ["run_tag", "domain", "rubric_version", "transcript_index",
                  "model", "label", "confidence", "reason", "criterion_quote"]

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.done = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    key = (r["run_tag"], r["domain"], r["rubric_version"],
                           int(r["transcript_index"]), r["model"])
                    self.done[key] = r
        file_exists = self.cache_path.exists()
        self._f = open(self.cache_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=self.FIELDNAMES)
        if not file_exists:
            self._writer.writeheader()

    def get_or_call(self, judge_transcript_fn, run_tag, domain, rubric_version,
                     transcript_index, transcript, rubric, model):
        key = (run_tag, domain, rubric_version, transcript_index, model)
        if key in self.done:
            r = self.done[key]
            return {"model": model, "label": r["label"], "confidence": r["confidence"],
                    "reason": r["reason"], "criterion_quote": r["criterion_quote"]}
        result = judge_transcript_fn(transcript=transcript, domain=domain, rubric=rubric, model=model)
        row = {
            "run_tag": run_tag, "domain": domain, "rubric_version": rubric_version,
            "transcript_index": transcript_index, "model": model,
            "label": result.get("label") or "LOW",
            "confidence": result.get("confidence") or "",
            "reason": result.get("reason") or "",
            "criterion_quote": result.get("criterion_quote") or "",
        }
        self._writer.writerow(row)
        self._f.flush()
        self.done[key] = row
        return {"model": model, "label": row["label"], "confidence": row["confidence"],
                "reason": row["reason"], "criterion_quote": row["criterion_quote"]}

    def close(self):
        self._f.close()


def fleiss_style_metrics(rows_of_3_labels: list) -> dict:
    """3-rater weighted/all-agree/pairwise/kappa/AC1 — same formulas used
    throughout this project (compute_agreement.py, judge_agent_cot.py)."""
    N = len(rows_of_3_labels)
    weighted = sum(1.0 if len(set(r)) == 1 else 0.667 for r in rows_of_3_labels) / N
    all_agree = sum(len(set(r)) == 1 for r in rows_of_3_labels) / N
    pairwise = sum(a == b for r in rows_of_3_labels for a, b in combinations(r, 2)) / (3 * N)
    Pa = sum((comb(r.count("HIGH"), 2) + comb(r.count("LOW"), 2)) / comb(3, 2) for r in rows_of_3_labels) / N
    piH = sum(r.count("HIGH") for r in rows_of_3_labels) / (3 * N)
    piL = 1 - piH
    Pe_f, Pe_g = piH**2 + piL**2, 2 * piH * piL
    kappa = (Pa - Pe_f) / (1 - Pe_f) if Pe_f != 1 else float("nan")
    ac1 = (Pa - Pe_g) / (1 - Pe_g) if Pe_g != 1 else float("nan")
    return {
        "n": N, "weighted": round(weighted, 4), "all_agree": round(all_agree, 4),
        "pairwise": round(pairwise, 4), "fleiss_kappa": round(kappa, 4), "gwet_ac1": round(ac1, 4),
    }


def two_rater_agreement_metrics(pairs: list) -> dict:
    """2-rater agreement/kappa/AC1 — same formula as zeroshot_baseline.py's
    agreement_metrics, reproduced here to keep this module dependency-free."""
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    p_o = agree / n
    p_high_a = sum(a == "HIGH" for a, _ in pairs) / n
    p_high_b = sum(b == "HIGH" for _, b in pairs) / n
    p_e_kappa = p_high_a * p_high_b + (1 - p_high_a) * (1 - p_high_b)
    kappa = (p_o - p_e_kappa) / (1 - p_e_kappa) if p_e_kappa != 1 else float("nan")
    pi_high = (p_high_a + p_high_b) / 2
    p_e_gwet = 2 * pi_high * (1 - pi_high)
    ac1 = (p_o - p_e_gwet) / (1 - p_e_gwet) if p_e_gwet != 1 else float("nan")
    return {"n": n, "agreement": round(p_o, 4), "cohen_kappa": round(kappa, 4), "gwet_ac1": round(ac1, 4)}


def paired_bootstrap_ci(paired_values: list, n_boot: int = 10000, seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean of paired per-item differences
    (e.g. final-weight minus v0-weight per transcript)."""
    import random
    rng = random.Random(seed)
    n = len(paired_values)
    point = sum(paired_values) / n
    boot_means = []
    for _ in range(n_boot):
        sample = [paired_values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot) - 1]
    return {"point_estimate": round(point, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n_boot": n_boot}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def git_commit_hash(repo_dir) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception as e:
        return f"unavailable ({e})"


def ollama_model_config(model: str) -> dict:
    """Digest + quantization + context length via `ollama show`."""
    try:
        r = subprocess.run(["ollama", "show", model], capture_output=True, text=True, timeout=30)
        text = r.stdout
        cfg = {"model": model, "raw_show_output": text.strip()}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("parameters"):
                cfg["parameters"] = line.split(None, 1)[1].strip()
            elif line.startswith("context length"):
                cfg["context_length"] = line.split(None, 2)[2].strip() if len(line.split()) > 2 else line
            elif line.startswith("quantization"):
                cfg["quantization"] = line.split(None, 1)[1].strip()
        r2 = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
        for line in r2.stdout.splitlines():
            if line.startswith(model.split(":")[0]):
                parts = line.split()
                if len(parts) >= 2:
                    cfg["digest_id"] = parts[1]
        return cfg
    except Exception as e:
        return {"model": model, "error": str(e)}


def record_provenance(models: list, dataset_paths: list, rubric_paths: list,
                       prompt_template: str, repo_dir) -> dict:
    """A6: version/config recording attached to every Pass-2 experiment result."""
    return {
        "experiment_date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit_hash(repo_dir),
        "prompt_template_sha256_16": sha256_text(prompt_template),
        "dataset_file_hashes": {str(p): sha256_file(p) for p in dataset_paths},
        "rubric_file_hashes": {str(p): sha256_file(p) for p in rubric_paths},
        "model_configs": {m: ollama_model_config(m) for m in models},
        "note": (
            "Follow-up experiments were conducted under the closest recoverable "
            "configuration and should be interpreted as diagnostic extensions "
            "rather than exact controlled replications of the original run. No "
            "digest snapshot from the original calibration run was saved to diff "
            "against; model files are unchanged on disk since that run per local "
            "modification timestamps, which is suggestive but not proof of an "
            "exact match."
        ),
    }
