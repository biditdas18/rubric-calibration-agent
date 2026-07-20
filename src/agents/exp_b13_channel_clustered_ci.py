#!/usr/bin/env python3
"""
B13 -- Channel-level dependence disclosure and correction.

Channel metadata IS available for the 60-transcript corpus
(snr-detector/data/labels/video_selection_60.csv, joined by global
transcript_index -- positional alignment verified against review_queue.csv
during A1/A3 recon). Recomputes the human-vs-CRUCIBLE bootstrap CIs
(Table 3 / Section 5.5.2) resampling at the channel level instead of the
transcript level, since several channels contribute more than one
transcript to the 30-item human-validation sample.

Output: results/channel_clustered_ci.json
"""
import csv
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
SNR_BASE = BASE.parent / "snr-detector"
VIDEO_SELECTION = SNR_BASE / "data" / "labels" / "video_selection_60.csv"
HUMAN_GOLD = BASE / "gold-sampled-dataset" / "human_validation_ground_truth.csv"
CALIBRATED_LABELS = BASE / "reports" / "rubric_calibration" / "calibrated_labels.csv"
OUT_PATH = BASE / "results" / "channel_clustered_ci.json"

N_BOOT = 10000


def bootstrap_ci_transcript(items, seed=0):
    rng = random.Random(seed)
    n = len(items)
    matches = [m for _, m in items]
    point = sum(matches) / n
    boots = []
    for _ in range(N_BOOT):
        sample = [matches[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return point, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]


def bootstrap_ci_channel(items, seed=0):
    rng = random.Random(seed)
    by_ch = {}
    for ch, m in items:
        by_ch.setdefault(ch, []).append(m)
    channels = list(by_ch.keys())
    n_ch = len(channels)
    matches = [m for _, m in items]
    point = sum(matches) / len(items)
    boots = []
    for _ in range(N_BOOT):
        sampled_channels = [channels[rng.randrange(n_ch)] for _ in range(n_ch)]
        pooled = []
        for ch in sampled_channels:
            pooled.extend(by_ch[ch])
        boots.append(sum(pooled) / len(pooled))
    boots.sort()
    return point, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]


def main():
    with open(VIDEO_SELECTION, encoding="utf-8") as f:
        sel60 = list(csv.DictReader(f))
    with open(HUMAN_GOLD, encoding="utf-8") as f:
        human_rows = list(csv.DictReader(f))
    with open(CALIBRATED_LABELS, encoding="utf-8") as f:
        local_by_key = {(r["domain"], int(r["transcript_index"])): r["signal_level"]
                         for r in csv.DictReader(f)}

    by_domain = {}
    for r in human_rows:
        idx = int(r["transcript_index"])
        domain = r["domain"]
        channel = sel60[idx]["channel"]
        match = 1 if local_by_key[(domain, idx)] == r["human_consensus_label"] else 0
        by_domain.setdefault(domain, []).append((channel, match))

    results = {"experiment": "B13 -- channel-clustered bootstrap CI", "n_boot": N_BOOT, "by_domain": {}}
    overall_items = []
    for dom, items in by_domain.items():
        overall_items.extend(items)
        n_ch = len(set(ch for ch, _ in items))
        pt, lo_t, hi_t = bootstrap_ci_transcript(items)
        pc, lo_c, hi_c = bootstrap_ci_channel(items)
        results["by_domain"][dom] = {
            "n_channels": n_ch, "point_estimate_pct": round(pt * 100, 1),
            "transcript_resampled_ci_pct": [round(lo_t * 100, 1), round(hi_t * 100, 1)],
            "channel_resampled_ci_pct": [round(lo_c * 100, 1), round(hi_c * 100, 1)],
            "ci_width_transcript": round((hi_t - lo_t) * 100, 1),
            "ci_width_channel": round((hi_c - lo_c) * 100, 1),
        }

    n_ch_overall = len(set(ch for ch, _ in overall_items))
    pt, lo_t, hi_t = bootstrap_ci_transcript(overall_items)
    pc, lo_c, hi_c = bootstrap_ci_channel(overall_items)
    results["overall"] = {
        "n_channels": n_ch_overall, "point_estimate_pct": round(pt * 100, 1),
        "transcript_resampled_ci_pct": [round(lo_t * 100, 1), round(hi_t * 100, 1)],
        "channel_resampled_ci_pct": [round(lo_c * 100, 1), round(hi_c * 100, 1)],
        "ci_width_transcript": round((hi_t - lo_t) * 100, 1),
        "ci_width_channel": round((hi_c - lo_c) * 100, 1),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"{'domain':28s} {'n_ch':>5s} {'transcript-CI':>18s} {'channel-CI':>18s}")
    for d, v in results["by_domain"].items():
        print(f"{d:28s} {v['n_channels']:5d} {v['transcript_resampled_ci_pct']} {v['channel_resampled_ci_pct']}")
    o = results["overall"]
    print(f"{'OVERALL':28s} {o['n_channels']:5d} {o['transcript_resampled_ci_pct']} {o['channel_resampled_ci_pct']}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
