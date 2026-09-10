#!/usr/bin/env python3
"""Analyze question-quality records already saved by experiment_all_in_one.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DIMENSIONS = (
    "skill_relevance",
    "difficulty_match",
    "clarity",
    "non_redundancy",
    "contextual_coherence",
)


def load_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            missing = [
                name for name in (
                    "strategy", "candidate_id", "turn", "difficulty", "cluster",
                    "question", "question_quality_scores", "overall_question_quality",
                    "regeneration_count", "question_generation_attempts",
                ) if name not in row
            ]
            if missing:
                raise ValueError(
                    f"Line {line_number} is not a question-quality record; missing {missing}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def metric_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_std": sample_std(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def first_attempt_score(row: Mapping) -> float:
    attempts = row["question_generation_attempts"]
    if not attempts:
        raise ValueError("question_generation_attempts cannot be empty")
    scores = attempts[0]["quality_scores"]
    return float(scores["overall_question_quality"])


def question_judge_calls(row: Mapping) -> int:
    calls = 0
    for attempt in row["question_generation_attempts"]:
        for detail in attempt.get("question_judge_details", []):
            calls += 1 + int(detail.get("format_retry_count", 0))
    return calls


def summarize_rows(rows: Sequence[Mapping]) -> Dict[str, float]:
    result: Dict[str, float] = {"n_turns": len(rows)}
    for dimension in DIMENSIONS:
        values = [float(row["question_quality_scores"][dimension]) for row in rows]
        result.update(metric_stats(values, dimension))

    overall = [float(row["overall_question_quality"]) for row in rows]
    first = [first_attempt_score(row) for row in rows]
    thresholds = [float(row.get("quality_threshold", 7.0)) for row in rows]
    regenerations = [int(row["regeneration_count"]) for row in rows]
    attempts = [len(row["question_generation_attempts"]) for row in rows]
    final_pass = [
        bool(row.get("quality_threshold_met", score >= threshold))
        for row, score, threshold in zip(rows, overall, thresholds)
    ]
    gate_enabled = [bool(row.get("quality_gate_enabled", True)) for row in rows]
    result.update(metric_stats(overall, "overall_question_quality"))
    result.update(metric_stats(first, "first_attempt_overall_quality"))
    result.update({
        "first_attempt_pass_rate": mean([
            float(score >= threshold) for score, threshold in zip(first, thresholds)
        ]),
        "final_pass_rate": mean([float(value) for value in final_pass]),
        "below_threshold_rate": mean([float(not value) for value in final_pass]),
        "quality_gate_enabled_rate": mean([float(value) for value in gate_enabled]),
        "exhausted_failure_rate": mean([
            float(enabled and not passed)
            for enabled, passed in zip(gate_enabled, final_pass)
        ]),
        "no_regeneration_rate": mean([float(value == 0) for value in regenerations]),
        "average_regeneration_count": mean(regenerations),
        "average_generation_attempts": mean(attempts),
        "average_question_judge_api_calls": mean([
            question_judge_calls(row) for row in rows
        ]),
        "format_retry_turn_rate": mean([
            float(question_judge_calls(row) > sum(
                len(attempt.get("question_judge_details", []))
                for attempt in row["question_generation_attempts"]
            ))
            for row in rows
        ]),
    })
    return result


def grouped_summaries(
    rows: Sequence[Mapping], keys: Sequence[str]
) -> List[Dict]:
    groups: Dict[Tuple, List[Mapping]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    results = []
    for group_key in sorted(groups, key=lambda value: tuple(map(str, value))):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary.update(summarize_rows(groups[group_key]))
        results.append(summary)
    return results


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float], repeats: int = 10000, seed: int = 42
) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    boot = sorted(
        mean([rng.choice(values) for _ in values])
        for _ in range(repeats)
    )
    return percentile(boot, 0.025), percentile(boot, 0.975)


def paired_strategy_comparison(
    rows: Sequence[Mapping], first: str = "bridge", second: str = "random"
) -> Dict:
    candidate_values: Dict[Tuple[str, object], List[float]] = defaultdict(list)
    for row in rows:
        candidate_values[(str(row["strategy"]), row["candidate_id"])].append(
            float(row["overall_question_quality"])
        )
    first_ids = {
        candidate_id for strategy, candidate_id in candidate_values if strategy == first
    }
    second_ids = {
        candidate_id for strategy, candidate_id in candidate_values if strategy == second
    }
    common = sorted(first_ids & second_ids, key=str)
    deltas = [
        mean(candidate_values[(first, candidate_id)])
        - mean(candidate_values[(second, candidate_id)])
        for candidate_id in common
    ]
    if not deltas:
        return {
            "available": False,
            "reason": f"No candidates shared by {first} and {second}",
        }
    ci_low, ci_high = bootstrap_mean_ci(deltas)
    delta_std = sample_std(deltas)
    return {
        "available": True,
        "comparison": f"{first}_minus_{second}",
        "unit": "candidate mean across turns",
        "n_paired_candidates": len(deltas),
        "mean_overall_quality_delta": mean(deltas),
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "paired_effect_size_dz": mean(deltas) / delta_std if delta_std > 0 else 0.0,
        "first_win_rate": mean([float(value > 0) for value in deltas]),
        "tie_rate": mean([float(value == 0) for value in deltas]),
        "note": (
            "Strategies select different skills and difficulties, so this paired delta "
            "measures end-to-end sequence quality rather than generator quality alone."
        ),
    }


def flatten_turn(row: Mapping) -> Dict:
    flattened = {
        "strategy": row["strategy"],
        "candidate_id": row["candidate_id"],
        "turn": row["turn"],
        "skill": row.get("skill", ""),
        "cluster": row["cluster"],
        "difficulty": row["difficulty"],
        "question": row["question"],
        "overall_question_quality": row["overall_question_quality"],
        "first_attempt_overall_quality": first_attempt_score(row),
        "quality_threshold": row.get("quality_threshold", 7.0),
        "quality_threshold_met": row.get("quality_threshold_met", ""),
        "regeneration_count": row["regeneration_count"],
        "generation_attempts": len(row["question_generation_attempts"]),
        "question_judge_api_calls": question_judge_calls(row),
    }
    flattened.update({
        dimension: row["question_quality_scores"][dimension]
        for dimension in DIMENSIONS
    })
    return flattened


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_compact(strategy_rows: Sequence[Mapping], comparison: Mapping) -> None:
    columns = (
        "strategy", "n_turns", "overall_question_quality_mean",
        "first_attempt_pass_rate", "final_pass_rate", "average_regeneration_count",
    )
    print("\nQuestion quality by strategy")
    print(" | ".join(columns))
    for row in strategy_rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print(" | ".join(values))
    print("\nPaired comparison")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=r"runs_question_quality_fast\exp3_qwen\turns.jsonl",
        help="Path to the existing turns.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory; defaults to <turns directory>/quality_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent / "quality_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)

    by_strategy = grouped_summaries(rows, ["strategy"])
    by_difficulty = grouped_summaries(rows, ["strategy", "difficulty"])
    by_cluster = grouped_summaries(rows, ["strategy", "cluster"])
    comparison = paired_strategy_comparison(rows)
    report = {
        "input": str(input_path.resolve()),
        "n_records": len(rows),
        "strategies": sorted({row["strategy"] for row in rows}),
        "overall": summarize_rows(rows),
        "by_strategy": by_strategy,
        "bridge_vs_random": comparison,
        "interpretation_notes": [
            "Question-quality scores never enter BayesianAbilityModel.",
            "Turns from the same candidate are repeated measures; the comparison pairs candidate means.",
            "Different strategies select different skill/difficulty mixtures; inspect stratified CSV files.",
        ],
    }

    (output_dir / "question_quality_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "question_quality_by_strategy.csv", by_strategy)
    write_csv(output_dir / "question_quality_by_difficulty.csv", by_difficulty)
    write_csv(output_dir / "question_quality_by_cluster.csv", by_cluster)
    write_csv(output_dir / "question_quality_turns.csv", [flatten_turn(row) for row in rows])
    print_compact(by_strategy, comparison)
    print(f"\nSaved analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
