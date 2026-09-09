#!/usr/bin/env python3
"""
Single-file experiments for Bayesian risk-aware technical interviews.

Kept:
  1. A fixed six-cluster / 29-skill graph.
  2. A joint Gaussian Bayesian ability model.
  3. BRIDGE, uncertainty-only, random, and strong fixed-order selectors.
  4. Synthetic Exp. 1/2 and an optional three-role Qwen end-to-end pilot.

Removed:
  JD parsing, resume parsing, LLM-generated skill trees, multi-agent routing,
  competency interviews and training-data code.

Examples:
  python experiment_all_in_one.py --mode exp1 --output runs
  python experiment_all_in_one.py --mode exp2 --seeds 20 --output runs
  python experiment_all_in_one.py --mode all  --seeds 20 --output runs

Qwen calls are never included in --mode all.  They require an explicit mode:
  export DASHSCOPE_API_KEY=...
  python experiment_all_in_one.py --mode qwen \
      --question-model qwen-turbo --question-judge-model qwen-max \
      --candidate-model qwen-max --judge-model qwen-max \
      --qwen-candidates 30 --qwen-questions 12 \
      --strategies bridge random --question-judge-repeats 3 \
      --quality-threshold 7 --max-regenerations 2 --judge-repeats 3 \
      --confirm-api-calls --output runs

All abilities and evaluator scores use the [1, 10] scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # Keep the stdlib-only synthetic experiments runnable.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int, initial: int = 0, desc: str = "") -> None:
            self.total, self.n, self.desc = total, initial, desc

        def __enter__(self):
            print(f"{self.desc}: {self.n}/{self.total}", file=sys.stderr)
            return self

        def __exit__(self, *_args) -> None:
            print(f"{self.desc}: {self.n}/{self.total}", file=sys.stderr)

        def update(self, amount: int = 1) -> None:
            self.n += amount


# =============================================================================
# Fixed experimental assets
# =============================================================================

SKILL_CLUSTERS: Dict[str, List[str]] = {
    "programming": [
        "Java", "Python", "Concurrency", "Data Structures", "Design Patterns",
    ],
    "database_cache": [
        "SQL Indexing", "Transactions", "MySQL Tuning", "Redis Caching",
        "Cache Consistency",
    ],
    "distributed_systems": [
        "CAP Trade-offs", "Consensus", "Distributed Transactions",
        "Service Discovery", "Fault Tolerance",
    ],
    "middleware": [
        "Kafka", "Message Reliability", "Idempotency", "Async Architecture",
    ],
    "cloud_devops": [
        "Docker", "Kubernetes", "CI/CD", "Observability", "Cloud Architecture",
    ],
    "system_design": [
        "Scalability", "Load Balancing", "API Design", "Security",
        "Performance Optimization",
    ],
}

SKILLS = [skill for cluster in SKILL_CLUSTERS.values() for skill in cluster]
SKILL_TO_CLUSTER = {
    skill: cluster
    for cluster, skills in SKILL_CLUSTERS.items()
    for skill in skills
}
CLUSTERS = list(SKILL_CLUSTERS)
assert len(SKILLS) == 29

# Equal cluster contribution; leaves within a cluster share its weight.
JOB_WEIGHTS = {
    skill: 1.0 / len(CLUSTERS) / len(SKILL_CLUSTERS[cluster])
    for cluster, skills in SKILL_CLUSTERS.items()
    for skill in skills
}

DIFFICULTY_TARGET = {"easy": 3.0, "medium": 5.5, "hard": 8.0}
EXP1_SKILLS = [
    "Java", "SQL Indexing", "CAP Trade-offs", "Kubernetes",
]

LEVEL_CONFIG = {
    "junior": {"mean": 3.5, "std": 1.2, "performance_std": 1.5, "strong": 1},
    "mid": {"mean": 5.8, "std": 0.9, "performance_std": 1.0, "strong": 3},
    "senior": {"mean": 7.5, "std": 0.7, "performance_std": 0.6, "strong": 5},
}

LEVEL_ZH = {
    "junior": "基础薄弱到初级",
    "mid": "中等熟练",
    "senior": "高级熟练",
}

FIXED_ORDER: List[str] = []
for position in range(max(len(v) for v in SKILL_CLUSTERS.values())):
    for cluster in CLUSTERS:
        skills = SKILL_CLUSTERS[cluster]
        if position < len(skills):
            FIXED_ORDER.append(skills[position])
assert sorted(FIXED_ORDER) == sorted(SKILLS)

_EPS = 1e-12


# =============================================================================
# General utilities
# =============================================================================

def clip(value: float, low: float = 1.0, high: float = 10.0) -> float:
    return min(max(float(value), low), high)


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": sum(values) / len(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return numerator / (dx * dy) if dx > 0 and dy > 0 else 0.0


def ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[order[position]] = average_rank
        start = end
    return result


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metric inputs must be non-empty and aligned")
    errors = [prediction - truth for truth, prediction in zip(y_true, y_pred)]
    return {
        "n": len(y_true),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
        "pearson": pearson(y_true, y_pred),
        "spearman": pearson(ranks(y_true), ranks(y_pred)),
    }


def difficulty_efficiency(mean: float, difficulty: str, bandwidth: float = 2.0) -> float:
    target = DIFFICULTY_TARGET[difficulty]
    z = (mean - target) / max(bandwidth, _EPS)
    return 0.15 + 0.85 * math.exp(-0.5 * z * z)


def closest_difficulty(mean: float) -> str:
    return min(DIFFICULTY_TARGET, key=lambda d: abs(mean - DIFFICULTY_TARGET[d]))


# =============================================================================
# Candidate simulation
# =============================================================================

@dataclass
class CandidateProfile:
    candidate_id: int
    level: str
    global_theta: float
    performance_std: float
    strong_clusters: List[str]
    skill_theta: Dict[str, float]


def balanced_levels(n_candidates: int) -> List[str]:
    if n_candidates % 3 != 0:
        raise ValueError(
            f"n_candidates={n_candidates} is not divisible by 3; "
            "use 30 (10 per level) or another multiple of three"
        )
    per_level = n_candidates // 3
    return ["junior"] * per_level + ["mid"] * per_level + ["senior"] * per_level


def generate_profiles(n_candidates: int, seed: int) -> List[CandidateProfile]:
    rng = random.Random(seed)
    levels = balanced_levels(n_candidates)
    rng.shuffle(levels)
    profiles = []
    for candidate_id, level in enumerate(levels):
        config = LEVEL_CONFIG[level]
        global_theta = clip(rng.gauss(config["mean"], config["std"]))
        strong = sorted(rng.sample(CLUSTERS, config["strong"]))
        cluster_effect = {}
        for cluster in CLUSTERS:
            base = 0.75 if cluster in strong else -0.35
            cluster_effect[cluster] = base + rng.gauss(0.0, 0.25)
        skill_theta = {
            skill: clip(
                global_theta
                + cluster_effect[SKILL_TO_CLUSTER[skill]]
                + rng.gauss(0.0, 0.45)
            )
            for skill in SKILLS
        }
        profiles.append(CandidateProfile(
            candidate_id=candidate_id,
            level=level,
            global_theta=global_theta,
            performance_std=config["performance_std"],
            strong_clusters=strong,
            skill_theta=skill_theta,
        ))
    return profiles


def true_job_score(profile: CandidateProfile) -> float:
    return sum(JOB_WEIGHTS[skill] * profile.skill_theta[skill] for skill in SKILLS)


def simulate_score(
    profile: CandidateProfile,
    skill: str,
    difficulty: str,
    observation_index: int,
    seed: int,
    evaluator_std: float = 0.6,
) -> Tuple[float, float]:
    """Return a noisy score and its true observation standard deviation."""
    efficiency = difficulty_efficiency(profile.skill_theta[skill], difficulty)
    base_std = math.sqrt(profile.performance_std ** 2 + evaluator_std ** 2)
    observation_std = base_std / math.sqrt(efficiency)
    rng = random.Random(stable_seed("observation", seed, profile.candidate_id, skill, observation_index))
    score = clip(profile.skill_theta[skill] + rng.gauss(0.0, observation_std))
    return score, observation_std


# =============================================================================
# Joint Gaussian Bayesian ability model
# =============================================================================

class BayesianAbilityModel:
    """Exact linear-Gaussian posterior over all 29 skill abilities."""

    def __init__(
        self,
        graph_enabled: bool = True,
        prior_mean: float = 5.0,
        prior_std: float = 2.5,
        observation_std: float = 1.2,
        graph_correlation: float = 0.65,
        graph_length_scale: float = 1.5,
    ) -> None:
        self.skills = list(SKILLS)
        self.index = {skill: index for index, skill in enumerate(self.skills)}
        self.mean = [float(prior_mean)] * len(self.skills)
        self.observations = {skill: 0 for skill in self.skills}
        self.default_observation_variance = observation_std ** 2
        prior_variance = prior_std ** 2
        self.covariance = [[0.0] * len(self.skills) for _ in self.skills]
        for i, first in enumerate(self.skills):
            for j, second in enumerate(self.skills):
                if i == j:
                    self.covariance[i][j] = prior_variance
                elif graph_enabled:
                    distance = 2 if SKILL_TO_CLUSTER[first] == SKILL_TO_CLUSTER[second] else 4
                    self.covariance[i][j] = (
                        prior_variance
                        * graph_correlation
                        * math.exp(-distance / graph_length_scale)
                    )

    def skill_mean(self, skill: str) -> float:
        return self.mean[self.index[skill]]

    def skill_variance(self, skill: str) -> float:
        index = self.index[skill]
        return max(self.covariance[index][index], _EPS)

    def update(self, skill: str, score: float, observation_std: Optional[float] = None) -> None:
        j = self.index[skill]
        noise = (
            self.default_observation_variance
            if observation_std is None
            else max(float(observation_std) ** 2, _EPS)
        )
        old_mean = list(self.mean)
        old_covariance = [list(row) for row in self.covariance]
        denominator = max(old_covariance[j][j] + noise, _EPS)
        innovation = clip(score) - old_mean[j]
        gain = [old_covariance[i][j] / denominator for i in range(len(self.skills))]
        self.mean = [old_mean[i] + gain[i] * innovation for i in range(len(self.skills))]
        n = len(self.skills)
        updated = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                updated[i][k] = (
                    old_covariance[i][k]
                    - old_covariance[i][j] * old_covariance[j][k] / denominator
                )
        for i in range(n):
            updated[i][i] = max(updated[i][i], _EPS)
            for k in range(i + 1, n):
                symmetric = 0.5 * (updated[i][k] + updated[k][i])
                updated[i][k] = symmetric
                updated[k][i] = symmetric
        self.covariance = updated
        self.observations[skill] += 1

    def job_mean(self) -> float:
        return sum(JOB_WEIGHTS[skill] * self.skill_mean(skill) for skill in SKILLS)

    def job_variance(self) -> float:
        total = 0.0
        for first, wf in JOB_WEIGHTS.items():
            i = self.index[first]
            for second, ws in JOB_WEIGHTS.items():
                j = self.index[second]
                total += wf * ws * self.covariance[i][j]
        return max(total, 0.0)

    def total_variance(self) -> float:
        return sum(self.covariance[i][i] for i in range(len(self.skills)))

    def risk_components(
        self,
        skill: str,
        difficulty: str,
        recent: Sequence[str],
        job_variance: Optional[float] = None,
        total_variance: Optional[float] = None,
    ) -> Dict[str, float]:
        efficiency = difficulty_efficiency(self.skill_mean(skill), difficulty)
        observation_variance = self.default_observation_variance / efficiency
        j = self.index[skill]
        denominator = self.covariance[j][j] + observation_variance
        covariance_with_job = sum(
            JOB_WEIGHTS[name] * self.covariance[self.index[name]][j]
            for name in SKILLS
        )
        decision_reduction = covariance_with_job ** 2 / max(denominator, _EPS)
        global_reduction = sum(row[j] ** 2 for row in self.covariance) / max(denominator, _EPS)
        if job_variance is None:
            job_variance = self.job_variance()
        if total_variance is None:
            total_variance = self.total_variance()
        relative_decision = decision_reduction / max(job_variance, _EPS)
        relative_global = global_reduction / max(total_variance, _EPS)
        repeat_count = sum(1 for name in recent if name == skill)
        question_cost = 1.0 + 0.15 * repeat_count
        utility = (0.70 * relative_decision + 0.30 * relative_global) / question_cost
        return {
            "utility": utility,
            "decision_reduction": decision_reduction,
            "global_reduction": global_reduction,
            "relative_decision_reduction": relative_decision,
            "relative_global_reduction": relative_global,
            "difficulty_efficiency": efficiency,
            "observation_std": math.sqrt(observation_variance),
            "question_cost": question_cost,
        }


# =============================================================================
# Selection strategies
# =============================================================================

def select_skill(
    strategy: str,
    model: BayesianAbilityModel,
    recent: Sequence[str],
    turn: int,
    seed: int,
    candidate_id: int,
) -> Tuple[str, str, Dict[str, float]]:
    if strategy == "random":
        rng = random.Random(stable_seed("policy", seed, strategy, candidate_id, turn))
        skill = rng.choice(SKILLS)
        difficulty = closest_difficulty(model.skill_mean(skill))
        return skill, difficulty, model.risk_components(skill, difficulty, recent)

    if strategy == "fixed":
        skill = FIXED_ORDER[turn % len(FIXED_ORDER)]
        difficulty = closest_difficulty(model.skill_mean(skill))
        return skill, difficulty, model.risk_components(skill, difficulty, recent)

    scored = []
    # These denominators are common to every candidate skill at this turn.
    # Computing them once avoids an unnecessary O(|S|^3) selector loop.
    current_job_variance = model.job_variance()
    current_total_variance = model.total_variance()
    for skill in SKILLS:
        difficulty = closest_difficulty(model.skill_mean(skill))
        components = model.risk_components(
            skill,
            difficulty,
            recent,
            job_variance=current_job_variance,
            total_variance=current_total_variance,
        )
        if strategy == "uncertainty":
            noise_variance = components["observation_std"] ** 2
            variance = model.skill_variance(skill)
            score = (variance ** 2 / (variance + noise_variance)) / max(variance, _EPS)
            score /= components["question_cost"]
        elif strategy == "bridge":
            score = components["utility"]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        scored.append((score, skill, difficulty, components))
    scored.sort(key=lambda item: (-item[0], item[1]))
    _, skill, difficulty, components = scored[0]
    return skill, difficulty, components


# =============================================================================
# Experiment 1: estimator accuracy and convergence
# =============================================================================

def run_exp1(output_root: Path, n_candidates: int, seed: int) -> Dict:
    output_dir = output_root / "exp1_estimator"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = generate_profiles(n_candidates, seed)
    atomic_json(output_dir / "profiles.json", [asdict(profile) for profile in profiles])

    online_true: Dict[str, List[float]] = defaultdict(list)
    online_pred: Dict[str, List[float]] = defaultdict(list)
    final_rows = []
    turn_rows = []

    for profile in profiles:
        bayesian = BayesianAbilityModel(graph_enabled=False)
        running_sum = defaultdict(float)
        counts = defaultdict(int)
        ewa = {}
        last = {}
        observation_count = defaultdict(int)

        for skill in EXP1_SKILLS:
            for repetition in range(3):
                difficulty = closest_difficulty(bayesian.skill_mean(skill))
                index = observation_count[skill]
                score, observation_std = simulate_score(
                    profile, skill, difficulty, index, seed
                )
                observation_count[skill] += 1

                last[skill] = score
                counts[skill] += 1
                running_sum[skill] += score
                running = running_sum[skill] / counts[skill]
                ewa[skill] = score if skill not in ewa else 0.3 * score + 0.7 * ewa[skill]
                bayesian.update(skill, score, observation_std=observation_std)

                truth = profile.skill_theta[skill]
                predictions = {
                    "single_eval": last[skill],
                    "running_mean": running,
                    "ewa": ewa[skill],
                    "bayesian": bayesian.skill_mean(skill),
                }
                for method, prediction in predictions.items():
                    online_true[method].append(truth)
                    online_pred[method].append(prediction)

                turn_rows.append({
                    "candidate_id": profile.candidate_id,
                    "level": profile.level,
                    "skill": skill,
                    "repetition": repetition + 1,
                    "difficulty": difficulty,
                    "true_theta": round(truth, 6),
                    "observed_score": round(score, 6),
                    "observation_std": round(observation_std, 6),
                    **{name: round(value, 6) for name, value in predictions.items()},
                    "bayesian_std": round(math.sqrt(bayesian.skill_variance(skill)), 6),
                })

        for skill in EXP1_SKILLS:
            final_rows.append({
                "candidate_id": profile.candidate_id,
                "level": profile.level,
                "skill": skill,
                "true_theta": profile.skill_theta[skill],
                "single_eval": last[skill],
                "running_mean": running_sum[skill] / counts[skill],
                "ewa": ewa[skill],
                "bayesian": bayesian.skill_mean(skill),
                "bayesian_std": math.sqrt(bayesian.skill_variance(skill)),
            })

    expected = n_candidates * len(EXP1_SKILLS) * 3
    if len(turn_rows) != expected:
        raise AssertionError(f"Expected {expected} observations, got {len(turn_rows)}")

    methods = ["single_eval", "running_mean", "ewa", "bayesian"]
    metrics = {
        "online": {
            method: regression_metrics(online_true[method], online_pred[method])
            for method in methods
        },
        "final_after_three": {
            method: regression_metrics(
                [row["true_theta"] for row in final_rows],
                [row[method] for row in final_rows],
            )
            for method in methods
        },
        "counts": {
            "candidates": n_candidates,
            "online_observations": len(turn_rows),
            "final_candidate_skill_units": len(final_rows),
        },
    }
    write_csv(output_dir / "turns.csv", turn_rows)
    write_csv(output_dir / "final_candidate_skill.csv", final_rows)
    atomic_json(output_dir / "metrics.json", metrics)
    return metrics


# =============================================================================
# Experiment 2: selection strategy comparison
# =============================================================================

def decision_accuracy(truth: Sequence[float], prediction: Sequence[float], threshold: float = 6.5) -> float:
    correct = sum(
        (actual >= threshold) == (estimated >= threshold)
        for actual, estimated in zip(truth, prediction)
    )
    return correct / len(truth)


def strategy_for_candidate(
    profile: CandidateProfile,
    strategy: str,
    questions: int,
    seed: int,
) -> Dict:
    model = BayesianAbilityModel(graph_enabled=True)
    recent: List[str] = []
    per_skill_count = defaultdict(int)
    trace = []
    for turn in range(questions):
        skill, difficulty, components = select_skill(
            strategy, model, recent, turn, seed, profile.candidate_id
        )
        score, observation_std = simulate_score(
            profile,
            skill,
            difficulty,
            per_skill_count[skill],
            seed,
        )
        per_skill_count[skill] += 1
        model.update(skill, score, observation_std=observation_std)
        recent = (recent + [skill])[-6:]
        trace.append({
            "turn": turn + 1,
            "skill": skill,
            "cluster": SKILL_TO_CLUSTER[skill],
            "difficulty": difficulty,
            "score": score,
            "posterior_job_mean": model.job_mean(),
            "posterior_job_std": math.sqrt(model.job_variance()),
            **components,
        })

    observed_skills = {row["skill"] for row in trace}
    observed_clusters = {row["cluster"] for row in trace}
    truth = true_job_score(profile)
    prediction = model.job_mean()
    return {
        "candidate_id": profile.candidate_id,
        "level": profile.level,
        "strategy": strategy,
        "true_job_score": truth,
        "estimated_job_score": prediction,
        "absolute_error": abs(prediction - truth),
        "skill_coverage": len(observed_skills) / len(SKILLS),
        "cluster_coverage": len(observed_clusters) / len(CLUSTERS),
        "posterior_job_std": math.sqrt(model.job_variance()),
        "trace": trace,
    }


def paired_sign_flip_p(differences: Sequence[float], samples: int = 20000, seed: int = 991) -> float:
    """Two-sided paired randomization test on the mean difference."""
    if not differences:
        return float("nan")
    observed = abs(sum(differences) / len(differences))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(samples):
        value = abs(sum((1 if rng.random() < 0.5 else -1) * d for d in differences) / len(differences))
        if value >= observed - _EPS:
            exceed += 1
    return (exceed + 1) / (samples + 1)


def run_exp2(
    output_root: Path,
    seeds: int,
    n_candidates: int,
    questions: int,
    strategies: Sequence[str],
    base_seed: int,
) -> Dict:
    output_dir = output_root / "exp2_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = []
    all_candidate_rows = []

    for offset in range(seeds):
        seed = base_seed + offset
        profiles = generate_profiles(n_candidates, seed)
        for strategy in strategies:
            results = [
                strategy_for_candidate(profile, strategy, questions, seed)
                for profile in profiles
            ]
            truth = [row["true_job_score"] for row in results]
            prediction = [row["estimated_job_score"] for row in results]
            metrics = regression_metrics(truth, prediction)
            metrics.update({
                "seed": seed,
                "strategy": strategy,
                "decision_accuracy": decision_accuracy(truth, prediction),
                "skill_coverage": sum(row["skill_coverage"] for row in results) / len(results),
                "cluster_coverage": sum(row["cluster_coverage"] for row in results) / len(results),
                "posterior_job_std": sum(row["posterior_job_std"] for row in results) / len(results),
            })
            seed_metrics.append(metrics)
            for row in results:
                compact = {key: value for key, value in row.items() if key != "trace"}
                compact["seed"] = seed
                all_candidate_rows.append(compact)

    summary = {}
    for strategy in strategies:
        rows = [row for row in seed_metrics if row["strategy"] == strategy]
        summary[strategy] = {
            metric: mean_std([float(row[metric]) for row in rows])
            for metric in (
                "mae", "rmse", "pearson", "spearman", "decision_accuracy",
                "skill_coverage", "cluster_coverage", "posterior_job_std",
            )
        }

    comparisons = {}
    if "bridge" in strategies:
        bridge_by_seed = {
            row["seed"]: row["mae"]
            for row in seed_metrics if row["strategy"] == "bridge"
        }
        for baseline in strategies:
            if baseline == "bridge":
                continue
            baseline_by_seed = {
                row["seed"]: row["mae"]
                for row in seed_metrics if row["strategy"] == baseline
            }
            common = sorted(set(bridge_by_seed) & set(baseline_by_seed))
            differences = [baseline_by_seed[s] - bridge_by_seed[s] for s in common]
            comparisons[f"bridge_vs_{baseline}"] = {
                "positive_means_bridge_lower_mae": True,
                "mean_mae_improvement": sum(differences) / len(differences),
                "paired_sign_flip_p": paired_sign_flip_p(
                    differences,
                    seed=stable_seed("sign-flip", baseline, base_seed),
                ),
            }

    result = {
        "setup": {
            "seeds": seeds,
            "candidates_per_seed": n_candidates,
            "questions": questions,
            "strategies": list(strategies),
        },
        "summary": summary,
        "comparisons": comparisons,
    }
    write_csv(output_dir / "seed_metrics.csv", seed_metrics)
    write_csv(output_dir / "candidate_results.csv", all_candidate_rows)
    atomic_json(output_dir / "summary.json", result)
    return result


# =============================================================================
# Optional Qwen end-to-end pilot
# =============================================================================

class QwenClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 90.0,
        retries: int = 4,
    ) -> None:
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.retries = retries

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
                if attempt + 1 == self.retries:
                    raise RuntimeError(f"Qwen request failed after {self.retries} attempts") from exc
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def ability_band(theta: float) -> str:
    if theta < 3.5:
        return "只掌握少量基础概念；回答应自然暴露遗漏或误解，不要假装精通"
    if theta < 5.5:
        return "掌握基础概念但生产经验有限；能够回答常见问题，复杂权衡不完整"
    if theta < 7.5:
        return "具备可靠项目经验；能够解释原理和常见权衡，但极端场景可能遗漏"
    return "具备深入生产经验；能够讨论原理、权衡、故障场景和工程边界"


QUESTION_QUALITY_DIMENSIONS = (
    "skill_relevance",
    "difficulty_match",
    "clarity",
    "non_redundancy",
    "contextual_coherence",
)


def generate_qwen_question(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    rejected_attempts: Sequence[Mapping],
) -> str:
    """Generate one interview question; the returned value is the raw model text."""
    history = list(dialogue_history)[-3:]
    history_text = "\n".join(
        f"第{row['turn'] + 1}轮：问题：{row['question']} 回答：{row['answer']}"
        for row in history
    ) or "无，这是第一轮。"
    rejected_text = "\n".join(
        f"- 问题：{row['question']}；评审反馈：{row['feedback']}"
        for row in rejected_attempts
    ) or "无"
    difficulty_guidance = {
        "easy": "考查核心概念与常见场景，避免复杂系统设计",
        "medium": "考查项目应用、故障分析和主要工程权衡",
        "hard": "考查复杂系统设计、边界条件、多重权衡与故障恢复",
    }[difficulty]
    system = (
        "你是技术面试问题生成器。只生成一个中文问题，不要给答案、解释、标题、"
        "评分或Markdown。问题必须聚焦指定技能，符合难度，并与对话自然衔接。"
    )
    user = (
        f"目标技能：{skill}\n目标难度：{difficulty}（{difficulty_guidance}）\n"
        f"最近对话：\n{history_text}\n"
        f"本轮已被拒绝的尝试（必须针对反馈改写且避免重复）：\n{rejected_text}\n"
        "请输出新的单一面试问题。"
    )
    return client.chat(model, system, user, temperature=0.8, max_tokens=300)


def parse_question_quality_json(raw: str) -> Dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Question judge output does not contain a JSON object")
    data = json.loads(cleaned[start:end + 1])
    parsed = {}
    for name in (*QUESTION_QUALITY_DIMENSIONS, "overall_question_quality"):
        value = float(data[name])
        if not 1.0 <= value <= 10.0:
            raise ValueError(f"Question quality score outside [1,10] for {name}: {value}")
        parsed[name] = value
    parsed["reason"] = str(data.get("reason", ""))[:500]
    return parsed


def evaluate_qwen_question(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    question: str,
    dialogue_history: Sequence[Mapping],
    repeats: int,
) -> Tuple[Dict[str, float], List[Dict]]:
    prior_dialogue = [
        {
            "skill": row["skill"],
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in dialogue_history[-6:]
    ]
    system = (
        "你是严格的高级技术面试问题评审器。分别按1到10分评价："
        "skill_relevance（技能相关性）、difficulty_match（难度匹配）、"
        "clarity（清晰且可回答）、non_redundancy（相对历史问题不重复）、"
        "contextual_coherence（与上一轮对话连贯），并给出overall_question_quality。"
        "只输出JSON，包含上述六个数字字段及reason字符串。"
    )
    user = (
        f"目标技能：{skill}\n目标难度：{difficulty}\n候选问题：{question}\n"
        f"此前对话：{json.dumps(prior_dialogue, ensure_ascii=False)}"
    )
    details = []
    for _ in range(repeats):
        raw = client.chat(model, system, user, temperature=0.2, max_tokens=400)
        parsed = parse_question_quality_json(raw)
        parsed["raw"] = raw
        details.append(parsed)
    score_names = (*QUESTION_QUALITY_DIMENSIONS, "overall_question_quality")
    means = {
        name: sum(row[name] for row in details) / len(details)
        for name in score_names
    }
    return means, details


def generate_qualified_question(
    client: QwenClient,
    question_model: str,
    question_judge_model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    judge_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
) -> Tuple[str, Dict[str, float], int, List[Dict]]:
    attempts = []
    rejected = []
    for attempt_index in range(max_regenerations + 1):
        raw_question = generate_qwen_question(
            client, question_model, skill, difficulty, dialogue_history, rejected
        )
        question = raw_question.strip()
        quality_scores, judge_details = evaluate_qwen_question(
            client,
            question_judge_model,
            skill,
            difficulty,
            question,
            dialogue_history,
            judge_repeats,
        )
        attempts.append({
            "attempt": attempt_index,
            "question": question,
            "question_model_raw_output": raw_question,
            "quality_scores": quality_scores,
            "question_judge_details": judge_details,
        })
        if quality_scores["overall_question_quality"] >= quality_threshold:
            break
        rejected.append({
            "question": question,
            "feedback": "; ".join(row["reason"] for row in judge_details if row["reason"]),
        })
    accepted = attempts[-1]
    return (
        accepted["question"],
        accepted["quality_scores"],
        len(attempts) - 1,
        attempts,
    )


def generate_qwen_answer(
    client: QwenClient,
    model: str,
    profile: CandidateProfile,
    skill: str,
    question: str,
) -> str:
    system = (
        "你正在模拟技术面试候选人，而不是真实人类。请严格服从给定能力边界，"
        "不要因为你本身知道答案就表现得比该角色更强。回答使用自然中文口语，"
        "长度120至250字，不输出Markdown。\n"
        f"候选人总体档位：{LEVEL_ZH[profile.level]}。\n"
        f"当前技能 {skill} 的能力边界：{ability_band(profile.skill_theta[skill])}。"
    )
    return client.chat(model, system, question, temperature=0.8, max_tokens=450)


def parse_score_json(raw: str) -> Dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Judge output does not contain a JSON object")
    data = json.loads(cleaned[start:end + 1])
    score = float(data["score"])
    if not 1.0 <= score <= 10.0:
        raise ValueError(f"Judge score outside [1,10]: {score}")
    return {"score": score, "reason": str(data.get("reason", ""))[:300]}


def evaluate_qwen_answer(
    client: QwenClient,
    model: str,
    skill: str,
    question: str,
    answer: str,
    repeats: int,
) -> Tuple[float, float, List[Dict]]:
    system = (
        "你是严格的技术面试评分器。只评价回答中实际出现的内容，不根据候选人身份推断。"
        "按准确性35%、深度30%、实践性20%、清晰度15%给出1到10分。"
        "只输出JSON：{\"score\": 1到10的数字, \"reason\": \"简短理由\"}。"
    )
    user = f"技能：{skill}\n问题：{question}\n回答：{answer}"
    details = []
    for _ in range(repeats):
        raw = client.chat(model, system, user, temperature=0.2, max_tokens=240)
        parsed = parse_score_json(raw)
        parsed["raw"] = raw
        details.append(parsed)
    scores = [row["score"] for row in details]
    score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return sum(scores) / len(scores), score_std, details


def run_qwen(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    judge_model: str,
    n_candidates: int,
    questions: int,
    strategies: Sequence[str],
    question_judge_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    judge_repeats: int,
    seed: int,
    request_delay: float,
) -> Dict:
    output_dir = output_root / "exp3_qwen"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.json"
    turns_path = output_dir / "turns.jsonl"
    setup_path = output_dir / "setup.json"
    profiles = generate_profiles(n_candidates, seed)

    setup = {
        "schema_version": 2,
        "seed": seed,
        "question_model": question_model,
        "question_judge_model": question_judge_model,
        "candidate_model": candidate_model,
        "judge_model": judge_model,
        "question_judge_repeats": question_judge_repeats,
        "quality_threshold": quality_threshold,
        "max_regenerations": max_regenerations,
        "judge_repeats": judge_repeats,
        "candidates": n_candidates,
        "questions": questions,
        "strategies": list(strategies),
    }
    if setup_path.exists():
        saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if saved_setup != setup:
            raise ValueError(
                "Existing exp3_qwen/setup.json does not match this run; "
                "use the original arguments or a different --output directory"
            )
    else:
        atomic_json(setup_path, setup)

    if profiles_path.exists():
        saved = json.loads(profiles_path.read_text(encoding="utf-8"))
        if saved != [asdict(profile) for profile in profiles]:
            raise ValueError("Existing profiles.json does not match the requested seed/setup")
    else:
        atomic_json(profiles_path, [asdict(profile) for profile in profiles])

    existing = load_jsonl(turns_path)
    grouped = defaultdict(list)
    seen_keys = set()
    for row in existing:
        if "question_quality_scores" not in row:
            raise ValueError(
                "Existing turns.jsonl uses the old fixed-question schema; "
                "use a different --output directory for the new experiment"
            )
        key = (row["strategy"], row["candidate_id"], row["turn"])
        if key in seen_keys:
            raise ValueError(f"Duplicate Qwen record: {key}")
        seen_keys.add(key)
        grouped[(row["strategy"], row["candidate_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["turn"])

    client = QwenClient(api_key, base_url)
    total = n_candidates * len(strategies) * questions
    with tqdm(total=total, initial=len(existing), desc="Qwen interview turns") as progress:
        for strategy in strategies:
            for profile in profiles:
                model = BayesianAbilityModel(graph_enabled=True)
                recent = []
                completed = grouped[(strategy, profile.candidate_id)]
                if len(completed) > questions or [row["turn"] for row in completed] != list(range(len(completed))):
                    raise ValueError(
                        f"Non-contiguous or excessive saved turns for {strategy}/{profile.candidate_id}"
                    )
                for row in completed:
                    # Only the answer judge's ability score enters this posterior.
                    model.update(
                        row["skill"],
                        row["judge_score_mean"],
                        observation_std=max(row["judge_score_std"], 0.35),
                    )
                    recent = (recent + [row["skill"]])[-6:]

                for turn in range(len(completed), questions):
                    skill, difficulty, components = select_skill(
                        strategy, model, recent, turn, seed, profile.candidate_id
                    )
                    question, quality_scores, regeneration_count, question_attempts = (
                        generate_qualified_question(
                            client=client,
                            question_model=question_model,
                            question_judge_model=question_judge_model,
                            skill=skill,
                            difficulty=difficulty,
                            dialogue_history=grouped[(strategy, profile.candidate_id)],
                            judge_repeats=question_judge_repeats,
                            quality_threshold=quality_threshold,
                            max_regenerations=max_regenerations,
                        )
                    )
                    answer = generate_qwen_answer(
                        client, candidate_model, profile, skill, question
                    )
                    score_mean, score_std, judge_details = evaluate_qwen_answer(
                        client, judge_model, skill, question, answer, judge_repeats
                    )
                    # A non-zero floor prevents a deterministic answer judge call
                    # from creating unjustified near-zero posterior uncertainty.
                    update_std = max(score_std, 0.35)
                    model.update(skill, score_mean, observation_std=update_std)
                    recent = (recent + [skill])[-6:]
                    record = {
                        "created_at": utc_now(),
                        "strategy": strategy,
                        "candidate_id": profile.candidate_id,
                        "level": profile.level,
                        "turn": turn,
                        "skill": skill,
                        "cluster": SKILL_TO_CLUSTER[skill],
                        "difficulty": difficulty,
                        "question": question,
                        "question_quality_scores": {
                            name: quality_scores[name]
                            for name in QUESTION_QUALITY_DIMENSIONS
                        },
                        "overall_question_quality": quality_scores["overall_question_quality"],
                        "quality_threshold": quality_threshold,
                        "quality_threshold_met": (
                            quality_scores["overall_question_quality"] >= quality_threshold
                        ),
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": question_attempts,
                        "answer": answer,
                        "assigned_skill_theta": profile.skill_theta[skill],
                        "judge_score_mean": score_mean,
                        "judge_score_std": score_std,
                        "judge_details": judge_details,
                        "posterior_skill_mean": model.skill_mean(skill),
                        "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                        "posterior_job_mean": model.job_mean(),
                        "posterior_job_std": math.sqrt(model.job_variance()),
                        "selector": components,
                    }
                    append_jsonl(turns_path, record)
                    grouped[(strategy, profile.candidate_id)].append(record)
                    progress.update(1)
                    time.sleep(max(request_delay, 0.0))

    candidate_rows = []
    for strategy in strategies:
        for profile in profiles:
            rows = sorted(
                grouped[(strategy, profile.candidate_id)],
                key=lambda row: row["turn"],
            )
            if len(rows) != questions:
                raise AssertionError(
                    f"Incomplete Qwen run for {strategy}/{profile.candidate_id}: {len(rows)}"
                )
            last = rows[-1]
            candidate_rows.append({
                "strategy": strategy,
                "candidate_id": profile.candidate_id,
                "level": profile.level,
                "true_job_score": true_job_score(profile),
                "estimated_job_score": last["posterior_job_mean"],
                "posterior_job_std": last["posterior_job_std"],
                "skill_coverage": len({row["skill"] for row in rows}) / len(SKILLS),
                "cluster_coverage": len({row["cluster"] for row in rows}) / len(CLUSTERS),
            })

    summary = {}
    for strategy in strategies:
        rows = [row for row in candidate_rows if row["strategy"] == strategy]
        truth = [row["true_job_score"] for row in rows]
        prediction = [row["estimated_job_score"] for row in rows]
        summary[strategy] = {
            **regression_metrics(truth, prediction),
            "decision_accuracy": decision_accuracy(truth, prediction),
            "skill_coverage": sum(row["skill_coverage"] for row in rows) / len(rows),
            "cluster_coverage": sum(row["cluster_coverage"] for row in rows) / len(rows),
        }
    result = {
        "warning": (
            "Candidates are Qwen simulations, not humans. Assigned abilities are "
            "controlled latent profiles, not verified human ground truth."
        ),
        "setup": {
            "question_model": question_model,
            "question_judge_model": question_judge_model,
            "question_judge_repeats": question_judge_repeats,
            "quality_threshold": quality_threshold,
            "max_regenerations": max_regenerations,
            "candidate_model": candidate_model,
            "judge_model": judge_model,
            "judge_repeats": judge_repeats,
            "candidates": n_candidates,
            "questions": questions,
            "strategies": list(strategies),
        },
        "summary": summary,
    }
    write_csv(output_dir / "candidate_results.csv", candidate_rows)
    atomic_json(output_dir / "summary.json", result)
    return result


# =============================================================================
# CLI
# =============================================================================

def print_compact(title: str, result: Dict) -> None:
    print(f"\n{title}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["exp1", "exp2", "qwen", "all"], default="all")
    parser.add_argument("--output", default="runs", help="Output root directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=30, help="Synthetic candidates per seed")
    parser.add_argument("--seeds", type=int, default=20, help="Number of Exp. 2 seeds")
    parser.add_argument("--questions", type=int, default=15, help="Questions per Exp. 2 interview")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["bridge", "uncertainty", "random", "fixed"],
        default=["bridge", "uncertainty", "random", "fixed"],
    )

    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--question-model", default="qwen-turbo")
    parser.add_argument("--question-judge-model", default="qwen-max")
    parser.add_argument("--question-judge-repeats", type=int, default=3)
    parser.add_argument("--quality-threshold", type=float, default=7.0)
    parser.add_argument("--max-regenerations", type=int, default=2)
    parser.add_argument("--candidate-model", default="qwen-max")
    parser.add_argument("--judge-model", default="qwen-max")
    parser.add_argument("--qwen-candidates", type=int, default=30)
    parser.add_argument("--qwen-questions", type=int, default=12)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument("--confirm-api-calls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "command": sys.argv,
        "scale": "[1,10]",
        "skill_clusters": {name: list(skills) for name, skills in SKILL_CLUSTERS.items()},
        "note": "--mode all runs only synthetic Exp. 1 and Exp. 2; Qwen is explicit.",
    }
    atomic_json(output_root / "manifest.json", manifest)

    if args.mode in ("exp1", "all"):
        result = run_exp1(output_root, args.candidates, args.seed)
        print_compact("Exp. 1 complete", result)

    if args.mode in ("exp2", "all"):
        result = run_exp2(
            output_root=output_root,
            seeds=args.seeds,
            n_candidates=args.candidates,
            questions=args.questions,
            strategies=args.strategies,
            base_seed=args.seed,
        )
        print_compact("Exp. 2 complete", result)

    if args.mode == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise SystemExit(
                "Qwen mode requires DASHSCOPE_API_KEY in the environment"
            )
        if args.question_judge_repeats < 1 or args.judge_repeats < 1:
            raise SystemExit("Judge repeat counts must be at least one")
        if not 1.0 <= args.quality_threshold <= 10.0:
            raise SystemExit("--quality-threshold must be within [1,10]")
        if args.max_regenerations < 0:
            raise SystemExit("--max-regenerations must be non-negative")
        calls_per_turn = (
            (args.max_regenerations + 1) * (1 + args.question_judge_repeats)
            + 1
            + args.judge_repeats
        )
        estimated_calls = (
            args.qwen_candidates
            * len(args.strategies)
            * args.qwen_questions
            * calls_per_turn
        )
        print(f"Estimated maximum API calls: {estimated_calls}")
        if not args.confirm_api_calls:
            raise SystemExit("Re-run with --confirm-api-calls after checking the estimated cost")
        result = run_qwen(
            output_root=output_root,
            api_key=api_key,
            base_url=args.base_url,
            question_model=args.question_model,
            question_judge_model=args.question_judge_model,
            candidate_model=args.candidate_model,
            judge_model=args.judge_model,
            n_candidates=args.qwen_candidates,
            questions=args.qwen_questions,
            strategies=args.strategies,
            question_judge_repeats=args.question_judge_repeats,
            quality_threshold=args.quality_threshold,
            max_regenerations=args.max_regenerations,
            judge_repeats=args.judge_repeats,
            seed=args.seed,
            request_delay=args.request_delay,
        )
        print_compact("Qwen pilot complete", result)


if __name__ == "__main__":
    main()
