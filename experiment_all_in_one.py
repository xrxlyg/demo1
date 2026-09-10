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
  python experiment_all_in_one.py --mode analyze --output runs

Qwen calls are never included in --mode all.  They require an explicit mode:
  export DASHSCOPE_API_KEY=...
  python experiment_all_in_one.py --mode qwen \
      --question-model qwen-turbo --question-judge-model qwen-max \
      --candidate-model qwen-max \
      --qwen-candidates 30 --qwen-questions 12 \
      --strategies bridge random --question-judge-repeats 3 \
      --quality-threshold 7 \
      --confirm-api-calls --output runs

Question-quality dimensions and simulated abilities use the [1, 10] scale.
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
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is empty")
        try:
            self.api_key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "DASHSCOPE_API_KEY must be the real ASCII API key; "
                "do not use the Chinese placeholder '你的新API Key'"
            ) from exc
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
    "technical_correctness",
    "difficulty_match",
    "clarity",
    "non_redundancy",
    "contextual_coherence",
    "diagnostic_value",
    "adaptive_relevance",
)


def build_question_context(
    strategy: str,
    ability_model: BayesianAbilityModel,
    skill: str,
    selector: Mapping[str, float],
) -> Dict:
    posterior_mean = ability_model.skill_mean(skill)
    posterior_std = math.sqrt(ability_model.skill_variance(skill))
    if posterior_mean < 4.0:
        boundary = "区分概念缺失、常见误解和基本掌握"
    elif posterior_mean < 7.0:
        boundary = "区分仅会解释概念与能够进行实际工程权衡"
    else:
        boundary = "区分熟练应用与能够处理边界条件、故障和复杂权衡"
    return {
        "skill": skill,
        "cluster": SKILL_TO_CLUSTER[skill],
        "related_skills": [
            name for name in SKILL_CLUSTERS[SKILL_TO_CLUSTER[skill]] if name != skill
        ],
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "decision_variance_reduction": selector["decision_reduction"],
        "global_variance_reduction": selector["global_reduction"],
        "selection_utility": selector["utility"],
        "diagnostic_goal": boundary,
        "prompt_variant": "bridge_adaptive" if strategy == "bridge" else "baseline",
    }


def generate_qwen_question(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    rejected_attempts: Sequence[Mapping],
    question_context: Mapping,
) -> str:
    """Generate one interview question; the returned value is the raw model text."""
    history = list(dialogue_history)[-3:]
    history_text = "\n".join(
        f"第{row['turn'] + 1}轮：问题：{row['question']} 回答：{row['answer']}"
        for row in history
    ) or "无，这是第一轮。"
    previous_questions = [row["question"] for row in dialogue_history]
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
    user_parts = [
        f"目标技能：{skill}\n目标难度：{difficulty}（{difficulty_guidance}）\n"
        f"最近对话：\n{history_text}\n"
        f"全部历史问题：{json.dumps(previous_questions, ensure_ascii=False)}\n"
    ]
    if question_context["prompt_variant"] == "bridge_adaptive":
        user_parts.append(
            "BRIDGE诊断上下文：\n"
            f"- 技能簇：{question_context['cluster']}\n"
            f"- 相关技能：{', '.join(question_context['related_skills'])}\n"
            f"- 当前能力后验均值：{question_context['posterior_mean']:.3f}/10\n"
            f"- 当前能力后验标准差：{question_context['posterior_std']:.3f}\n"
            f"- 选择效用：{question_context['selection_utility']:.6f}\n"
            f"- 诊断目标：{question_context['diagnostic_goal']}\n"
            "请围绕当前能力边界设计具有区分度的追问。\n"
        )
    user_parts.append(
        f"本轮已被拒绝的尝试（必须针对反馈改写且避免重复）：\n{rejected_text}\n"
        "请输出新的单一面试问题。"
    )
    return client.chat(model, system, "".join(user_parts), temperature=0.8, max_tokens=300)


def parse_question_quality_json(raw: str) -> Dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Question judge output does not contain a JSON object")
    data = json.loads(cleaned[start:end + 1])
    parsed = {}
    for name in QUESTION_QUALITY_DIMENSIONS:
        value = float(data[name])
        if not 1.0 <= value <= 10.0:
            raise ValueError(f"Question quality score outside [1,10] for {name}: {value}")
        parsed[name] = value
    parsed["reason"] = str(data.get("reason", ""))[:500]
    return parsed


def request_judge_json(
    client: QwenClient,
    model: str,
    system: str,
    user: str,
    parser,
    max_format_attempts: int = 3,
    max_tokens: int = 400,
) -> Dict:
    """Retry judge calls whose content is not valid JSON and retain bad outputs."""
    invalid_outputs = []
    current_user = user
    for format_attempt in range(max_format_attempts):
        raw = client.chat(model, system, current_user, temperature=0.2, max_tokens=max_tokens)
        try:
            parsed = parser(raw)
        except (ValueError, KeyError, TypeError) as exc:
            invalid_outputs.append({
                "raw": raw,
                "error": f"{type(exc).__name__}: {exc}",
            })
            if format_attempt + 1 == max_format_attempts:
                raise ValueError(
                    f"Judge failed to return valid JSON after {max_format_attempts} attempts; "
                    f"last output: {raw[:500]!r}"
                ) from exc
            current_user = (
                f"{user}\n\n你上一次的输出无法解析：{raw[:1000]}\n"
                "请重新评分。只输出严格JSON对象，不要Markdown代码块或任何额外文字。"
            )
            continue
        parsed["raw"] = raw
        parsed["format_retry_count"] = format_attempt
        parsed["invalid_raw_outputs"] = invalid_outputs
        return parsed
    raise RuntimeError("unreachable")


def evaluate_qwen_question(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    question: str,
    dialogue_history: Sequence[Mapping],
    repeats: int,
    question_context: Mapping,
) -> Tuple[Dict[str, float], List[Dict]]:
    blind_context = {
        key: value for key, value in question_context.items()
        if key != "prompt_variant"
    }
    prior_dialogue = [
        {
            "skill": row["skill"],
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in dialogue_history
    ]
    system = (
        "你是严格的高级技术面试问题评审器。按以下锚点评分：1-2严重不合格，"
        "3-4有明显问题，5-6基本可用但需修改，7-8合格且良好，9-10高质量。"
        "分别评价skill_relevance（技能相关性）、technical_correctness（技术事实正确）、"
        "difficulty_match（难度匹配）、clarity（清晰且可回答）、"
        "non_redundancy（相对全部历史问题不重复）、"
        "contextual_coherence（与上一轮对话连贯）、"
        "diagnostic_value（能区分不同能力水平）、"
        "adaptive_relevance（针对当前后验能力边界）。"
        "只输出JSON，包含上述八个数字字段及reason字符串；不要输出综合分。"
    )
    user = (
        f"目标技能：{skill}\n目标难度：{difficulty}\n候选问题：{question}\n"
        f"当前诊断状态：{json.dumps(blind_context, ensure_ascii=False)}\n"
        f"此前对话：{json.dumps(prior_dialogue, ensure_ascii=False)}"
    )
    details = []
    for _ in range(repeats):
        details.append(request_judge_json(
            client=client,
            model=model,
            system=system,
            user=user,
            parser=parse_question_quality_json,
            max_tokens=400,
        ))
    means = {
        name: sum(row[name] for row in details) / len(details)
        for name in QUESTION_QUALITY_DIMENSIONS
    }
    means["overall_question_quality"] = sum(
        means[name] for name in QUESTION_QUALITY_DIMENSIONS
    ) / len(QUESTION_QUALITY_DIMENSIONS)
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
    regenerate_low_quality: bool,
    question_context: Mapping,
) -> Tuple[str, Dict[str, float], int, List[Dict]]:
    attempts = []
    rejected = []
    maximum_attempts = max_regenerations + 1 if regenerate_low_quality else 1
    for attempt_index in range(maximum_attempts):
        raw_question = generate_qwen_question(
            client,
            question_model,
            skill,
            difficulty,
            dialogue_history,
            rejected,
            question_context,
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
            question_context,
        )
        attempts.append({
            "attempt": attempt_index,
            "question": question,
            "question_model_raw_output": raw_question,
            "quality_scores": quality_scores,
            "question_judge_details": judge_details,
        })
        if (
            not regenerate_low_quality
            or quality_scores["overall_question_quality"] >= quality_threshold
        ):
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


def summarize_question_records(rows: Sequence[Mapping]) -> Dict:
    summary = {"n_questions": len(rows)}
    for name in (*QUESTION_QUALITY_DIMENSIONS, "overall_question_quality"):
        values = [float(row[name]) for row in rows] if name == "overall_question_quality" else [
            float(row["question_quality_scores"][name]) for row in rows
        ]
        summary[f"{name}_mean"] = sum(values) / len(values)
        summary[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    summary.update({
        "below_threshold_rate": sum(
            not bool(row["quality_threshold_met"]) for row in rows
        ) / len(rows),
        "average_regeneration_count": sum(
            int(row["regeneration_count"]) for row in rows
        ) / len(rows),
        "average_realized_job_variance_reduction": sum(
            float(row["realized_job_variance_reduction"]) for row in rows
        ) / len(rows),
        "total_realized_job_variance_reduction": sum(
            float(row["realized_job_variance_reduction"]) for row in rows
        ),
    })
    return summary


def run_qwen(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    n_candidates: int,
    questions: int,
    strategies: Sequence[str],
    question_judge_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    regenerate_low_quality: bool,
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
        "schema_version": 4,
        "seed": seed,
        "question_model": question_model,
        "question_judge_model": question_judge_model,
        "candidate_model": candidate_model,
        "question_judge_repeats": question_judge_repeats,
        "quality_threshold": quality_threshold,
        "max_regenerations": max_regenerations,
        "regenerate_low_quality": regenerate_low_quality,
        "ability_update_source": "deterministic_latent_simulation",
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
                per_skill_count = defaultdict(int)
                completed = grouped[(strategy, profile.candidate_id)]
                if len(completed) > questions or [row["turn"] for row in completed] != list(range(len(completed))):
                    raise ValueError(
                        f"Non-contiguous or excessive saved turns for {strategy}/{profile.candidate_id}"
                    )
                for row in completed:
                    model.update(
                        row["skill"],
                        row["ability_observation_score"],
                        observation_std=row["ability_observation_std"],
                    )
                    recent = (recent + [row["skill"]])[-6:]
                    per_skill_count[row["skill"]] += 1

                for turn in range(len(completed), questions):
                    skill, difficulty, components = select_skill(
                        strategy, model, recent, turn, seed, profile.candidate_id
                    )
                    question_context = build_question_context(
                        strategy, model, skill, components
                    )
                    job_variance_before = model.job_variance()
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
                            regenerate_low_quality=regenerate_low_quality,
                            question_context=question_context,
                        )
                    )
                    answer = generate_qwen_answer(
                        client, candidate_model, profile, skill, question
                    )
                    ability_score, ability_observation_std = simulate_score(
                        profile=profile,
                        skill=skill,
                        difficulty=difficulty,
                        observation_index=per_skill_count[skill],
                        seed=seed,
                    )
                    model.update(
                        skill,
                        ability_score,
                        observation_std=ability_observation_std,
                    )
                    job_variance_after = model.job_variance()
                    recent = (recent + [skill])[-6:]
                    per_skill_count[skill] += 1
                    record = {
                        "created_at": utc_now(),
                        "strategy": strategy,
                        "candidate_id": profile.candidate_id,
                        "level": profile.level,
                        "turn": turn,
                        "skill": skill,
                        "cluster": SKILL_TO_CLUSTER[skill],
                        "difficulty": difficulty,
                        "question_prompt_variant": question_context["prompt_variant"],
                        "question_context": question_context,
                        "question": question,
                        "question_quality_scores": {
                            name: quality_scores[name]
                            for name in QUESTION_QUALITY_DIMENSIONS
                        },
                        "overall_question_quality": quality_scores["overall_question_quality"],
                        "quality_threshold": quality_threshold,
                        "quality_gate_enabled": regenerate_low_quality,
                        "quality_threshold_met": (
                            quality_scores["overall_question_quality"] >= quality_threshold
                        ),
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": question_attempts,
                        "answer": answer,
                        "assigned_skill_theta": profile.skill_theta[skill],
                        "ability_observation_source": "deterministic_latent_simulation",
                        "ability_observation_score": ability_score,
                        "ability_observation_std": ability_observation_std,
                        "posterior_skill_mean": model.skill_mean(skill),
                        "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                        "posterior_job_mean": model.job_mean(),
                        "posterior_job_std": math.sqrt(model.job_variance()),
                        "job_variance_before": job_variance_before,
                        "job_variance_after": job_variance_after,
                        "realized_job_variance_reduction": (
                            job_variance_before - job_variance_after
                        ),
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

    ability_diagnostics = {}
    question_quality_summary = {}
    for strategy in strategies:
        ability_rows = [row for row in candidate_rows if row["strategy"] == strategy]
        turn_rows = [
            row for (row_strategy, _), group in grouped.items()
            if row_strategy == strategy for row in group
        ]
        truth = [row["true_job_score"] for row in ability_rows]
        prediction = [row["estimated_job_score"] for row in ability_rows]
        ability_diagnostics[strategy] = {
            **regression_metrics(truth, prediction),
            "decision_accuracy": decision_accuracy(truth, prediction),
            "skill_coverage": sum(row["skill_coverage"] for row in ability_rows) / len(ability_rows),
            "cluster_coverage": sum(row["cluster_coverage"] for row in ability_rows) / len(ability_rows),
        }
        question_quality_summary[strategy] = summarize_question_records(turn_rows)
    result = {
        "warning": (
            "Question quality is the primary outcome. Candidate answers are Qwen "
            "simulations used only as dialogue context; Bayesian state updates use "
            "reproducible observations from controlled latent profiles."
        ),
        "setup": {
            "question_model": question_model,
            "question_judge_model": question_judge_model,
            "question_judge_repeats": question_judge_repeats,
            "quality_threshold": quality_threshold,
            "max_regenerations": max_regenerations,
            "regenerate_low_quality": regenerate_low_quality,
            "candidate_model": candidate_model,
            "ability_update_source": "deterministic_latent_simulation",
            "candidates": n_candidates,
            "questions": questions,
            "strategies": list(strategies),
        },
        "summary": question_quality_summary,
        "ability_state_diagnostics": ability_diagnostics,
    }
    write_csv(output_dir / "candidate_results.csv", candidate_rows)
    atomic_json(output_dir / "summary.json", result)
    return result


# =============================================================================
# Question-quality analysis (kept here so the experiment needs one Python file)
# =============================================================================

def load_question_quality_jsonl(path: Path) -> List[Dict]:
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
            required = (
                "strategy", "candidate_id", "turn", "difficulty", "cluster",
                "question", "question_quality_scores", "overall_question_quality",
                "regeneration_count", "question_generation_attempts",
            )
            missing = [name for name in required if name not in row]
            if missing:
                raise ValueError(
                    f"Line {line_number} is not a question-quality record; "
                    f"missing {missing}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def quality_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def quality_sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def quality_metric_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": quality_mean(values),
        f"{prefix}_std": quality_sample_std(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def first_attempt_quality(row: Mapping) -> float:
    attempts = row["question_generation_attempts"]
    if not attempts:
        raise ValueError("question_generation_attempts cannot be empty")
    return float(attempts[0]["quality_scores"]["overall_question_quality"])


def question_judge_call_count(row: Mapping) -> int:
    return sum(
        1 + int(detail.get("format_retry_count", 0))
        for attempt in row["question_generation_attempts"]
        for detail in attempt.get("question_judge_details", [])
    )


def summarize_quality_rows(rows: Sequence[Mapping]) -> Dict[str, float]:
    result: Dict[str, float] = {"n_turns": len(rows)}
    active_dimensions = [
        name for name in QUESTION_QUALITY_DIMENSIONS
        if all(name in row["question_quality_scores"] for row in rows)
    ]
    for name in active_dimensions:
        values = [float(row["question_quality_scores"][name]) for row in rows]
        result.update(quality_metric_stats(values, name))

    overall = [float(row["overall_question_quality"]) for row in rows]
    first = [first_attempt_quality(row) for row in rows]
    thresholds = [float(row.get("quality_threshold", 7.0)) for row in rows]
    regenerations = [int(row["regeneration_count"]) for row in rows]
    attempts = [len(row["question_generation_attempts"]) for row in rows]
    final_pass = [
        bool(row.get("quality_threshold_met", score >= threshold))
        for row, score, threshold in zip(rows, overall, thresholds)
    ]
    gate_enabled = [bool(row.get("quality_gate_enabled", True)) for row in rows]
    result.update(quality_metric_stats(overall, "overall_question_quality"))
    result.update(quality_metric_stats(first, "first_attempt_overall_quality"))

    within_question_std = []
    for row in rows:
        details = row["question_generation_attempts"][-1].get(
            "question_judge_details", []
        )
        repeated_overall = [
            quality_mean([float(detail[name]) for name in active_dimensions])
            for detail in details
            if active_dimensions and all(name in detail for name in active_dimensions)
        ]
        if repeated_overall:
            within_question_std.append(quality_sample_std(repeated_overall))

    result.update({
        "first_attempt_pass_rate": quality_mean([
            float(score >= threshold) for score, threshold in zip(first, thresholds)
        ]),
        "final_pass_rate": quality_mean([float(value) for value in final_pass]),
        "below_threshold_rate": quality_mean([float(not value) for value in final_pass]),
        "quality_gate_enabled_rate": quality_mean([float(value) for value in gate_enabled]),
        "exhausted_failure_rate": quality_mean([
            float(enabled and not passed)
            for enabled, passed in zip(gate_enabled, final_pass)
        ]),
        "no_regeneration_rate": quality_mean([
            float(value == 0) for value in regenerations
        ]),
        "average_regeneration_count": quality_mean(regenerations),
        "average_generation_attempts": quality_mean(attempts),
        "average_question_judge_api_calls": quality_mean([
            question_judge_call_count(row) for row in rows
        ]),
        "format_retry_turn_rate": quality_mean([
            float(question_judge_call_count(row) > sum(
                len(attempt.get("question_judge_details", []))
                for attempt in row["question_generation_attempts"]
            ))
            for row in rows
        ]),
        "average_within_question_judge_std": (
            quality_mean(within_question_std) if within_question_std else 0.0
        ),
    })
    if all("realized_job_variance_reduction" in row for row in rows):
        reductions = [float(row["realized_job_variance_reduction"]) for row in rows]
        result["average_realized_job_variance_reduction"] = quality_mean(reductions)
        result["total_realized_job_variance_reduction"] = sum(reductions)
    return result


def grouped_quality_summaries(
    rows: Sequence[Mapping], keys: Sequence[str]
) -> List[Dict]:
    groups: Dict[Tuple, List[Mapping]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    results = []
    for group_key in sorted(groups, key=lambda value: tuple(map(str, value))):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary.update(summarize_quality_rows(groups[group_key]))
        results.append(summary)
    return results


def quality_percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def quality_bootstrap_ci(
    values: Sequence[float], repeats: int = 10000, seed: int = 42
) -> Tuple[float, float]:
    rng = random.Random(seed)
    boot = sorted(
        quality_mean([rng.choice(values) for _ in values])
        for _ in range(repeats)
    )
    return quality_percentile(boot, 0.025), quality_percentile(boot, 0.975)


def paired_question_quality_comparison(
    rows: Sequence[Mapping], first: str = "bridge", second: str = "random"
) -> Dict:
    candidate_values: Dict[Tuple[str, object], List[float]] = defaultdict(list)
    for row in rows:
        candidate_values[(str(row["strategy"]), row["candidate_id"])].append(
            float(row["overall_question_quality"])
        )
    first_ids = {
        candidate_id for strategy, candidate_id in candidate_values
        if strategy == first
    }
    second_ids = {
        candidate_id for strategy, candidate_id in candidate_values
        if strategy == second
    }
    common = sorted(first_ids & second_ids, key=str)
    deltas = [
        quality_mean(candidate_values[(first, candidate_id)])
        - quality_mean(candidate_values[(second, candidate_id)])
        for candidate_id in common
    ]
    if not deltas:
        return {
            "available": False,
            "reason": f"No candidates shared by {first} and {second}",
        }
    ci_low, ci_high = quality_bootstrap_ci(deltas)
    delta_std = quality_sample_std(deltas)
    return {
        "available": True,
        "comparison": f"{first}_minus_{second}",
        "unit": "candidate mean across turns",
        "n_paired_candidates": len(deltas),
        "mean_overall_quality_delta": quality_mean(deltas),
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "paired_effect_size_dz": quality_mean(deltas) / delta_std if delta_std > 0 else 0.0,
        "first_win_rate": quality_mean([float(value > 0) for value in deltas]),
        "tie_rate": quality_mean([float(value == 0) for value in deltas]),
        "note": (
            "Strategies select different skills and difficulties, so this paired delta "
            "measures end-to-end sequence quality rather than generator quality alone."
        ),
    }


def flatten_quality_turn(row: Mapping) -> Dict:
    flattened = {
        "strategy": row["strategy"],
        "candidate_id": row["candidate_id"],
        "turn": row["turn"],
        "skill": row.get("skill", ""),
        "cluster": row["cluster"],
        "difficulty": row["difficulty"],
        "question": row["question"],
        "overall_question_quality": row["overall_question_quality"],
        "first_attempt_overall_quality": first_attempt_quality(row),
        "quality_threshold": row.get("quality_threshold", 7.0),
        "quality_threshold_met": row.get("quality_threshold_met", ""),
        "regeneration_count": row["regeneration_count"],
        "generation_attempts": len(row["question_generation_attempts"]),
        "question_judge_api_calls": question_judge_call_count(row),
    }
    flattened.update({
        name: row["question_quality_scores"][name]
        for name in QUESTION_QUALITY_DIMENSIONS
        if name in row["question_quality_scores"]
    })
    return flattened


def write_quality_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_question_quality_analysis(input_path: Path, output_dir: Path) -> Dict:
    rows = load_question_quality_jsonl(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_strategy = grouped_quality_summaries(rows, ["strategy"])
    by_difficulty = grouped_quality_summaries(rows, ["strategy", "difficulty"])
    by_cluster = grouped_quality_summaries(rows, ["strategy", "cluster"])
    comparison = paired_question_quality_comparison(rows)
    report = {
        "input": str(input_path.resolve()),
        "n_records": len(rows),
        "strategies": sorted({row["strategy"] for row in rows}),
        "overall": summarize_quality_rows(rows),
        "by_strategy": by_strategy,
        "bridge_vs_random": comparison,
        "interpretation_notes": [
            "Question-quality scores never enter BayesianAbilityModel.",
            "Turns from the same candidate are repeated measures; candidate means are paired.",
            "Different strategies select different skill/difficulty mixtures; inspect stratified CSV files.",
        ],
    }
    atomic_json(output_dir / "question_quality_summary.json", report)
    write_quality_csv(output_dir / "question_quality_by_strategy.csv", by_strategy)
    write_quality_csv(output_dir / "question_quality_by_difficulty.csv", by_difficulty)
    write_quality_csv(output_dir / "question_quality_by_cluster.csv", by_cluster)
    write_quality_csv(
        output_dir / "question_quality_turns.csv",
        [flatten_quality_turn(row) for row in rows],
    )
    print("\nQuestion quality by strategy")
    for row in by_strategy:
        print(
            f"{row['strategy']}: n={row['n_turns']}, "
            f"overall={row['overall_question_quality_mean']:.4f}, "
            f"below_threshold={row['below_threshold_rate']:.4f}"
        )
    print("\nPaired comparison")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"\nSaved analysis to: {output_dir.resolve()}")
    return report


# =============================================================================
# CLI
# =============================================================================

def print_compact(title: str, result: Dict) -> None:
    print(f"\n{title}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["exp1", "exp2", "qwen", "analyze", "all"],
        default="all",
    )
    parser.add_argument("--output", default="runs", help="Output root directory")
    parser.add_argument(
        "--analysis-input",
        default=None,
        help="turns.jsonl path; defaults to <output>/exp3_qwen/turns.jsonl",
    )
    parser.add_argument(
        "--analysis-output",
        default=None,
        help="Analysis directory; defaults beside turns.jsonl as quality_analysis",
    )
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
    parser.add_argument(
        "--regenerate-low-quality",
        action="store_true",
        help=(
            "Regenerate questions below --quality-threshold. Disabled by default, "
            "so every generated question is retained for unbiased analysis."
        ),
    )
    parser.add_argument("--candidate-model", default="qwen-max")
    parser.add_argument("--qwen-candidates", type=int, default=30)
    parser.add_argument("--qwen-questions", type=int, default=12)
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument("--confirm-api-calls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    if args.mode == "analyze":
        input_path = (
            Path(args.analysis_input)
            if args.analysis_input
            else output_root / "exp3_qwen" / "turns.jsonl"
        )
        analysis_output = (
            Path(args.analysis_output)
            if args.analysis_output
            else input_path.parent / "quality_analysis"
        )
        run_question_quality_analysis(input_path, analysis_output)
        return

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
        if args.question_judge_repeats < 1:
            raise SystemExit("--question-judge-repeats must be at least one")
        if not 1.0 <= args.quality_threshold <= 10.0:
            raise SystemExit("--quality-threshold must be within [1,10]")
        if args.max_regenerations < 0:
            raise SystemExit("--max-regenerations must be non-negative")
        question_attempts = (
            args.max_regenerations + 1 if args.regenerate_low_quality else 1
        )
        calls_per_turn = (
            question_attempts * (1 + args.question_judge_repeats)
            + 1
        )
        estimated_calls = (
            args.qwen_candidates
            * len(args.strategies)
            * args.qwen_questions
            * calls_per_turn
        )
        print(f"Estimated planned API calls (excluding network/format retries): {estimated_calls}")
        if not args.confirm_api_calls:
            raise SystemExit("Re-run with --confirm-api-calls after checking the estimated cost")
        result = run_qwen(
            output_root=output_root,
            api_key=api_key,
            base_url=args.base_url,
            question_model=args.question_model,
            question_judge_model=args.question_judge_model,
            candidate_model=args.candidate_model,
            n_candidates=args.qwen_candidates,
            questions=args.qwen_questions,
            strategies=args.strategies,
            question_judge_repeats=args.question_judge_repeats,
            quality_threshold=args.quality_threshold,
            max_regenerations=args.max_regenerations,
            regenerate_low_quality=args.regenerate_low_quality,
            seed=args.seed,
            request_delay=args.request_delay,
        )
        print_compact("Qwen pilot complete", result)


if __name__ == "__main__":
    main()
