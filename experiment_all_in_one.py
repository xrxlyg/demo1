#!/usr/bin/env python3
"""
Single-file experiments for hierarchical Bayesian technical interviews.

The primary model is a four-level capability tree with an explicit Gaussian
variable at the root, domain, sub-capability, and leaf-skill levels.  Leaf
observations are conditioned through the joint posterior, so one answer updates
its ancestors and other skills in proportion to their shared ancestry.

Included:
  1. A fixed 1-root / 9-domain / 27-sub-capability / 108-leaf tree.
  2. Exact linear-Gaussian tree, flat-graph, and independent-skill posteriors.
  3. Tree-BRIDGE, Tree-Random, uncertainty, fixed, and propagation ablations.
  4. Synthetic estimator/selector experiments and an optional Qwen pilot.
  5. Fixed-target prompt ablation and candidate-paired quality analysis.
  6. V2 contrast-card prompting, budget-matched draft/revision, and blind A/B judging.
  7. V2.1 structured diagnostic blueprints, a shared technical-validity guard,
     and optional counterfactual H0/H1 discrimination evaluation.
  8. V2.2 atomic diagnostic probes with answerability-first realization.  It
     preserves V2.1 while reducing unsupported assumptions and multi-part tasks.
  9. V2.3 shared neutral technical scaffolds, same-branch history filtering,
     and a strict technical-validity gate in blind pairwise comparison.
 10. V2.4 shared factual scenarios with independently designed, budget-matched
     probes.  Tree may adapt the task to H0/H1, but neither arm may alter facts.

Examples:
  python experiment_all_in_one.py --mode exp1 --output runs
  python experiment_all_in_one.py --mode exp2 --seeds 20 --output runs
  python experiment_all_in_one.py --mode prompt_ablation \
      --qwen-candidates 3 --qwen-questions 3 --confirm-api-calls --output runs
  python experiment_all_in_one.py --mode prompt_ablation_v2 \
      --qwen-candidates 3 --qwen-questions 3 --pairwise-judge-repeats 3 \
      --confirm-api-calls --output runs_v2
  python experiment_all_in_one.py --mode prompt_ablation_v21 \
      --qwen-candidates 3 --qwen-questions 3 --pairwise-judge-repeats 3 \
      --diagnostic-discrimination-repeats 1 \
      --confirm-api-calls --output runs_v21
  python experiment_all_in_one.py --mode prompt_ablation_v22 \
      --qwen-candidates 3 --qwen-questions 3 --pairwise-judge-repeats 3 \
      --diagnostic-discrimination-repeats 1 \
      --confirm-api-calls --output runs_v22
  python experiment_all_in_one.py --mode prompt_ablation_v23 \
      --qwen-candidates 3 --qwen-questions 3 --pairwise-judge-repeats 3 \
      --diagnostic-discrimination-repeats 1 \
      --confirm-api-calls --output runs_v23
  python experiment_all_in_one.py --mode prompt_ablation_v24 \
      --qwen-candidates 3 --qwen-questions 3 --pairwise-judge-repeats 3 \
      --diagnostic-discrimination-repeats 1 \
      --confirm-api-calls --output runs_v24
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
# Fixed four-level capability tree
# =============================================================================

CAPABILITY_TREE: Dict[str, Dict[str, List[str]]] = {
    "Programming": {
        "Language Foundations": ["Java", "Python", "Go", "Type Systems"],
        "Algorithms": [
            "Data Structures", "Algorithm Design", "Complexity Analysis",
            "Dynamic Programming",
        ],
        "Concurrent Programming": [
            "Threads", "Locks", "Async Programming", "Memory Model",
        ],
    },
    "Database and Storage": {
        "Relational Databases": [
            "SQL", "Query Planning", "SQL Indexing", "Transactions",
        ],
        "Storage Engines": [
            "MySQL Tuning", "MVCC", "Database Replication", "Database Sharding",
        ],
        "Caching": [
            "Redis Caching", "Cache Consistency", "Cache Eviction",
            "Distributed Cache",
        ],
    },
    "Distributed Systems": {
        "Distributed Fundamentals": [
            "CAP Trade-offs", "Consistency Models", "Data Partitioning",
            "Distributed Fault Models",
        ],
        "Consistency and Coordination": [
            "Consensus", "Leader Election", "Distributed Locks",
            "Cluster Membership",
        ],
        "Distributed Transactions": [
            "Two-Phase Commit", "Saga Pattern", "Idempotency",
            "Eventual Consistency",
        ],
    },
    "Middleware and Messaging": {
        "Messaging Platforms": ["Kafka", "RabbitMQ", "Publish-Subscribe", "Message Ordering"],
        "Delivery Semantics": [
            "At-Least-Once Delivery", "Exactly-Once Semantics", "Dead-Letter Queues",
            "Backpressure",
        ],
        "Service Integration": [
            "API Gateway", "Service Discovery", "RPC", "Event-Driven Architecture",
        ],
    },
    "Cloud and DevOps": {
        "Containers and Orchestration": [
            "Docker", "Kubernetes", "Container Scheduling", "Service Mesh",
        ],
        "Delivery Automation": [
            "CI/CD", "Infrastructure as Code", "Release Strategies",
            "Artifact Management",
        ],
        "Cloud Operations": [
            "Observability", "Site Reliability Engineering", "Incident Response",
            "Capacity Planning",
        ],
    },
    "System Design": {
        "Architecture": [
            "Service Decomposition", "Scalability", "Load Balancing", "Multi-Tenancy",
        ],
        "Interface Design": ["API Design", "REST", "GraphQL", "API Versioning"],
        "Performance Engineering": [
            "Performance Optimization", "Queueing", "Content Delivery Networks",
            "Rate Limiting",
        ],
    },
    "Security": {
        "Application Security": [
            "Authentication", "Authorization", "OWASP Risks", "Input Validation",
        ],
        "Data Security": [
            "Encryption", "Key Management", "Secrets Management", "Data Privacy",
        ],
        "Cloud Security": [
            "Identity and Access Management", "Network Security",
            "Software Supply Chain Security", "Threat Modeling",
        ],
    },
    "Software Engineering Practice": {
        "Design Quality": ["Design Patterns", "SOLID Principles", "Refactoring", "Code Review"],
        "Team Practice": ["Git", "Agile Delivery", "Requirements Analysis", "Documentation"],
        "Sustainable Delivery": [
            "Engineering Estimation", "Technical Debt", "Feature Flags",
            "Backward Compatibility",
        ],
    },
    "Testing and Reliability": {
        "Testing Methods": [
            "Unit Testing", "Integration Testing", "End-to-End Testing",
            "Property-Based Testing",
        ],
        "Resilience Engineering": [
            "Fault Tolerance", "Circuit Breakers", "Retry Strategies",
            "Chaos Engineering",
        ],
        "Quality Engineering": [
            "Static Analysis", "Debugging", "Profiling", "Testability",
        ],
    },
}

ROOT_NODE = "root"
ROOT_LABEL = "Software Engineering Ability"


@dataclass(frozen=True)
class CapabilityNode:
    node_id: str
    label: str
    parent_id: Optional[str]
    level: int
    is_leaf: bool
    path: Tuple[str, ...]


def _node_token(text: str) -> str:
    return "_".join("".join(character.lower() if character.isalnum() else " " for character in text).split())


def build_capability_nodes() -> Tuple[List[CapabilityNode], Dict[str, str]]:
    nodes = [CapabilityNode(ROOT_NODE, ROOT_LABEL, None, 0, False, (ROOT_LABEL,))]
    leaf_to_node: Dict[str, str] = {}
    for domain, branches in CAPABILITY_TREE.items():
        domain_id = f"domain::{_node_token(domain)}"
        nodes.append(CapabilityNode(
            domain_id, domain, ROOT_NODE, 1, False, (ROOT_LABEL, domain),
        ))
        for branch, skills in branches.items():
            branch_id = f"branch::{_node_token(domain)}::{_node_token(branch)}"
            nodes.append(CapabilityNode(
                branch_id, branch, domain_id, 2, False,
                (ROOT_LABEL, domain, branch),
            ))
            for skill in skills:
                if skill in leaf_to_node:
                    raise ValueError(f"Duplicate leaf skill in CAPABILITY_TREE: {skill}")
                leaf_id = f"leaf::{_node_token(domain)}::{_node_token(branch)}::{_node_token(skill)}"
                leaf_to_node[skill] = leaf_id
                nodes.append(CapabilityNode(
                    leaf_id, skill, branch_id, 3, True,
                    (ROOT_LABEL, domain, branch, skill),
                ))
    return nodes, leaf_to_node


CAPABILITY_NODES, SKILL_TO_NODE = build_capability_nodes()
NODE_BY_ID = {node.node_id: node for node in CAPABILITY_NODES}
SKILLS = [node.label for node in CAPABILITY_NODES if node.is_leaf]
CLUSTERS = list(CAPABILITY_TREE)
SKILL_CLUSTERS: Dict[str, List[str]] = {
    domain: [skill for skills in branches.values() for skill in skills]
    for domain, branches in CAPABILITY_TREE.items()
}
SKILL_TO_CLUSTER = {
    skill: domain for domain, skills in SKILL_CLUSTERS.items() for skill in skills
}
SKILL_TO_BRANCH = {
    skill: branch
    for _domain, branches in CAPABILITY_TREE.items()
    for branch, skills in branches.items()
    for skill in skills
}
SKILL_PATHS = {
    node.label: node.path for node in CAPABILITY_NODES if node.is_leaf
}
assert len(CAPABILITY_NODES) == 145
assert len(CLUSTERS) == 9
assert sum(len(branches) for branches in CAPABILITY_TREE.values()) == 27
assert len(SKILLS) == 108

# Equal domain contribution; leaves within a domain share its weight.
JOB_WEIGHTS = {
    skill: 1.0 / len(CLUSTERS) / len(SKILL_CLUSTERS[cluster])
    for cluster, skills in SKILL_CLUSTERS.items()
    for skill in skills
}


JOB_DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "uniform": {domain: 1.0 / len(CLUSTERS) for domain in CLUSTERS},
    "backend": {
        "Programming": 0.18,
        "Database and Storage": 0.18,
        "Distributed Systems": 0.16,
        "Middleware and Messaging": 0.15,
        "Cloud and DevOps": 0.08,
        "System Design": 0.15,
        "Security": 0.05,
        "Software Engineering Practice": 0.03,
        "Testing and Reliability": 0.02,
    },
    "cloud_sre": {
        "Programming": 0.08,
        "Database and Storage": 0.08,
        "Distributed Systems": 0.14,
        "Middleware and Messaging": 0.10,
        "Cloud and DevOps": 0.24,
        "System Design": 0.14,
        "Security": 0.09,
        "Software Engineering Practice": 0.04,
        "Testing and Reliability": 0.09,
    },
    "distributed_systems": {
        "Programming": 0.12,
        "Database and Storage": 0.12,
        "Distributed Systems": 0.28,
        "Middleware and Messaging": 0.15,
        "Cloud and DevOps": 0.08,
        "System Design": 0.15,
        "Security": 0.04,
        "Software Engineering Practice": 0.02,
        "Testing and Reliability": 0.04,
    },
}


def job_weights_for(profile_name: str = "uniform") -> Dict[str, float]:
    """Expand preregistered domain weights to normalized leaf-skill weights."""
    if profile_name not in JOB_DOMAIN_WEIGHTS:
        raise KeyError(f"Unknown job profile: {profile_name}")
    domain_weights = JOB_DOMAIN_WEIGHTS[profile_name]
    if set(domain_weights) != set(CLUSTERS):
        raise ValueError(f"Job profile {profile_name!r} does not cover every domain")
    total = sum(domain_weights.values())
    if total <= 0:
        raise ValueError("Job weights must sum to a positive value")
    return {
        skill: domain_weights[domain] / total / len(SKILL_CLUSTERS[domain])
        for domain in CLUSTERS
        for skill in SKILL_CLUSTERS[domain]
    }


JOB_WEIGHT_PROFILES = {
    name: job_weights_for(name) for name in JOB_DOMAIN_WEIGHTS
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
    for domain in CLUSTERS:
        skills = SKILL_CLUSTERS[domain]
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


def write_jsonl_atomic(path: Path, records: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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
    node_theta: Dict[str, float]


def balanced_levels(n_candidates: int) -> List[str]:
    if n_candidates % 3 != 0:
        raise ValueError(
            f"n_candidates={n_candidates} is not divisible by 3; "
            "use 30 (10 per level) or another multiple of three"
        )
    per_level = n_candidates // 3
    return ["junior"] * per_level + ["mid"] * per_level + ["senior"] * per_level


PROFILE_REGIMES = (
    "hierarchical", "weak_hierarchy", "flat_correlated", "independent",
    "shuffled_tree",
)


def _aggregate_internal_truth(skill_theta: Mapping[str, float]) -> Dict[str, float]:
    """Create descriptive internal-node truth from descendant leaf abilities."""
    node_theta: Dict[str, float] = {
        SKILL_TO_NODE[skill]: float(value) for skill, value in skill_theta.items()
    }
    domain_values = []
    for domain, branches in CAPABILITY_TREE.items():
        domain_id = next(
            node.node_id for node in CAPABILITY_NODES
            if node.level == 1 and node.label == domain
        )
        branch_values = []
        for branch, skills in branches.items():
            branch_id = next(
                node.node_id for node in CAPABILITY_NODES
                if node.level == 2 and node.label == branch
                and node.parent_id == domain_id
            )
            value = sum(skill_theta[skill] for skill in skills) / len(skills)
            node_theta[branch_id] = value
            branch_values.append(value)
        domain_value = sum(branch_values) / len(branch_values)
        node_theta[domain_id] = domain_value
        domain_values.append(domain_value)
    node_theta[ROOT_NODE] = sum(domain_values) / len(domain_values)
    return node_theta


def generate_profiles(
    n_candidates: int,
    seed: int,
    regime: str = "hierarchical",
    misspecification_rate: float = 0.0,
) -> List[CandidateProfile]:
    if regime not in PROFILE_REGIMES:
        raise ValueError(f"regime must be one of {PROFILE_REGIMES}, got {regime!r}")
    if not 0.0 <= misspecification_rate <= 1.0:
        raise ValueError("misspecification_rate must be within [0, 1]")
    rng = random.Random(seed)
    levels = balanced_levels(n_candidates)
    rng.shuffle(levels)
    profiles = []
    for candidate_id, level in enumerate(levels):
        config = LEVEL_CONFIG[level]
        global_theta = clip(rng.gauss(config["mean"], config["std"]))
        strong = sorted(rng.sample(CLUSTERS, config["strong"]))
        node_theta: Dict[str, float] = {ROOT_NODE: global_theta}
        skill_theta: Dict[str, float] = {}
        if regime in ("hierarchical", "weak_hierarchy", "shuffled_tree"):
            if regime == "weak_hierarchy":
                domain_std, branch_std, leaf_std, strong_bonus, weak_penalty = (
                    0.65, 0.85, 1.25, 0.40, -0.20,
                )
            else:
                domain_std, branch_std, leaf_std, strong_bonus, weak_penalty = (
                    0.30, 0.40, 0.50, 0.75, -0.35,
                )
            for domain, branches in CAPABILITY_TREE.items():
                domain_id = next(
                    node.node_id for node in CAPABILITY_NODES
                    if node.level == 1 and node.label == domain
                )
                domain_shift = strong_bonus if domain in strong else weak_penalty
                domain_theta = clip(global_theta + domain_shift + rng.gauss(0.0, domain_std))
                node_theta[domain_id] = domain_theta
                for branch, skills in branches.items():
                    branch_id = next(
                        node.node_id for node in CAPABILITY_NODES
                        if node.level == 2 and node.label == branch
                        and node.parent_id == domain_id
                    )
                    branch_theta = clip(domain_theta + rng.gauss(0.0, branch_std))
                    node_theta[branch_id] = branch_theta
                    for skill in skills:
                        value = clip(branch_theta + rng.gauss(0.0, leaf_std))
                        skill_theta[skill] = value
                        node_theta[SKILL_TO_NODE[skill]] = value
            if regime == "shuffled_tree":
                rate = misspecification_rate if misspecification_rate > 0 else 0.20
                count = max(2, int(round(rate * len(SKILLS))))
                selected = rng.sample(SKILLS, min(count, len(SKILLS)))
                values = [skill_theta[skill] for skill in selected]
                values = values[1:] + values[:1]
                for skill, value in zip(selected, values):
                    skill_theta[skill] = value
                    node_theta[SKILL_TO_NODE[skill]] = value
        elif regime == "flat_correlated":
            common = rng.gauss(0.0, 0.35)
            branch_factors = {
                (domain, branch): rng.gauss(0.0, 0.75)
                for domain, branches in CAPABILITY_TREE.items()
                for branch in branches
            }
            for domain, branches in CAPABILITY_TREE.items():
                domain_shift = 0.45 if domain in strong else -0.20
                for branch, skills in branches.items():
                    shared = branch_factors[(domain, branch)]
                    for skill in skills:
                        skill_theta[skill] = clip(
                            global_theta + common + domain_shift
                            + shared + rng.gauss(0.0, 0.85)
                        )
            node_theta = _aggregate_internal_truth(skill_theta)
        else:  # independent
            for skill in SKILLS:
                skill_theta[skill] = clip(rng.gauss(config["mean"], 1.35))
            node_theta = _aggregate_internal_truth(skill_theta)
        profiles.append(CandidateProfile(
            candidate_id=candidate_id,
            level=level,
            global_theta=global_theta,
            performance_std=config["performance_std"],
            strong_clusters=strong,
            skill_theta=skill_theta,
            node_theta=node_theta,
        ))
    return profiles


def true_job_score(
    profile: CandidateProfile,
    job_weights: Optional[Mapping[str, float]] = None,
) -> float:
    weights = JOB_WEIGHTS if job_weights is None else job_weights
    return sum(weights[skill] * profile.skill_theta[skill] for skill in SKILLS)


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
    """Exact joint Gaussian posterior under tree, flat, or independent priors.

    In ``tree`` mode every one of the 145 nodes is a random variable.  The
    prior is induced by ``theta_child = theta_parent + eta_level``.  Therefore
    covariance between any two nodes is exactly the accumulated innovation
    variance on their shared ancestral path.  Conditioning on a leaf score is
    standard Gaussian conditioning over the full node covariance matrix.
    """

    MODES = ("tree", "matched_flat", "flat", "independent")

    def __init__(
        self,
        graph_enabled: Optional[bool] = None,
        propagation: str = "tree",
        prior_mean: float = 5.0,
        prior_std: float = 2.5,
        observation_std: float = 1.2,
        graph_correlation: float = 0.65,
        graph_length_scale: float = 1.5,
        root_std: float = 1.5,
        inheritance_stds: Sequence[float] = (1.2, 1.0, 1.1),
        job_weights: Optional[Mapping[str, float]] = None,
        objective_weights: Sequence[float] = (0.55, 0.25, 0.20),
        repeat_penalty: float = 0.15,
    ) -> None:
        if graph_enabled is not None:
            propagation = "tree" if graph_enabled else "independent"
        if propagation not in self.MODES:
            raise ValueError(f"propagation must be one of {self.MODES}, got {propagation!r}")
        if len(inheritance_stds) != 3:
            raise ValueError("inheritance_stds must contain L1, L2, and leaf standard deviations")
        if len(objective_weights) != 3 or any(value < 0 for value in objective_weights):
            raise ValueError("objective_weights must contain three non-negative values")
        if repeat_penalty < 0:
            raise ValueError("repeat_penalty must be non-negative")
        self.propagation = propagation
        self.skills = list(SKILLS)
        self.nodes = (
            [node.node_id for node in CAPABILITY_NODES]
            if propagation == "tree"
            else [SKILL_TO_NODE[skill] for skill in SKILLS]
        )
        self.index = {node_id: index for index, node_id in enumerate(self.nodes)}
        self.mean = [float(prior_mean)] * len(self.nodes)
        self.observations = {skill: 0 for skill in self.skills}
        self.default_observation_variance = observation_std ** 2
        self.root_std = float(root_std)
        self.inheritance_stds = tuple(float(value) for value in inheritance_stds)
        raw_job_weights = JOB_WEIGHTS if job_weights is None else dict(job_weights)
        if set(raw_job_weights) != set(SKILLS):
            raise ValueError("job_weights must contain every leaf skill exactly once")
        job_weight_total = sum(float(value) for value in raw_job_weights.values())
        if job_weight_total <= 0:
            raise ValueError("job_weights must sum to a positive value")
        self.job_weights = {
            skill: float(raw_job_weights[skill]) / job_weight_total for skill in SKILLS
        }
        self.objective_weights = tuple(float(value) for value in objective_weights)
        self.repeat_penalty = float(repeat_penalty)
        self.covariance = [[0.0] * len(self.nodes) for _ in self.nodes]
        if propagation == "tree":
            innovation_variance = {
                0: self.root_std ** 2,
                1: self.inheritance_stds[0] ** 2,
                2: self.inheritance_stds[1] ** 2,
                3: self.inheritance_stds[2] ** 2,
            }
            ancestor_sets = {
                node_id: set(self._ancestor_ids(node_id, include_self=True))
                for node_id in self.nodes
            }
            for i, first in enumerate(self.nodes):
                for j in range(i, len(self.nodes)):
                    second = self.nodes[j]
                    shared = ancestor_sets[first] & ancestor_sets[second]
                    value = sum(
                        innovation_variance[NODE_BY_ID[node_id].level]
                        for node_id in shared
                    )
                    self.covariance[i][j] = value
                    self.covariance[j][i] = value
        else:
            prior_variance = prior_std ** 2
            for i, first_id in enumerate(self.nodes):
                first = NODE_BY_ID[first_id].label
                for j, second_id in enumerate(self.nodes):
                    second = NODE_BY_ID[second_id].label
                    if propagation == "matched_flat":
                        innovation_variance = {
                            0: self.root_std ** 2,
                            1: self.inheritance_stds[0] ** 2,
                            2: self.inheritance_stds[1] ** 2,
                            3: self.inheritance_stds[2] ** 2,
                        }
                        first_ancestors = set(self._ancestor_ids(first_id, include_self=True))
                        second_ancestors = set(self._ancestor_ids(second_id, include_self=True))
                        value = sum(
                            innovation_variance[NODE_BY_ID[node_id].level]
                            for node_id in first_ancestors & second_ancestors
                        )
                    elif i == j:
                        value = prior_variance
                    elif propagation == "flat":
                        if SKILL_TO_BRANCH[first] == SKILL_TO_BRANCH[second]:
                            distance = 2
                        elif SKILL_TO_CLUSTER[first] == SKILL_TO_CLUSTER[second]:
                            distance = 4
                        else:
                            distance = 6
                        value = (
                            prior_variance * graph_correlation
                            * math.exp(-distance / graph_length_scale)
                        )
                    else:
                        value = 0.0
                    self.covariance[i][j] = value

    @staticmethod
    def _ancestor_ids(node_id: str, include_self: bool = False) -> List[str]:
        result = [node_id] if include_self else []
        current = NODE_BY_ID[node_id]
        while current.parent_id is not None:
            result.append(current.parent_id)
            current = NODE_BY_ID[current.parent_id]
        result.reverse()
        return result

    def _skill_index(self, skill: str) -> int:
        if skill not in SKILL_TO_NODE:
            raise KeyError(f"Unknown leaf skill: {skill}")
        return self.index[SKILL_TO_NODE[skill]]

    def skill_mean(self, skill: str) -> float:
        return self.mean[self._skill_index(skill)]

    def skill_variance(self, skill: str) -> float:
        index = self._skill_index(skill)
        return max(self.covariance[index][index], _EPS)

    def node_mean(self, node_id: str) -> float:
        if node_id not in self.index:
            raise KeyError(f"Node {node_id!r} is not explicit in {self.propagation} mode")
        return self.mean[self.index[node_id]]

    def node_variance(self, node_id: str) -> float:
        if node_id not in self.index:
            raise KeyError(f"Node {node_id!r} is not explicit in {self.propagation} mode")
        index = self.index[node_id]
        return max(self.covariance[index][index], _EPS)

    def posterior_path(self, skill: str) -> List[Dict[str, object]]:
        node_ids = self._ancestor_ids(SKILL_TO_NODE[skill], include_self=True)
        result = []
        for node_id in node_ids:
            node = NODE_BY_ID[node_id]
            if node_id in self.index:
                mean = self.node_mean(node_id)
                variance = self.node_variance(node_id)
            else:
                descendants = [
                    name for name in SKILLS
                    if tuple(SKILL_PATHS[name][:node.level + 1]) == node.path
                ]
                mean = sum(self.skill_mean(name) for name in descendants) / len(descendants)
                variance = sum(self.skill_variance(name) for name in descendants) / len(descendants)
            result.append({
                "node_id": node_id,
                "label": node.label,
                "level": node.level,
                "posterior_mean": mean,
                "posterior_std": math.sqrt(variance),
            })
        return result

    def update(self, skill: str, score: float, observation_std: Optional[float] = None) -> None:
        j = self._skill_index(skill)
        noise = (
            self.default_observation_variance
            if observation_std is None
            else max(float(observation_std) ** 2, _EPS)
        )
        old_mean = list(self.mean)
        old_covariance = [list(row) for row in self.covariance]
        denominator = max(old_covariance[j][j] + noise, _EPS)
        innovation = clip(score) - old_mean[j]
        gain = [old_covariance[i][j] / denominator for i in range(len(self.nodes))]
        self.mean = [old_mean[i] + gain[i] * innovation for i in range(len(self.nodes))]
        n = len(self.nodes)
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
        return self.weighted_job_mean(self.job_weights)

    def weighted_job_mean(self, weights: Mapping[str, float]) -> float:
        return sum(float(weights[skill]) * self.skill_mean(skill) for skill in SKILLS)

    def job_variance(self) -> float:
        return self.weighted_job_variance(self.job_weights)

    def weighted_job_variance(self, weights: Mapping[str, float]) -> float:
        total = 0.0
        for first, wf in weights.items():
            i = self._skill_index(first)
            for second, ws in weights.items():
                j = self._skill_index(second)
                total += wf * ws * self.covariance[i][j]
        return max(total, 0.0)

    def total_variance(self) -> float:
        return sum(self.covariance[i][i] for i in range(len(self.nodes)))

    def hierarchy_variance(self, skill: str) -> float:
        if self.propagation != "tree":
            return 0.0
        path = self._ancestor_ids(SKILL_TO_NODE[skill], include_self=False)
        return sum(self.node_variance(node_id) for node_id in path)

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
        j = self._skill_index(skill)
        denominator = self.covariance[j][j] + observation_variance
        covariance_with_job = sum(
            self.job_weights[name] * self.covariance[self._skill_index(name)][j]
            for name in SKILLS
        )
        decision_reduction = covariance_with_job ** 2 / max(denominator, _EPS)
        global_reduction = sum(row[j] ** 2 for row in self.covariance) / max(denominator, _EPS)
        hierarchy_node_ids = (
            self._ancestor_ids(SKILL_TO_NODE[skill], include_self=False)
            if self.propagation == "tree" else []
        )
        hierarchy_reduction = sum(
            self.covariance[self.index[node_id]][j] ** 2
            for node_id in hierarchy_node_ids
        ) / max(denominator, _EPS)
        if job_variance is None:
            job_variance = self.job_variance()
        if total_variance is None:
            total_variance = self.total_variance()
        relative_decision = decision_reduction / max(job_variance, _EPS)
        relative_global = global_reduction / max(total_variance, _EPS)
        hierarchy_variance = self.hierarchy_variance(skill)
        relative_hierarchy = hierarchy_reduction / max(hierarchy_variance, _EPS)
        repeat_count = sum(1 for name in recent if name == skill)
        question_cost = 1.0 + self.repeat_penalty * repeat_count
        lambda_decision, lambda_global, lambda_hierarchy = self.objective_weights
        utility = (
            lambda_decision * relative_decision
            + lambda_global * relative_global
            + lambda_hierarchy * relative_hierarchy
        ) / question_cost
        return {
            "utility": utility,
            "decision_reduction": decision_reduction,
            "global_reduction": global_reduction,
            "hierarchy_reduction": hierarchy_reduction,
            "relative_decision_reduction": relative_decision,
            "relative_global_reduction": relative_global,
            "relative_hierarchy_reduction": relative_hierarchy,
            "difficulty_efficiency": efficiency,
            "observation_std": math.sqrt(observation_variance),
            "question_cost": question_cost,
            "lambda_decision": lambda_decision,
            "lambda_global": lambda_global,
            "lambda_hierarchy": lambda_hierarchy,
            "repeat_penalty": self.repeat_penalty,
            "propagation": self.propagation,
        }


# =============================================================================
# Selection strategies
# =============================================================================

STRATEGY_SPECS = {
    "bridge": ("tree", "bridge"),
    "tree_bridge": ("tree", "bridge"),
    "random": ("tree", "random"),
    "tree_random": ("tree", "random"),
    "flat_bridge": ("flat", "bridge"),
    "matched_flat_bridge": ("matched_flat", "bridge"),
    "independent_bridge": ("independent", "bridge"),
    "uncertainty": ("tree", "uncertainty"),
    "tree_uncertainty": ("tree", "uncertainty"),
    "fixed": ("tree", "fixed"),
    "tree_fixed": ("tree", "fixed"),
    "tree_bridge_no_job": ("tree", "bridge"),
    "tree_bridge_no_global": ("tree", "bridge"),
    "tree_bridge_no_hierarchy": ("tree", "bridge"),
    "tree_bridge_no_repeat": ("tree", "bridge"),
    "tree_bridge_uniform_job": ("tree", "bridge"),
}


STRATEGY_MODEL_OPTIONS: Dict[str, Dict[str, object]] = {
    "tree_bridge_no_job": {"objective_weights": (0.0, 0.25, 0.20)},
    "tree_bridge_no_global": {"objective_weights": (0.55, 0.0, 0.20)},
    "tree_bridge_no_hierarchy": {"objective_weights": (0.55, 0.25, 0.0)},
    "tree_bridge_no_repeat": {"repeat_penalty": 0.0},
}


def strategy_spec(strategy: str) -> Tuple[str, str]:
    try:
        return STRATEGY_SPECS[strategy]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {strategy}") from exc


def select_skill(
    strategy: str,
    model: BayesianAbilityModel,
    recent: Sequence[str],
    turn: int,
    seed: int,
    candidate_id: int,
) -> Tuple[str, str, Dict[str, float]]:
    _propagation, policy = strategy_spec(strategy)
    if policy == "random":
        rng = random.Random(stable_seed("policy", seed, strategy, candidate_id, turn))
        skill = rng.choice(SKILLS)
        difficulty = closest_difficulty(model.skill_mean(skill))
        return skill, difficulty, model.risk_components(skill, difficulty, recent)

    if policy == "fixed":
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
        if policy == "uncertainty":
            noise_variance = components["observation_std"] ** 2
            variance = model.skill_variance(skill)
            score = (variance ** 2 / (variance + noise_variance)) / max(variance, _EPS)
            score /= components["question_cost"]
        elif policy == "bridge":
            score = components["utility"]
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


def binary_decision_metrics(
    truth: Sequence[float],
    prediction: Sequence[float],
    probabilities: Optional[Sequence[float]] = None,
    threshold: float = 6.5,
) -> Dict[str, float]:
    actual = [value >= threshold for value in truth]
    predicted = [value >= threshold for value in prediction]
    tp = sum(a and p for a, p in zip(actual, predicted))
    tn = sum((not a) and (not p) for a, p in zip(actual, predicted))
    fp = sum((not a) and p for a, p in zip(actual, predicted))
    fn = sum(a and (not p) for a, p in zip(actual, predicted))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    scores = list(prediction if probabilities is None else probabilities)
    positive_scores = [score for score, label in zip(scores, actual) if label]
    negative_scores = [score for score, label in zip(scores, actual) if not label]
    if positive_scores and negative_scores:
        auc = sum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positive_scores for negative in negative_scores
        ) / (len(positive_scores) * len(negative_scores))
    else:
        auc = float("nan")
    return {
        "decision_accuracy": (tp + tn) / len(actual),
        "balanced_accuracy": (
            0.5 * (sensitivity + specificity)
            if not math.isnan(sensitivity) and not math.isnan(specificity)
            else float("nan")
        ),
        "f1": f1,
        "auroc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def strategy_for_candidate(
    profile: CandidateProfile,
    strategy: str,
    questions: int,
    seed: int,
    job_profile: str = "uniform",
    decision_threshold: float = 6.5,
) -> Dict:
    propagation, _policy = strategy_spec(strategy)
    job_weights = JOB_WEIGHT_PROFILES[job_profile]
    selection_job_profile = "uniform" if strategy == "tree_bridge_uniform_job" else job_profile
    selection_job_weights = JOB_WEIGHT_PROFILES[selection_job_profile]
    model = BayesianAbilityModel(
        propagation=propagation,
        job_weights=selection_job_weights,
        **STRATEGY_MODEL_OPTIONS.get(strategy, {}),
    )
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
        truth_job = true_job_score(profile, job_weights)
        posterior_job_mean = model.weighted_job_mean(job_weights)
        posterior_job_std = math.sqrt(model.weighted_job_variance(job_weights))
        trace.append({
            "turn": turn + 1,
            "skill": skill,
            "cluster": SKILL_TO_CLUSTER[skill],
            "branch": SKILL_TO_BRANCH[skill],
            "capability_path": list(SKILL_PATHS[skill]),
            "difficulty": difficulty,
            "score": score,
            "posterior_job_mean": posterior_job_mean,
            "posterior_job_std": posterior_job_std,
            "absolute_job_error": abs(posterior_job_mean - truth_job),
            "decision_correct": float(
                (posterior_job_mean >= decision_threshold)
                == (truth_job >= decision_threshold)
            ),
            **components,
        })

    observed_skills = {row["skill"] for row in trace}
    observed_clusters = {row["cluster"] for row in trace}
    truth = true_job_score(profile, job_weights)
    prediction = model.weighted_job_mean(job_weights)
    leaf_errors = [model.skill_mean(skill) - profile.skill_theta[skill] for skill in SKILLS]
    internal_node_errors = [
        model.node_mean(node.node_id) - profile.node_theta[node.node_id]
        for node in CAPABILITY_NODES if not node.is_leaf and node.node_id in model.index
    ]
    posterior_variance = model.weighted_job_variance(job_weights)
    posterior_std = math.sqrt(posterior_variance)
    pass_probability = 0.5 * (
        1.0 + math.erf((prediction - decision_threshold) / max(posterior_std * math.sqrt(2.0), _EPS))
    )
    true_pass = float(truth >= decision_threshold)
    first_mae_half_turn = next(
        (row["turn"] for row in trace if row["absolute_job_error"] <= 0.5),
        questions + 1,
    )
    return {
        "candidate_id": profile.candidate_id,
        "level": profile.level,
        "strategy": strategy,
        "propagation": propagation,
        "job_profile": job_profile,
        "selection_job_profile": selection_job_profile,
        "true_job_score": truth,
        "estimated_job_score": prediction,
        "absolute_error": abs(prediction - truth),
        "skill_coverage": len(observed_skills) / len(SKILLS),
        "cluster_coverage": len(observed_clusters) / len(CLUSTERS),
        "posterior_job_std": posterior_std,
        "leaf_mae": sum(abs(error) for error in leaf_errors) / len(leaf_errors),
        "leaf_rmse": math.sqrt(sum(error * error for error in leaf_errors) / len(leaf_errors)),
        "internal_node_mae": (
            sum(abs(error) for error in internal_node_errors) / len(internal_node_errors)
            if internal_node_errors else float("nan")
        ),
        "job_95_coverage": float(abs(truth - prediction) <= 1.96 * posterior_std),
        "job_nll": (
            0.5 * math.log(2.0 * math.pi * max(posterior_variance, _EPS))
            + 0.5 * (truth - prediction) ** 2 / max(posterior_variance, _EPS)
        ),
        "pass_probability": pass_probability,
        "brier_score": (pass_probability - true_pass) ** 2,
        "first_turn_job_mae_le_0_5": first_mae_half_turn,
        "reached_job_mae_le_0_5": float(first_mae_half_turn <= questions),
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


def bootstrap_mean_ci(
    values: Sequence[float], repeats: int = 10000, seed: int = 42
) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(repeats)
    )
    return (
        boot[max(0, int(0.025 * (len(boot) - 1)))],
        boot[min(len(boot) - 1, int(0.975 * (len(boot) - 1)))],
    )


def paired_difference_summary(
    differences: Sequence[float], seed: int, positive_interpretation: str
) -> Dict[str, object]:
    values = [float(value) for value in differences]
    ci_low, ci_high = bootstrap_mean_ci(values, seed=seed)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    mean_difference = sum(values) / len(values) if values else float("nan")
    return {
        "n_pairs": len(values),
        "mean_difference": mean_difference,
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "paired_effect_size_dz": mean_difference / std if std > 0 else 0.0,
        "positive_win_rate": sum(value > _EPS for value in values) / len(values),
        "tie_rate": sum(abs(value) <= _EPS for value in values) / len(values),
        "negative_loss_rate": sum(value < -_EPS for value in values) / len(values),
        "paired_sign_flip_p": paired_sign_flip_p(values, seed=seed),
        "positive_interpretation": positive_interpretation,
    }


def run_exp2(
    output_root: Path,
    seeds: int,
    n_candidates: int,
    questions: int,
    strategies: Sequence[str],
    base_seed: int,
    job_profile: str = "uniform",
    profile_regime: str = "hierarchical",
    misspecification_rate: float = 0.0,
    experiment_dir: str = "exp2_selection",
) -> Dict:
    output_dir = output_root / experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = []
    all_candidate_rows = []
    all_trace_rows = []

    for offset in range(seeds):
        seed = base_seed + offset
        profiles = generate_profiles(
            n_candidates, seed, regime=profile_regime,
            misspecification_rate=misspecification_rate,
        )
        for strategy in strategies:
            results = [
                strategy_for_candidate(
                    profile, strategy, questions, seed, job_profile=job_profile,
                )
                for profile in profiles
            ]
            truth = [row["true_job_score"] for row in results]
            prediction = [row["estimated_job_score"] for row in results]
            metrics = regression_metrics(truth, prediction)
            decision_metrics = binary_decision_metrics(
                truth, prediction, [row["pass_probability"] for row in results],
            )
            metrics.update({
                "seed": seed,
                "strategy": strategy,
                **decision_metrics,
                "skill_coverage": sum(row["skill_coverage"] for row in results) / len(results),
                "cluster_coverage": sum(row["cluster_coverage"] for row in results) / len(results),
                "posterior_job_std": sum(row["posterior_job_std"] for row in results) / len(results),
                "leaf_mae": sum(row["leaf_mae"] for row in results) / len(results),
                "leaf_rmse": sum(row["leaf_rmse"] for row in results) / len(results),
                "job_95_coverage": sum(row["job_95_coverage"] for row in results) / len(results),
                "job_nll": sum(row["job_nll"] for row in results) / len(results),
                "brier_score": sum(row["brier_score"] for row in results) / len(results),
                "first_turn_job_mae_le_0_5": sum(
                    row["first_turn_job_mae_le_0_5"] for row in results
                ) / len(results),
                "reached_job_mae_le_0_5": sum(
                    row["reached_job_mae_le_0_5"] for row in results
                ) / len(results),
            })
            seed_metrics.append(metrics)
            for row in results:
                compact = {key: value for key, value in row.items() if key != "trace"}
                compact["seed"] = seed
                all_candidate_rows.append(compact)
                for trace_row in row["trace"]:
                    all_trace_rows.append({
                        "seed": seed,
                        "candidate_id": row["candidate_id"],
                        "level": row["level"],
                        "strategy": strategy,
                        "profile_regime": profile_regime,
                        "job_profile": job_profile,
                        **trace_row,
                    })

    summary = {}
    for strategy in strategies:
        rows = [row for row in seed_metrics if row["strategy"] == strategy]
        summary[strategy] = {
            metric: mean_std([
                float(row[metric]) for row in rows
                if not math.isnan(float(row[metric]))
            ])
            for metric in (
                "mae", "rmse", "pearson", "spearman", "decision_accuracy",
                "balanced_accuracy", "f1", "auroc", "sensitivity", "specificity",
                "skill_coverage", "cluster_coverage", "posterior_job_std",
                "leaf_mae", "leaf_rmse", "job_95_coverage", "job_nll",
                "brier_score", "first_turn_job_mae_le_0_5",
                "reached_job_mae_le_0_5",
            )
        }

    comparisons = {}
    bridge_name = "tree_bridge" if "tree_bridge" in strategies else "bridge"
    if bridge_name in strategies:
        bridge_by_seed = {
            row["seed"]: row["mae"]
            for row in seed_metrics if row["strategy"] == bridge_name
        }
        for baseline in strategies:
            if baseline == bridge_name:
                continue
            baseline_by_seed = {
                row["seed"]: row["mae"]
                for row in seed_metrics if row["strategy"] == baseline
            }
            common = sorted(set(bridge_by_seed) & set(baseline_by_seed))
            differences = [baseline_by_seed[s] - bridge_by_seed[s] for s in common]
            candidate_by_key = {
                (row["seed"], row["candidate_id"], row["strategy"]): row
                for row in all_candidate_rows
            }
            common_candidates = sorted({
                (row["seed"], row["candidate_id"])
                for row in all_candidate_rows if row["strategy"] == bridge_name
            } & {
                (row["seed"], row["candidate_id"])
                for row in all_candidate_rows if row["strategy"] == baseline
            })
            candidate_differences = [
                candidate_by_key[(seed_value, candidate_id, baseline)]["absolute_error"]
                - candidate_by_key[(seed_value, candidate_id, bridge_name)]["absolute_error"]
                for seed_value, candidate_id in common_candidates
            ]
            comparisons[f"{bridge_name}_vs_{baseline}"] = {
                "positive_means_bridge_lower_mae": True,
                "mean_mae_improvement": sum(differences) / len(differences),
                "paired_sign_flip_p": paired_sign_flip_p(
                    differences,
                    seed=stable_seed("sign-flip", baseline, base_seed),
                ),
                "candidate_level_mae": paired_difference_summary(
                    candidate_differences,
                    seed=stable_seed("candidate-bootstrap", baseline, base_seed),
                    positive_interpretation=f"{bridge_name} has lower absolute job-score error",
                ),
            }

    learning_curve = []
    for strategy in strategies:
        for turn in range(1, questions + 1):
            rows = [
                row for row in all_trace_rows
                if row["strategy"] == strategy and row["turn"] == turn
            ]
            learning_curve.append({
                "strategy": strategy,
                "turn": turn,
                "n_seed_candidates": len(rows),
                "job_mae": sum(row["absolute_job_error"] for row in rows) / len(rows),
                "decision_accuracy": sum(row["decision_correct"] for row in rows) / len(rows),
                "posterior_job_std": sum(row["posterior_job_std"] for row in rows) / len(rows),
                "mean_decision_reduction": sum(row["decision_reduction"] for row in rows) / len(rows),
                "mean_global_reduction": sum(row["global_reduction"] for row in rows) / len(rows),
                "mean_hierarchy_reduction": sum(row["hierarchy_reduction"] for row in rows) / len(rows),
            })

    result = {
        "setup": {
            "seeds": seeds,
            "candidates_per_seed": n_candidates,
            "questions": questions,
            "strategies": list(strategies),
            "profile_regime": profile_regime,
            "misspecification_rate": misspecification_rate,
            "job_profile": job_profile,
            "coverage_guard": False,
        },
        "summary": summary,
        "comparisons": comparisons,
        "learning_curve": learning_curve,
    }
    write_csv(output_dir / "seed_metrics.csv", seed_metrics)
    write_csv(output_dir / "candidate_results.csv", all_candidate_rows)
    write_csv(output_dir / "turn_traces.csv", all_trace_rows)
    write_csv(output_dir / "learning_curve.csv", learning_curve)
    atomic_json(output_dir / "summary.json", result)
    return result


# =============================================================================
# Publication experiments: calibrated estimation and controlled ablations
# =============================================================================

ESTIMATOR_METHODS = (
    "single_eval", "running_mean", "ewa", "tree", "matched_flat", "flat",
    "independent",
)


def _prediction_map(
    method: str,
    models: Mapping[str, BayesianAbilityModel],
    last: Mapping[str, float],
    running_sum: Mapping[str, float],
    counts: Mapping[str, int],
    ewa: Mapping[str, float],
) -> Dict[str, float]:
    if method in models:
        return {skill: models[method].skill_mean(skill) for skill in SKILLS}
    if method == "single_eval":
        return {skill: float(last.get(skill, 5.0)) for skill in SKILLS}
    if method == "running_mean":
        return {
            skill: (
                float(running_sum[skill]) / int(counts[skill])
                if counts.get(skill, 0) else 5.0
            )
            for skill in SKILLS
        }
    if method == "ewa":
        return {skill: float(ewa.get(skill, 5.0)) for skill in SKILLS}
    raise KeyError(method)


def run_estimator_benchmark(
    output_root: Path,
    seeds: int,
    n_candidates: int,
    questions: int,
    base_seed: int,
    regimes: Sequence[str],
    job_profile: str = "uniform",
) -> Dict:
    """Compare estimators on identical observations and report calibration."""
    output_dir = output_root / "exp_estimator_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = JOB_WEIGHT_PROFILES[job_profile]
    checkpoints = sorted({value for value in (1, 3, 6, 9, 12, questions) if value <= questions})
    candidate_rows: List[Dict[str, object]] = []
    observation_rows: List[Dict[str, object]] = []

    for regime in regimes:
        for offset in range(seeds):
            seed = base_seed + offset
            profiles = generate_profiles(n_candidates, seed, regime=regime)
            for profile in profiles:
                models = {
                    mode: BayesianAbilityModel(propagation=mode, job_weights=weights)
                    for mode in ("tree", "matched_flat", "flat", "independent")
                }
                last: Dict[str, float] = {}
                running_sum: Dict[str, float] = defaultdict(float)
                counts: Dict[str, int] = defaultdict(int)
                ewa: Dict[str, float] = {}
                observation_count: Dict[str, int] = defaultdict(int)
                for turn in range(1, questions + 1):
                    skill = FIXED_ORDER[(turn - 1) % len(FIXED_ORDER)]
                    difficulty = ("easy", "medium", "hard")[(turn - 1) % 3]
                    score, observation_std = simulate_score(
                        profile, skill, difficulty, observation_count[skill], seed,
                    )
                    observation_count[skill] += 1
                    last[skill] = score
                    counts[skill] += 1
                    running_sum[skill] += score
                    ewa[skill] = score if skill not in ewa else 0.3 * score + 0.7 * ewa[skill]
                    for model in models.values():
                        model.update(skill, score, observation_std=observation_std)
                    observation_rows.append({
                        "regime": regime,
                        "seed": seed,
                        "candidate_id": profile.candidate_id,
                        "level": profile.level,
                        "turn": turn,
                        "skill": skill,
                        "difficulty": difficulty,
                        "true_theta": profile.skill_theta[skill],
                        "observed_score": score,
                        "observation_std": observation_std,
                    })
                    if turn not in checkpoints:
                        continue

                    truth_job = true_job_score(profile, weights)
                    true_pass = float(truth_job >= 6.5)
                    for method in ESTIMATOR_METHODS:
                        predictions = _prediction_map(
                            method, models, last, running_sum, counts, ewa,
                        )
                        leaf_errors = [
                            predictions[skill_name] - profile.skill_theta[skill_name]
                            for skill_name in SKILLS
                        ]
                        predicted_job = sum(
                            weights[skill_name] * predictions[skill_name]
                            for skill_name in SKILLS
                        )
                        model = models.get(method)
                        if model is not None:
                            job_variance = model.weighted_job_variance(weights)
                            job_std = math.sqrt(job_variance)
                            pass_probability = 0.5 * (
                                1.0 + math.erf(
                                    (predicted_job - 6.5)
                                    / max(job_std * math.sqrt(2.0), _EPS)
                                )
                            )
                            leaf_coverage = sum(
                                abs(profile.skill_theta[skill_name] - predictions[skill_name])
                                <= 1.96 * math.sqrt(model.skill_variance(skill_name))
                                for skill_name in SKILLS
                            ) / len(SKILLS)
                            leaf_nll = sum(
                                0.5 * math.log(
                                    2.0 * math.pi * max(model.skill_variance(skill_name), _EPS)
                                )
                                + 0.5 * (
                                    profile.skill_theta[skill_name] - predictions[skill_name]
                                ) ** 2 / max(model.skill_variance(skill_name), _EPS)
                                for skill_name in SKILLS
                            ) / len(SKILLS)
                            job_coverage = float(
                                abs(truth_job - predicted_job) <= 1.96 * job_std
                            )
                            job_nll = (
                                0.5 * math.log(2.0 * math.pi * max(job_variance, _EPS))
                                + 0.5 * (truth_job - predicted_job) ** 2
                                / max(job_variance, _EPS)
                            )
                        else:
                            job_std = leaf_coverage = leaf_nll = job_coverage = job_nll = float("nan")
                            pass_probability = float(predicted_job >= 6.5)
                        internal_errors = []
                        if method == "tree":
                            internal_errors = [
                                models["tree"].node_mean(node.node_id)
                                - profile.node_theta[node.node_id]
                                for node in CAPABILITY_NODES if not node.is_leaf
                            ]
                        candidate_rows.append({
                            "regime": regime,
                            "seed": seed,
                            "candidate_id": profile.candidate_id,
                            "level": profile.level,
                            "turn": turn,
                            "method": method,
                            "leaf_mae": sum(abs(error) for error in leaf_errors) / len(leaf_errors),
                            "leaf_rmse": math.sqrt(
                                sum(error * error for error in leaf_errors) / len(leaf_errors)
                            ),
                            "internal_node_mae": (
                                sum(abs(error) for error in internal_errors) / len(internal_errors)
                                if internal_errors else float("nan")
                            ),
                            "true_job_score": truth_job,
                            "estimated_job_score": predicted_job,
                            "job_absolute_error": abs(predicted_job - truth_job),
                            "job_squared_error": (predicted_job - truth_job) ** 2,
                            "job_posterior_std": job_std,
                            "job_95_coverage": job_coverage,
                            "leaf_95_coverage": leaf_coverage,
                            "job_nll": job_nll,
                            "leaf_nll": leaf_nll,
                            "decision_correct": float(
                                (predicted_job >= 6.5) == (truth_job >= 6.5)
                            ),
                            "pass_probability": pass_probability,
                            "brier_score": (pass_probability - true_pass) ** 2,
                        })

    metric_names = (
        "leaf_mae", "leaf_rmse", "internal_node_mae", "job_absolute_error",
        "job_squared_error", "job_posterior_std", "job_95_coverage",
        "leaf_95_coverage", "job_nll", "leaf_nll", "decision_correct",
        "brier_score",
    )
    summary_rows: List[Dict[str, object]] = []
    for regime in regimes:
        for turn in checkpoints:
            for method in ESTIMATOR_METHODS:
                rows = [
                    row for row in candidate_rows
                    if row["regime"] == regime and row["turn"] == turn
                    and row["method"] == method
                ]
                summary: Dict[str, object] = {
                    "regime": regime,
                    "turn": turn,
                    "method": method,
                    "n_seed_candidates": len(rows),
                }
                for metric in metric_names:
                    values = [
                        float(row[metric]) for row in rows
                        if not math.isnan(float(row[metric]))
                    ]
                    stats = mean_std(values)
                    summary[f"{metric}_mean"] = stats["mean"]
                    summary[f"{metric}_std"] = stats["std"]
                decision_stats = binary_decision_metrics(
                    [float(row["true_job_score"]) for row in rows],
                    [float(row["estimated_job_score"]) for row in rows],
                    [float(row["pass_probability"]) for row in rows],
                )
                summary.update(decision_stats)
                summary_rows.append(summary)

    comparisons: Dict[str, object] = {}
    final_rows = [row for row in candidate_rows if row["turn"] == questions]
    row_index = {
        (row["regime"], row["seed"], row["candidate_id"], row["method"]): row
        for row in final_rows
    }
    for regime in regimes:
        for baseline in ESTIMATOR_METHODS:
            if baseline == "tree":
                continue
            keys = sorted({
                (row["seed"], row["candidate_id"])
                for row in final_rows if row["regime"] == regime and row["method"] == "tree"
            })
            differences = [
                float(row_index[(regime, seed, candidate_id, baseline)]["job_absolute_error"])
                - float(row_index[(regime, seed, candidate_id, "tree")]["job_absolute_error"])
                for seed, candidate_id in keys
            ]
            comparisons[f"{regime}:tree_vs_{baseline}"] = paired_difference_summary(
                differences,
                seed=stable_seed("estimator", regime, baseline, base_seed),
                positive_interpretation="tree has lower absolute job-score error",
            )

    result = {
        "setup": {
            "seeds": seeds,
            "candidates_per_seed": n_candidates,
            "questions": questions,
            "checkpoints": checkpoints,
            "regimes": list(regimes),
            "methods": list(ESTIMATOR_METHODS),
            "job_profile": job_profile,
            "shared_observations_across_methods": True,
        },
        "summary": summary_rows,
        "comparisons": comparisons,
    }
    write_csv(output_dir / "observations.csv", observation_rows)
    write_csv(output_dir / "candidate_checkpoint_metrics.csv", candidate_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    atomic_json(output_dir / "summary.json", result)
    return result


def run_structure_ablation(
    output_root: Path, seeds: int, n_candidates: int, questions: int, base_seed: int,
) -> Dict:
    return run_exp2(
        output_root, seeds, n_candidates, questions,
        ("tree_bridge", "matched_flat_bridge", "flat_bridge", "independent_bridge"),
        base_seed, experiment_dir="exp_structure_ablation",
    )


def run_objective_ablation(
    output_root: Path, seeds: int, n_candidates: int, questions: int, base_seed: int,
) -> Dict:
    return run_exp2(
        output_root, seeds, n_candidates, questions,
        (
            "tree_bridge", "tree_bridge_no_job", "tree_bridge_no_global",
            "tree_bridge_no_hierarchy", "tree_bridge_no_repeat",
            "tree_uncertainty", "tree_random",
        ),
        base_seed, experiment_dir="exp_objective_ablation",
    )


def run_robustness_benchmark(
    output_root: Path, seeds: int, n_candidates: int, questions: int, base_seed: int,
) -> Dict:
    conditions = [
        ("hierarchical", 0.0),
        ("weak_hierarchy", 0.0),
        ("flat_correlated", 0.0),
        ("independent", 0.0),
        ("shuffled_tree", 0.10),
        ("shuffled_tree", 0.20),
        ("shuffled_tree", 0.40),
    ]
    results: Dict[str, object] = {}
    matrix_rows: List[Dict[str, object]] = []
    for regime, rate in conditions:
        label = regime if rate == 0 else f"{regime}_{int(rate * 100)}pct"
        condition_result = run_exp2(
            output_root, seeds, n_candidates, questions,
            ("tree_bridge", "matched_flat_bridge", "independent_bridge", "tree_random"),
            base_seed,
            profile_regime=regime,
            misspecification_rate=rate,
            experiment_dir=f"exp_robustness/{label}",
        )
        results[label] = condition_result
        for strategy, metrics in condition_result["summary"].items():
            row: Dict[str, object] = {
                "condition": label,
                "profile_regime": regime,
                "misspecification_rate": rate,
                "strategy": strategy,
            }
            for metric, values in metrics.items():
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_std"] = values["std"]
            matrix_rows.append(row)
    result = {"conditions": results}
    write_csv(output_root / "exp_robustness" / "robustness_matrix.csv", matrix_rows)
    atomic_json(output_root / "exp_robustness" / "summary.json", result)
    return result


def run_job_conditioned_benchmark(
    output_root: Path, seeds: int, n_candidates: int, questions: int, base_seed: int,
    job_profiles: Sequence[str],
) -> Dict:
    results: Dict[str, object] = {}
    comparison_rows: List[Dict[str, object]] = []
    for job_profile in job_profiles:
        if job_profile == "uniform":
            continue
        job_result = run_exp2(
            output_root, seeds, n_candidates, questions,
            ("tree_bridge", "tree_bridge_uniform_job", "tree_random"),
            base_seed,
            job_profile=job_profile,
            experiment_dir=f"exp_job_conditioned/{job_profile}",
        )
        results[job_profile] = job_result
        for strategy, metrics in job_result["summary"].items():
            row: Dict[str, object] = {"job_profile": job_profile, "strategy": strategy}
            for metric, values in metrics.items():
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_std"] = values["std"]
            comparison_rows.append(row)
    result = {
        "job_profiles": list(job_profiles),
        "domain_weights": JOB_DOMAIN_WEIGHTS,
        "results": results,
    }
    write_csv(
        output_root / "exp_job_conditioned" / "job_conditioned_comparison.csv",
        comparison_rows,
    )
    atomic_json(output_root / "exp_job_conditioned" / "summary.json", result)
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
GENERAL_QUALITY_DIMENSIONS = (
    "skill_relevance", "technical_correctness", "difficulty_match", "clarity",
    "non_redundancy",
)
ADAPTIVE_QUALITY_DIMENSIONS = (
    "contextual_coherence", "diagnostic_value", "adaptive_relevance",
)
COMPOSITE_QUALITY_METRICS = (
    "general_question_quality", "adaptive_diagnostic_quality",
    "overall_question_quality",
)


# The V2 prompt translates posterior numbers into observable evidence.  This is
# deliberately deterministic: the stronger judge never plans or writes the
# question, so qwen-turbo remains the question generator.
BRANCH_DIAGNOSTIC_LENSES: Dict[str, str] = {
    "Language Foundations": "利用具体程序行为或类型错误解释底层语言机制",
    "Algorithms": "选择合适算法，推导复杂度并处理一个决定性的边界条件",
    "Concurrent Programming": "推演一次线程交错并判断对应的安全性或活性后果",
    "Relational Databases": "把查询或事务行为关联到执行计划、隔离规则或并发异常",
    "Storage Engines": "追踪一次读写路径并解释持久化、复制或分片权衡",
    "Caching": "诊断一个缓存故障场景并论证一致性或淘汰决策",
    "Distributed Fundamentals": "从给定网络或故障假设推导一致性后果",
    "Consistency and Coordination": "推演一条故障时间线并区分安全性、活性与恢复行为",
    "Distributed Transactions": "识别部分失败边界并论证补偿、幂等或提交行为",
    "Messaging Platforms": "在具体消息流中推理顺序、分区或消费者行为",
    "Delivery Semantics": "追踪重复、丢失或延迟投递并指出必须维持的业务不变量",
    "Service Integration": "诊断一次跨服务故障并论证接口或服务发现行为",
    "Containers and Orchestration": "追踪调度或运行状态并定位对应的容器编排机制",
    "Delivery Automation": "分析一次发布失败并论证回滚、晋级或制品保证",
    "Cloud Operations": "根据一项事故信号形成诊断并选择可验证的缓解措施",
    "Architecture": "在明确负载或故障约束下作出一个架构决策并论证权衡",
    "Interface Design": "处理一个兼容性或契约边界而不进行宽泛的接口重设计",
    "Performance Engineering": "根据有限证据定位瓶颈并论证一项干预及其代价",
    "Application Security": "追踪一条真实攻击路径并定位准确的信任或授权边界",
    "Data Security": "分析一个数据暴露场景并论证对应的密钥、秘密或隐私控制",
    "Cloud Security": "识别一条权限或供应链路径并在正确边界选择控制措施",
    "Design Quality": "根据耦合、职责与可维护性评价一项具体代码修改",
    "Team Practice": "使用可观察工程证据解决一个交付或协作失败",
    "Sustainable Delivery": "在保持兼容并限制技术债的条件下作出一项变更决策",
    "Testing Methods": "设计一个能够区分疑似故障与其他合理解释的测试",
    "Resilience Engineering": "追踪一条故障传播路径并论证隔离或恢复机制",
    "Quality Engineering": "选择调试、性能分析或静态分析证据以隔离一个根因",
}
assert set(BRANCH_DIAGNOSTIC_LENSES) == {
    branch for branches in CAPABILITY_TREE.values() for branch in branches
}


def posterior_band(value: float) -> str:
    if value < 3.5:
        return "基础概念阶段"
    if value < 5.5:
        return "基础应用阶段"
    if value < 7.5:
        return "具备生产应用能力"
    return "高级能力阶段"


def build_diagnostic_contrast_card(
    ability_model: BayesianAbilityModel,
    skill: str,
    dialogue_history: Sequence[Mapping],
) -> Dict[str, object]:
    mean = ability_model.skill_mean(skill)
    std = math.sqrt(ability_model.skill_variance(skill))
    lower = clip(mean - std)
    upper = clip(mean + std)
    lens = BRANCH_DIAGNOSTIC_LENSES[SKILL_TO_BRANCH[skill]]
    last = dialogue_history[-1] if dialogue_history else None
    previous_gap = (
        "尚无历史回答，需要获取一项能够区分能力状态的决定性证据。"
        if last is None
        else (
            "检验上一轮回答中一个尚未验证的假设或遗漏边界："
            + str(last.get("answer", ""))[:320]
        )
    )
    forbidden = [str(row.get("question", ""))[:220] for row in dialogue_history[-4:]]
    return {
        "estimated_band": posterior_band(mean),
        "posterior_mean": mean,
        "posterior_std": std,
        "lower_state_value": lower,
        "upper_state_value": upper,
        "lower_hypothesis": (
            f"候选人能够陈述{skill}的基本概念，但还不能{lens}。"
        ),
        "upper_hypothesis": (
            f"候选人能够正确{lens}，并论证相应的工程权衡。"
        ),
        "required_discriminating_evidence": lens,
        "previous_answer_gap": previous_gap,
        "forbidden_repetition": forbidden,
        "single_task_constraint": True,
    }


def add_quality_composites(scores: Dict[str, float]) -> Dict[str, float]:
    scores["general_question_quality"] = quality_mean([
        scores[name] for name in GENERAL_QUALITY_DIMENSIONS
    ])
    scores["adaptive_diagnostic_quality"] = quality_mean([
        scores[name] for name in ADAPTIVE_QUALITY_DIMENSIONS
    ])
    scores["overall_question_quality"] = quality_mean([
        scores[name] for name in QUESTION_QUALITY_DIMENSIONS
    ])
    return scores


def build_question_context(
    strategy: str,
    ability_model: BayesianAbilityModel,
    skill: str,
    selector: Mapping[str, float],
    dialogue_history: Sequence[Mapping] = (),
) -> Dict:
    posterior_mean = ability_model.skill_mean(skill)
    posterior_std = math.sqrt(ability_model.skill_variance(skill))
    if posterior_mean < 4.0:
        boundary = "区分概念缺失、常见误解和基本掌握"
    elif posterior_mean < 7.0:
        boundary = "区分仅会解释概念与能够进行实际工程权衡"
    else:
        boundary = "区分熟练应用与能够处理边界条件、故障和复杂权衡"
    path_posteriors = ability_model.posterior_path(skill)
    siblings = [
        name for name in CAPABILITY_TREE[SKILL_TO_CLUSTER[skill]][SKILL_TO_BRANCH[skill]]
        if name != skill
    ]
    sibling_posteriors = sorted([
        {
            "skill": name,
            "posterior_mean": ability_model.skill_mean(name),
            "posterior_std": math.sqrt(ability_model.skill_variance(name)),
        }
        for name in siblings
    ], key=lambda row: (-float(row["posterior_std"]), str(row["skill"])))[:2]
    evidence = []
    for row in list(dialogue_history)[-2:]:
        observed_skill = str(row["skill"])
        evidence.append({
            "skill": observed_skill,
            "observed_score": row.get("ability_observation_score"),
            "observation_std": row.get("ability_observation_std"),
            "posterior_mean": ability_model.skill_mean(observed_skill),
            "posterior_std": math.sqrt(ability_model.skill_variance(observed_skill)),
        })
    weaknesses = sorted(
        (
            {
                "skill": name,
                "posterior_mean": ability_model.skill_mean(name),
                "posterior_std": math.sqrt(ability_model.skill_variance(name)),
            }
            for name in CAPABILITY_TREE[SKILL_TO_CLUSTER[skill]][SKILL_TO_BRANCH[skill]]
            if name != skill
        ),
        key=lambda row: (float(row["posterior_mean"]), -float(row["posterior_std"])),
    )[:2]
    _propagation, policy = strategy_spec(strategy)
    return {
        "skill": skill,
        "cluster": SKILL_TO_CLUSTER[skill],
        "branch": SKILL_TO_BRANCH[skill],
        "capability_path": list(SKILL_PATHS[skill]),
        "path_posteriors": path_posteriors,
        "related_skills": siblings,
        "sibling_posteriors": sibling_posteriors,
        "observed_evidence": evidence,
        "weaknesses": weaknesses,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "decision_variance_reduction": selector["decision_reduction"],
        "global_variance_reduction": selector["global_reduction"],
        "hierarchy_variance_reduction": selector["hierarchy_reduction"],
        "relative_decision_variance_reduction": selector["relative_decision_reduction"],
        "relative_branch_variance_reduction": selector["relative_hierarchy_reduction"],
        "selection_utility": selector["utility"],
        "diagnostic_goal": boundary,
        "diagnostic_hypothesis": (
            f"围绕 {skill} 设计一个问题，用于{boundary}；"
            f"当前相邻能力证据{'不足' if not evidence else '仅部分可用'}。"
        ),
        "propagation": ability_model.propagation,
        "prompt_variant": (
            "bridge_adaptive"
            if policy == "bridge" and ability_model.propagation == "tree"
            else "baseline"
        ),
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
    history = list(dialogue_history)[-1:]
    history_text = "\n".join(
        f"第{row['turn'] + 1}轮：问题：{row['question']} 回答：{row['answer']}"
        for row in history
    ) or "无，这是第一轮。"
    previous_questions = [row["question"] for row in dialogue_history[-6:]]
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
        "评分或Markdown。问题最多两句话，只包含一个主要任务，必须聚焦指定技能、"
        "符合难度，并与上一轮回答存在明确证据时才自然衔接。"
    )
    user_parts = [
        f"目标能力路径：{' → '.join(question_context['capability_path'])}\n"
        f"目标技能：{skill}\n目标难度：{difficulty}（{difficulty_guidance}）\n"
        f"最近对话：\n{history_text}\n"
        f"全部历史问题：{json.dumps(previous_questions, ensure_ascii=False)}\n"
    ]
    if question_context["prompt_variant"] == "bridge_adaptive":
        path_text = "\n".join(
            f"  L{row['level']} {row['label']}: "
            f"均值={row['posterior_mean']:.2f}, 标准差={row['posterior_std']:.2f}"
            for row in question_context["path_posteriors"]
        )
        sibling_text = "; ".join(
            f"{row['skill']}({row['posterior_mean']:.2f}±{row['posterior_std']:.2f})"
            for row in question_context["sibling_posteriors"]
        ) or "无"
        evidence_text = "; ".join(
            f"{row['skill']}: 观测={row['observed_score']}, "
            f"后验={row['posterior_mean']:.2f}±{row['posterior_std']:.2f}"
            for row in question_context["observed_evidence"]
        ) or "无"
        weakness_text = "; ".join(
            f"{row['skill']}({row['posterior_mean']:.2f}±{row['posterior_std']:.2f})"
            for row in question_context["weaknesses"]
        ) or "无"
        user_parts.append(
            "分层BRIDGE诊断上下文：\n"
            f"- 四层后验：\n{path_text}\n"
            f"- 最相关兄弟技能：{sibling_text}\n"
            f"- 最近两条能力证据：{evidence_text}\n"
            f"- 当前分支薄弱点：{weakness_text}\n"
            f"- 当前诊断假设：{question_context['diagnostic_hypothesis']}\n"
            f"- 预期岗位方差下降：{question_context['decision_variance_reduction']:.6f}\n"
            f"- 预期分支方差下降：{question_context['hierarchy_variance_reduction']:.6f}\n"
            f"- 诊断目标：{question_context['diagnostic_goal']}\n"
            "只利用最相关信息，设计一个能区分当前两种最可能能力状态的问题；"
            "避免罗列多个子问题，也不要机械复述上述数值。\n"
        )
    elif question_context["prompt_variant"] == "tree_v2":
        card = question_context["diagnostic_contrast_card"]
        user_parts.append(
            "分层诊断对比卡（不要在问题中复述这些标签或数值）：\n"
            f"- 当前能力状态：{card['estimated_band']}，不确定性={card['posterior_std']:.2f}\n"
            f"- 较低能力假设H0：{card['lower_hypothesis']}\n"
            f"- 较高能力假设H1：{card['upper_hypothesis']}\n"
            f"- 必须获得的区分证据：{card['required_discriminating_evidence']}\n"
            f"- 上一轮尚未验证的缺口：{card['previous_answer_gap']}\n"
            f"- 禁止重复的问题：{json.dumps(card['forbidden_repetition'], ensure_ascii=False)}\n"
            "围绕一个具体场景提出一个主要任务，使H0和H1倾向于产生可区分的回答。"
            "不要询问宽泛定义，不要组合多个并列子问题。\n"
        )
    elif question_context["prompt_variant"] == "plain_v2":
        user_parts.append(
            "生成一个清晰、技术正确、符合指定难度的单一场景问题。"
            "避免重复历史问题，不要组合多个并列子问题。\n"
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
    temperature: float = 0.2,
) -> Dict:
    """Retry judge calls whose content is not valid JSON and retain bad outputs."""
    invalid_outputs = []
    current_user = user
    for format_attempt in range(max_format_attempts):
        raw = client.chat(
            model, system, current_user,
            temperature=temperature, max_tokens=max_tokens,
        )
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
                "请重新完成任务。只输出严格JSON对象，不要Markdown代码块或任何额外文字。"
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
        if key not in ("prompt_variant", "propagation")
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
    return add_quality_composites(means), details


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


def revise_qwen_question(
    client: QwenClient,
    model: str,
    draft: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    question_context: Mapping,
) -> str:
    """Use the same low-cost generator for a second, budget-matched revision."""
    system = (
        "你是技术面试问题编辑器。只输出修订后的一个中文问题，不给答案、解释、"
        "标题、评分或Markdown。最多两句话，只允许一个主要任务。"
    )
    common = (
        f"目标技能：{skill}\n目标难度：{difficulty}\n问题初稿：{draft}\n"
        f"最近历史：{json.dumps(list(dialogue_history)[-2:], ensure_ascii=False)}\n"
    )
    if question_context["prompt_variant"] == "tree_v2":
        card = question_context["diagnostic_contrast_card"]
        instruction = (
            "按诊断目标修订：问题必须通过一个具体场景区分H0和H1，并检验上一轮尚未"
            "验证的一个缺口。不得重复历史问题，不得询问宽泛定义，不得并列多个任务。\n"
            f"H0：{card['lower_hypothesis']}\n"
            f"H1：{card['upper_hypothesis']}\n"
            f"区分证据：{card['required_discriminating_evidence']}\n"
            f"上一轮缺口：{card['previous_answer_gap']}\n"
            f"禁止重复：{json.dumps(card['forbidden_repetition'], ensure_ascii=False)}"
        )
    else:
        instruction = (
            "按通用质量标准修订：确保技术正确、清晰、符合难度，使用一个具体场景，"
            "避免重复历史问题并删除并列子问题。"
        )
    return client.chat(model, system, common + instruction, temperature=0.6, max_tokens=300).strip()


def generate_two_stage_question(
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
    """Draft, revise, then judge; low-quality outputs remain by default."""
    attempts = []
    rejected = []
    maximum_attempts = max_regenerations + 1 if regenerate_low_quality else 1
    for attempt_index in range(maximum_attempts):
        draft_raw = generate_qwen_question(
            client, question_model, skill, difficulty, dialogue_history,
            rejected, question_context,
        )
        final_question = revise_qwen_question(
            client, question_model, draft_raw.strip(), skill, difficulty,
            dialogue_history, question_context,
        )
        quality_scores, judge_details = evaluate_qwen_question(
            client, question_judge_model, skill, difficulty, final_question,
            dialogue_history, judge_repeats, question_context,
        )
        attempts.append({
            "attempt": attempt_index,
            "draft_question": draft_raw.strip(),
            "question": final_question,
            "question_model_raw_output": final_question,
            "quality_scores": quality_scores,
            "question_judge_details": judge_details,
            "two_stage_revision": True,
        })
        if (
            not regenerate_low_quality
            or quality_scores["overall_question_quality"] >= quality_threshold
        ):
            break
        rejected.append({
            "question": final_question,
            "feedback": "; ".join(
                row["reason"] for row in judge_details if row["reason"]
            ),
        })
    accepted = attempts[-1]
    return (
        accepted["question"], accepted["quality_scores"],
        len(attempts) - 1, attempts,
    )


# V2.1 uses the same two qwen-turbo calls for both variants.  The first call
# creates a machine-readable blueprint; the second realizes it as one question.
# The shared guard is intentionally identical so the ablation isolates the
# diagnostic tree context rather than a generic prompt-quality advantage.
V21_BLUEPRINT_FIELDS = (
    "core_concept",
    "scenario_facts",
    "expected_answer_points",
    "technical_risks",
    "single_task",
    "difficulty_rationale",
    "novelty_mechanism",
    "diagnostic_evidence",
)

V21_SHARED_VALIDITY_GUARD = (
    "技术有效性是硬约束。问题必须技术正确，并且能够仅根据题面给出的事实作答；"
    "只考查一个核心概念和一个主要任务；不得包含错误前提、未说明的环境假设、"
    "互相独立的并列任务或含糊指代；题面条件、术语和预期答案必须一致；"
    "如果诊断野心与正确性或清晰度冲突，优先保证正确性和清晰度。"
)

V21_LEAKAGE_TERMS = (
    "H0", "H1", "后验", "能力状态", "能力估计", "诊断假设",
    "不确定性", "标准差", "方差下降", "Tree", "BRIDGE",
)


def _json_object_from_text(raw: str, label: str) -> Dict:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"{label} output does not contain a JSON object")
    data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError(f"{label} output must be a JSON object")
    return data


def parse_question_blueprint_json(raw: str) -> Dict[str, object]:
    data = _json_object_from_text(raw, "Question blueprint")
    parsed: Dict[str, object] = {}
    list_fields = {
        "scenario_facts", "expected_answer_points", "technical_risks",
    }
    for field in V21_BLUEPRINT_FIELDS:
        value = data[field]
        if field in list_fields:
            if not isinstance(value, list) or not value:
                raise ValueError(f"Blueprint field {field} must be a non-empty list")
            items = [str(item).strip()[:300] for item in value if str(item).strip()]
            if not items:
                raise ValueError(f"Blueprint field {field} contains no usable items")
            parsed[field] = items[:5]
        else:
            text_value = str(value).strip()
            if not text_value:
                raise ValueError(f"Blueprint field {field} cannot be empty")
            parsed[field] = text_value[:600]
    return parsed


def blueprint_prompt_payload(blueprint: Mapping) -> Dict[str, object]:
    return {field: blueprint[field] for field in V21_BLUEPRINT_FIELDS}


def build_v21_question_blueprint(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    rejected_attempts: Sequence[Mapping],
    question_context: Mapping,
) -> Dict[str, object]:
    """Plan one answerable diagnostic task without asking the stronger judge."""
    system = (
        "你是技术面试问题规划器。先规划、不要直接写最终问题。只输出严格JSON对象，"
        "不得输出Markdown。JSON必须包含core_concept字符串、scenario_facts字符串数组、"
        "expected_answer_points字符串数组、technical_risks字符串数组、single_task字符串、"
        "difficulty_rationale字符串、novelty_mechanism字符串和diagnostic_evidence字符串。"
        + V21_SHARED_VALIDITY_GUARD
    )
    previous_questions = [str(row.get("question", ""))[:300] for row in dialogue_history[-6:]]
    recent_dialogue = [
        {
            "skill": row.get("skill"),
            "question": str(row.get("question", ""))[:350],
            "answer": str(row.get("answer", ""))[:500],
        }
        for row in dialogue_history[-2:]
    ]
    common = (
        f"目标能力路径：{' → '.join(question_context['capability_path'])}\n"
        f"目标技能：{skill}\n目标难度：{difficulty}\n"
        f"最近对话：{json.dumps(recent_dialogue, ensure_ascii=False)}\n"
        f"已有问题（不可重复核心故障机制）："
        f"{json.dumps(previous_questions, ensure_ascii=False)}\n"
        f"共同质量约束：{V21_SHARED_VALIDITY_GUARD}\n"
    )
    if question_context["prompt_variant"] == "tree_v21":
        card = question_context["diagnostic_contrast_card"]
        variant_instruction = (
            "这是能力树自适应规划。使用H0/H1只确定最有区分力的可观察证据；"
            "最终问题不得出现H0、H1、后验、能力估计或策略名称。\n"
            f"H0：{card['lower_hypothesis']}\n"
            f"H1：{card['upper_hypothesis']}\n"
            f"目标区分证据：{card['required_discriminating_evidence']}\n"
            f"尚未验证的证据缺口：{card['previous_answer_gap']}\n"
            "diagnostic_evidence必须说明候选人的回答中哪一项可观察行为能区分H0/H1；"
            "scenario_facts必须给足判断该行为所需的条件。"
        )
    else:
        variant_instruction = (
            "这是普通固定目标规划，不提供能力假设。围绕目标技能设计一个具有代表性的"
            "单一场景；diagnostic_evidence填写优秀回答应展示的一项可观察技术证据。"
        )
    rejected = [
        {
            "question": str(row.get("question", ""))[:350],
            "feedback": str(row.get("feedback", ""))[:500],
        }
        for row in rejected_attempts[-2:]
    ]
    return request_judge_json(
        client=client,
        model=model,
        system=system,
        user=(
            common + variant_instruction
            + f"\n显式低分重试历史：{json.dumps(rejected, ensure_ascii=False)}"
        ),
        parser=parse_question_blueprint_json,
        max_tokens=750,
        temperature=0.45,
    )


def generate_v21_question_from_blueprint(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    question_context: Mapping,
    blueprint: Mapping,
) -> str:
    """Realize a blueprint with a shared, explicit validity-first self-check."""
    system = (
        "你是技术面试问题生成器。只输出一个最终中文面试问题，不要输出答案、解释、"
        "标题、评分、JSON或Markdown。最多两句话，只允许一个主要任务。"
        + V21_SHARED_VALIDITY_GUARD
        + "输出前静默检查术语、事实一致性、条件充分性、预期答案、难度与单任务结构。"
    )
    recent_dialogue = [
        {
            "question": str(row.get("question", ""))[:350],
            "answer": str(row.get("answer", ""))[:500],
        }
        for row in dialogue_history[-2:]
    ]
    if question_context["prompt_variant"] == "tree_v21":
        variant_instruction = (
            "蓝图中的diagnostic_evidence用于决定让候选人解释、推演或判断什么；"
            "不要把能力假设或诊断语言写进问题。使用最小充分场景获取这一项证据。"
        )
    else:
        variant_instruction = (
            "根据蓝图生成自然、典型且符合难度的普通技术面试问题。"
        )
    user = (
        f"目标能力路径：{' → '.join(question_context['capability_path'])}\n"
        f"目标技能：{skill}\n目标难度：{difficulty}\n"
        f"问题蓝图：{json.dumps(blueprint_prompt_payload(blueprint), ensure_ascii=False)}\n"
        f"最近对话：{json.dumps(recent_dialogue, ensure_ascii=False)}\n"
        f"硬约束：{V21_SHARED_VALIDITY_GUARD}\n"
        f"{variant_instruction}\n只输出最终问题。"
    )
    return client.chat(model, system, user, temperature=0.55, max_tokens=350).strip()


def v21_question_leakage_flags(question: str) -> List[str]:
    lowered = question.lower()
    return [term for term in V21_LEAKAGE_TERMS if term.lower() in lowered]


def generate_blueprint_question(
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
    """V2.1 blueprint -> guarded question -> blind judge pipeline."""
    attempts = []
    rejected = []
    maximum_attempts = max_regenerations + 1 if regenerate_low_quality else 1
    for attempt_index in range(maximum_attempts):
        blueprint = build_v21_question_blueprint(
            client, question_model, skill, difficulty, dialogue_history,
            rejected, question_context,
        )
        question = generate_v21_question_from_blueprint(
            client, question_model, skill, difficulty, dialogue_history,
            question_context, blueprint,
        )
        quality_scores, judge_details = evaluate_qwen_question(
            client, question_judge_model, skill, difficulty, question,
            dialogue_history, judge_repeats, question_context,
        )
        attempts.append({
            "attempt": attempt_index,
            "question_blueprint": blueprint_prompt_payload(blueprint),
            "blueprint_model_raw_output": blueprint.get("raw", ""),
            "blueprint_format_retry_count": blueprint.get("format_retry_count", 0),
            "blueprint_invalid_raw_outputs": blueprint.get("invalid_raw_outputs", []),
            "question": question,
            "question_model_raw_output": question,
            "quality_scores": quality_scores,
            "question_judge_details": judge_details,
            "generation_pipeline": "structured_blueprint_then_validity_guarded_question",
            "technical_validity_guard": True,
            "strategy_leakage_flags": v21_question_leakage_flags(question),
        })
        if (
            not regenerate_low_quality
            or quality_scores["overall_question_quality"] >= quality_threshold
        ):
            break
        rejected.append({
            "question": question,
            "feedback": "; ".join(
                row["reason"] for row in judge_details if row["reason"]
            ),
        })
    accepted = attempts[-1]
    return (
        accepted["question"], accepted["quality_scores"],
        len(attempts) - 1, attempts,
    )


# V2.2 is deliberately a new mode rather than an in-place V2.1 rewrite.  The
# ablation remains budget matched (one blueprint call and one realization call
# per variant), while the blueprint is narrowed to a single, answerable probe.
V22_BLUEPRINT_FIELDS = (
    "core_concept",
    "stable_facts",
    "expected_answer_points",
    "single_decision",
    "correct_answer_outline",
    "assumptions_to_avoid",
    "diagnostic_probe",
    "history_link",
)

V22_SHARED_VALIDITY_GUARD = (
    "技术正确性、条件充分性和单任务结构是硬约束。题目只能依赖题面明确给出的事实与"
    "稳定的标准概念；若答案取决于版本、配置、运行环境或业务目标，必须在题面明确"
    "给出相应条件。只考查一个核心概念，只要求一次判断、推演或选择及其一个核心"
    "理由，不得同时要求设计、实现、监控、排障等多个任务。预期答案必须能由题面"
    "推出；若诊断性与正确性、清晰度冲突，必须简化诊断目标。"
)


def parse_v22_question_blueprint_json(raw: str) -> Dict[str, object]:
    data = _json_object_from_text(raw, "V2.2 question blueprint")
    parsed: Dict[str, object] = {}
    list_fields = {
        "stable_facts", "expected_answer_points", "assumptions_to_avoid",
    }
    for field in V22_BLUEPRINT_FIELDS:
        value = data[field]
        if field in list_fields:
            if not isinstance(value, list) or not value:
                raise ValueError(f"V2.2 blueprint field {field} must be a non-empty list")
            items = [str(item).strip()[:260] for item in value if str(item).strip()]
            if not items:
                raise ValueError(f"V2.2 blueprint field {field} contains no usable items")
            parsed[field] = items[:4]
        else:
            text_value = str(value).strip()
            if not text_value:
                raise ValueError(f"V2.2 blueprint field {field} cannot be empty")
            parsed[field] = text_value[:600]
    return parsed


def v22_blueprint_prompt_payload(blueprint: Mapping) -> Dict[str, object]:
    return {field: blueprint[field] for field in V22_BLUEPRINT_FIELDS}


def build_v22_question_blueprint(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    rejected_attempts: Sequence[Mapping],
    question_context: Mapping,
) -> Dict[str, object]:
    """Plan one minimal, technically checkable probe without using the judge."""
    system = (
        "你是严谨的技术面试问题规划器。先规划，不要直接写最终问题。只输出严格JSON"
        "对象，不得输出Markdown。JSON必须包含core_concept字符串、stable_facts字符串"
        "数组、expected_answer_points字符串数组、single_decision字符串、"
        "correct_answer_outline字符串、assumptions_to_avoid字符串数组、"
        "diagnostic_probe字符串和history_link字符串。"
        + V22_SHARED_VALIDITY_GUARD
    )
    previous_questions = [
        str(row.get("question", ""))[:300] for row in dialogue_history[-6:]
    ]
    recent_dialogue = [
        {
            "skill": row.get("skill"),
            "question": str(row.get("question", ""))[:320],
            "answer": str(row.get("answer", ""))[:420],
        }
        for row in dialogue_history[-2:]
    ]
    difficulty_guidance = {
        "easy": "一个稳定核心概念或常见边界，不使用复杂生产假设",
        "medium": "一个明确工程场景中的机制判断或主要权衡",
        "hard": "一个明确故障或边界条件下的单一推演，不扩展为系统设计题",
    }[difficulty]
    common = (
        f"目标能力路径：{' → '.join(question_context['capability_path'])}\n"
        f"目标技能：{skill}\n目标难度：{difficulty}（{difficulty_guidance}）\n"
        f"最近对话：{json.dumps(recent_dialogue, ensure_ascii=False)}\n"
        f"已有问题（不得重复同一核心机制）："
        f"{json.dumps(previous_questions, ensure_ascii=False)}\n"
        f"共同硬约束：{V22_SHARED_VALIDITY_GUARD}\n"
        "stable_facts限制为1到3项最小充分事实；expected_answer_points限制为1到2项；"
        "single_decision只能包含一个动词任务；correct_answer_outline必须证明该任务可由"
        "题面事实回答；assumptions_to_avoid列出不得擅自补充的环境或实现假设。\n"
    )
    if question_context["prompt_variant"] == "tree_v22":
        card = question_context["diagnostic_contrast_card"]
        variant_instruction = (
            "这是能力树自适应规划。H0/H1仅用于选择一个最小诊断探针，不得增加题面"
            "复杂度。diagnostic_probe只能区分一个原子差异，例如一个常见误解、一次"
            "结果预测、一个边界判断或一个主要权衡；两个状态都必须能够合理回答同一"
            "问题，不得把H1结论预设为题面事实。history_link仅在最近回答包含可直接"
            "引用的明确证据缺口时填写，否则写‘无’。最终问题不得出现任何策略信息。\n"
            f"H0：{card['lower_hypothesis']}\n"
            f"H1：{card['upper_hypothesis']}\n"
            f"候选区分证据：{card['required_discriminating_evidence']}\n"
            f"可能的历史缺口：{card['previous_answer_gap']}"
        )
    else:
        variant_instruction = (
            "这是普通固定目标规划，不提供能力假设。选择一个典型且最小的技术能力"
            "探针；diagnostic_probe填写优秀回答应展示的一项可观察证据。不要为了显得"
            "复杂而引入额外组件、故障或限制。history_link仅在最近回答确有明确关联时"
            "填写，否则写‘无’。"
        )
    rejected = [
        {
            "question": str(row.get("question", ""))[:320],
            "feedback": str(row.get("feedback", ""))[:420],
        }
        for row in rejected_attempts[-2:]
    ]
    return request_judge_json(
        client=client,
        model=model,
        system=system,
        user=(
            common + variant_instruction
            + f"\n显式低分重试历史：{json.dumps(rejected, ensure_ascii=False)}"
        ),
        parser=parse_v22_question_blueprint_json,
        max_tokens=750,
        temperature=0.30,
    )


def generate_v22_question_from_blueprint(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
    question_context: Mapping,
    blueprint: Mapping,
) -> str:
    """Realize an atomic probe after an answerability-first silent check."""
    system = (
        "你是严谨的技术面试问题生成器。只输出一个最终中文面试问题，不要输出答案、"
        "解释、标题、评分、JSON或Markdown。最多两句话，只有一个问号和一个主要任务。"
        + V22_SHARED_VALIDITY_GUARD
        + "输出前先静默完成三步检查：第一，根据stable_facts独立推导正确答案；第二，"
        "删除无法由题面支持的前提、版本默认值和多余任务；第三，确认术语、条件、"
        "预期答案和难度一致。若蓝图过于复杂或存在不确定事实，改写为更保守的标准"
        "机制问题，不得照搬风险内容。"
    )
    recent_dialogue = [
        {
            "question": str(row.get("question", ""))[:320],
            "answer": str(row.get("answer", ""))[:420],
        }
        for row in dialogue_history[-2:]
    ]
    if question_context["prompt_variant"] == "tree_v22":
        variant_instruction = (
            "仅在技术上安全时使用diagnostic_probe，使不同水平的回答自然呈现差异；"
            "不要把诊断假设、能力状态或区分目标写进题面，也不要因此添加第二个任务。"
        )
    else:
        variant_instruction = (
            "根据同一蓝图结构生成自然、典型、符合难度的普通技术面试问题。"
        )
    user = (
        f"目标能力路径：{' → '.join(question_context['capability_path'])}\n"
        f"目标技能：{skill}\n目标难度：{difficulty}\n"
        f"问题蓝图：{json.dumps(v22_blueprint_prompt_payload(blueprint), ensure_ascii=False)}\n"
        f"最近对话：{json.dumps(recent_dialogue, ensure_ascii=False)}\n"
        f"硬约束：{V22_SHARED_VALIDITY_GUARD}\n"
        f"{variant_instruction}\n只输出最终问题。"
    )
    return client.chat(model, system, user, temperature=0.30, max_tokens=300).strip()


def generate_v22_blueprint_question(
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
    """V2.2 blueprint -> atomic question -> independent blind evaluation."""
    attempts = []
    rejected = []
    maximum_attempts = max_regenerations + 1 if regenerate_low_quality else 1
    for attempt_index in range(maximum_attempts):
        blueprint = build_v22_question_blueprint(
            client, question_model, skill, difficulty, dialogue_history,
            rejected, question_context,
        )
        question = generate_v22_question_from_blueprint(
            client, question_model, skill, difficulty, dialogue_history,
            question_context, blueprint,
        )
        quality_scores, judge_details = evaluate_qwen_question(
            client, question_judge_model, skill, difficulty, question,
            dialogue_history, judge_repeats, question_context,
        )
        passes_quality = (
            quality_scores["overall_question_quality"] >= quality_threshold
            and quality_scores["technical_correctness"] >= 7.0
            and quality_scores["general_question_quality"] >= 7.0
        )
        attempts.append({
            "attempt": attempt_index,
            "question_blueprint": v22_blueprint_prompt_payload(blueprint),
            "blueprint_model_raw_output": blueprint.get("raw", ""),
            "blueprint_format_retry_count": blueprint.get("format_retry_count", 0),
            "blueprint_invalid_raw_outputs": blueprint.get("invalid_raw_outputs", []),
            "question": question,
            "question_model_raw_output": question,
            "quality_scores": quality_scores,
            "question_judge_details": judge_details,
            "generation_pipeline": "atomic_blueprint_then_answerability_guarded_question",
            "technical_validity_guard": True,
            "strategy_leakage_flags": v21_question_leakage_flags(question),
            "explicit_retry_gate_passed": passes_quality,
        })
        if not regenerate_low_quality or passes_quality:
            break
        failed_metrics = [
            name for name, passed in (
                (
                    "overall_question_quality",
                    quality_scores["overall_question_quality"] >= quality_threshold,
                ),
                (
                    "technical_correctness",
                    quality_scores["technical_correctness"] >= 7.0,
                ),
                (
                    "general_question_quality",
                    quality_scores["general_question_quality"] >= 7.0,
                ),
            )
            if not passed
        ]
        rejected.append({
            "question": question,
            "feedback": (
                f"未通过显式重试门槛：{', '.join(failed_metrics)}；"
                + "; ".join(
                    row["reason"] for row in judge_details if row["reason"]
                )
            )[:1000],
        })
    accepted = attempts[-1]
    return (
        accepted["question"], accepted["quality_scores"],
        len(attempts) - 1, attempts,
    )


# V2.3 fixes a design flaw exposed by the V2.2 paired records: Plain and Tree
# independently planned their technical content, so the treatment changed both
# diagnostic context and factual premises.  V2.3 plans one neutral scaffold per
# matched pair, then gives each variant one budget-matched realization call.
V23_SHARED_VALIDITY_GUARD = (
    "所有探针必须仅考查目标技能，并且由同一组题面事实支持。不得把一种存在合理替代"
    "方案的工程实践写成唯一正确答案；若答案取决于目标、负载、操作类型、版本、配置"
    "或环境，必须明确给出决定性条件，或要求给出一种可行方案及其核心权衡。最终问题"
    "必须实际写出回答所需事实，禁止使用‘根据题面提供的稳定事实’等未呈现事实的元"
    "表述。每个探针只包含一个任务和一个核心评分点。"
)


def parse_v23_scaffold_json(raw: str) -> Dict[str, object]:
    data = _json_object_from_text(raw, "V2.3 shared technical scaffold")
    core_concept = str(data["core_concept"]).strip()[:400]
    if not core_concept:
        raise ValueError("V2.3 scaffold core_concept cannot be empty")
    facts = [
        str(item).strip()[:280] for item in data["stable_facts"]
        if str(item).strip()
    ]
    if not 1 <= len(facts) <= 3:
        raise ValueError("V2.3 scaffold requires 1-3 stable_facts")
    assumptions = [
        str(item).strip()[:260] for item in data["assumptions_to_avoid"]
        if str(item).strip()
    ]
    if not assumptions:
        raise ValueError("V2.3 scaffold requires assumptions_to_avoid")
    raw_options = data["probe_options"]
    if not isinstance(raw_options, list) or not 2 <= len(raw_options) <= 3:
        raise ValueError("V2.3 scaffold requires 2-3 probe_options")
    options = []
    seen_ids = set()
    for raw_option in raw_options:
        if not isinstance(raw_option, dict):
            raise ValueError("Each V2.3 probe option must be a JSON object")
        probe_id = str(raw_option["probe_id"]).strip().upper()[:20]
        task = str(raw_option["single_task"]).strip()[:500]
        answer = str(raw_option["answer_outline"]).strip()[:700]
        indices = raw_option["required_fact_indices"]
        if (
            not probe_id or probe_id in seen_ids or not task or not answer
            or not isinstance(indices, list) or not indices
        ):
            raise ValueError("Invalid V2.3 probe option")
        parsed_indices = [int(value) for value in indices]
        if any(value < 1 or value > len(facts) for value in parsed_indices):
            raise ValueError("V2.3 required_fact_indices reference missing facts")
        seen_ids.add(probe_id)
        options.append({
            "probe_id": probe_id,
            "single_task": task,
            "answer_outline": answer,
            "required_fact_indices": sorted(set(parsed_indices)),
        })
    return {
        "core_concept": core_concept,
        "stable_facts": facts,
        "probe_options": options,
        "assumptions_to_avoid": assumptions[:4],
    }


def v23_scaffold_payload(scaffold: Mapping) -> Dict[str, object]:
    return {
        key: scaffold[key]
        for key in (
            "core_concept", "stable_facts", "probe_options",
            "assumptions_to_avoid",
        )
    }


def v23_same_branch_history(
    dialogue_history: Sequence[Mapping], skill: str,
) -> List[Mapping]:
    """Return only history whose target belongs to the current capability branch."""
    target_branch = SKILL_TO_BRANCH[skill]
    return [
        row for row in dialogue_history
        if str(row.get("skill", "")) in SKILL_TO_BRANCH
        and SKILL_TO_BRANCH[str(row["skill"])] == target_branch
    ][-2:]


def build_v23_shared_scaffold(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
) -> Dict[str, object]:
    """Create one neutral factual scaffold shared by Plain and Tree."""
    system = (
        "你是V2.3技术脚手架规划器。规划一组中性、可核验的技术事实和2到3个原子"
        "探针，不生成最终问题，不使用候选人后验、能力假设或策略信息。只输出严格"
        "JSON，不得输出Markdown。字段必须是core_concept字符串、stable_facts字符串"
        "数组、probe_options对象数组和assumptions_to_avoid字符串数组。每个"
        "probe_options对象必须包含probe_id、single_task、answer_outline和"
        "required_fact_indices；required_fact_indices使用从1开始的事实序号。"
        + V23_SHARED_VALIDITY_GUARD
    )
    difficulty_guidance = {
        "easy": "核心概念或一个常见边界；不能仅靠猜测关键词",
        "medium": "明确场景中的一次机制推演、判断或主要权衡；不能只问定义",
        "hard": "明确边界或故障条件下的一次深入推演；不能扩展为多部分系统设计",
    }[difficulty]
    previous_questions = [
        {
            "skill": str(row.get("skill", "")),
            "question": str(row.get("question", ""))[:280],
        }
        for row in dialogue_history[-6:]
    ]
    user = (
        f"目标技能：{skill}\n目标难度：{difficulty}（{difficulty_guidance}）\n"
        f"历史问题仅用于避免重复，不得复制其中的事实或答案："
        f"{json.dumps(previous_questions, ensure_ascii=False)}\n"
        f"硬约束：{V23_SHARED_VALIDITY_GUARD}\n"
        "stable_facts必须是1到3项能够直接写入最终问题的具体条件。所有probe_options"
        "必须属于同一core_concept并达到相同难度。若目标是算法或数据结构选择，必须"
        "明确操作类型、优化目标以及会改变答案的关键约束；若存在多种合理技术方案，"
        "single_task必须要求说明一种可行方案与一个核心权衡，不能询问唯一最佳方案。"
    )
    return request_judge_json(
        client=client,
        model=model,
        system=system,
        user=user,
        parser=parse_v23_scaffold_json,
        max_tokens=900,
        temperature=0.20,
    )


def parse_v23_realization_json(raw: str, allowed_probe_ids: Sequence[str]) -> Dict:
    data = _json_object_from_text(raw, "V2.3 question realization")
    probe_id = str(data["probe_id"]).strip().upper()
    question = str(data["question"]).strip()[:1000]
    answerability = str(data["answerability_check"]).strip()[:700]
    if probe_id not in set(allowed_probe_ids):
        raise ValueError(f"V2.3 realization selected unknown probe {probe_id!r}")
    if not question or not answerability:
        raise ValueError("V2.3 realization question/check cannot be empty")
    if "根据题面提供的稳定事实" in question or "根据上述稳定事实" in question:
        raise ValueError("V2.3 question contains a forbidden missing-facts meta phrase")
    return {
        "probe_id": probe_id,
        "question": question,
        "answerability_check": answerability,
    }


def realize_v23_question(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    question_context: Mapping,
    shared_scaffold: Mapping,
    rejected_attempts: Sequence[Mapping],
) -> Dict[str, object]:
    """Select and realize one supported probe without changing shared facts."""
    allowed_ids = [
        str(option["probe_id"]) for option in shared_scaffold["probe_options"]
    ]
    system = (
        "你是V2.3技术面试问题生成器。只能从共享技术脚手架中选择一个probe_id并"
        "实现为一个中文问题，不得新增、改变或省略该探针回答所需的事实，不得引入"
        "脚手架之外的技术结论。只输出严格JSON，包含probe_id、question和"
        "answerability_check。question最多两句话、一个问号、一个主要任务，不输出"
        "答案或策略信息；answerability_check简短说明问题如何由所选探针和事实支持。"
        + V23_SHARED_VALIDITY_GUARD
    )
    if question_context["prompt_variant"] == "tree_v23":
        card = question_context["diagnostic_contrast_card"]
        relevant_history = question_context.get("same_branch_history", [])
        variant_instruction = (
            "从已有probe_options中选择最能区分H0/H1的一项；只允许改变探针选择和"
            "自然措辞，绝不允许增加事实或把H1结论写成题面前提。若没有明显更合适的"
            "探针，选择技术上最清晰、条件最充分的一项。只有下列同分支历史可以用于"
            "自然承接；若为空，不得引用其他技能的历史。\n"
            f"H0：{card['lower_hypothesis']}\nH1：{card['upper_hypothesis']}\n"
            f"需要的区分证据：{card['required_discriminating_evidence']}\n"
            f"同分支历史：{json.dumps(relevant_history, ensure_ascii=False)}"
        )
    else:
        variant_instruction = (
            "不使用H0/H1或后验信息，从已有probe_options中选择最具代表性、技术条件"
            "最充分的一项，并生成普通技术面试问题。"
        )
    rejected = [
        {
            "question": str(row.get("question", ""))[:320],
            "feedback": str(row.get("feedback", ""))[:600],
        }
        for row in rejected_attempts[-2:]
    ]
    return request_judge_json(
        client=client,
        model=model,
        system=system,
        user=(
            f"目标技能：{skill}\n目标难度：{difficulty}\n"
            f"共享技术脚手架："
            f"{json.dumps(v23_scaffold_payload(shared_scaffold), ensure_ascii=False)}\n"
            f"{variant_instruction}\n"
            f"显式重试反馈：{json.dumps(rejected, ensure_ascii=False)}"
        ),
        parser=lambda raw: parse_v23_realization_json(raw, allowed_ids),
        max_tokens=500,
        temperature=0.15,
    )


def generate_v23_shared_scaffold_question(
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
    shared_scaffold: Mapping,
) -> Tuple[str, Dict[str, float], int, List[Dict]]:
    attempts = []
    rejected = []
    maximum_attempts = max_regenerations + 1 if regenerate_low_quality else 1
    for attempt_index in range(maximum_attempts):
        realization = realize_v23_question(
            client, question_model, skill, difficulty, question_context,
            shared_scaffold, rejected,
        )
        question = str(realization["question"])
        quality_scores, judge_details = evaluate_qwen_question(
            client, question_judge_model, skill, difficulty, question,
            dialogue_history, judge_repeats, question_context,
        )
        passes_quality = (
            quality_scores["overall_question_quality"] >= quality_threshold
            and quality_scores["technical_correctness"] >= 7.0
            and quality_scores["general_question_quality"] >= 7.0
        )
        attempts.append({
            "attempt": attempt_index,
            "shared_technical_scaffold": v23_scaffold_payload(shared_scaffold),
            "shared_scaffold_model_raw_output": shared_scaffold.get("raw", ""),
            "selected_probe_id": realization["probe_id"],
            "answerability_check": realization["answerability_check"],
            "realization_model_raw_output": realization.get("raw", ""),
            "realization_format_retry_count": realization.get(
                "format_retry_count", 0
            ),
            "question": question,
            "question_model_raw_output": realization.get("raw", ""),
            "quality_scores": quality_scores,
            "question_judge_details": judge_details,
            "generation_pipeline": "shared_scaffold_then_budget_matched_realization",
            "technical_validity_guard": True,
            "strategy_leakage_flags": v21_question_leakage_flags(question),
            "explicit_retry_gate_passed": passes_quality,
        })
        if not regenerate_low_quality or passes_quality:
            break
        failed_metrics = [
            name for name, passed in (
                (
                    "overall_question_quality",
                    quality_scores["overall_question_quality"] >= quality_threshold,
                ),
                (
                    "technical_correctness",
                    quality_scores["technical_correctness"] >= 7.0,
                ),
                (
                    "general_question_quality",
                    quality_scores["general_question_quality"] >= 7.0,
                ),
            )
            if not passed
        ]
        rejected.append({
            "question": question,
            "feedback": (
                f"未通过显式重试门槛：{', '.join(failed_metrics)}；"
                + "; ".join(
                    row["reason"] for row in judge_details if row["reason"]
                )
            )[:1000],
        })
    accepted = attempts[-1]
    return (
        accepted["question"], accepted["quality_scores"],
        len(attempts) - 1, attempts,
    )


# V2.3 kept both arms inside a neutral probe menu created before H0/H1 was
# available.  That protected validity but also removed most of the treatment:
# Tree could select only a generic probe already available to Plain.  V2.4 keeps
# the factual scenario identical and lets each arm design one atomic task inside
# that boundary.  Thus the treatment is diagnostic task selection, not facts.
V24_SHARED_FACT_GUARD = (
    "共享场景必须仅涉及目标技能，使用稳定、可核验且足以作答的事实。不得把存在合理"
    "替代方案的工程实践写成普遍唯一正确答案；若结论依赖负载、操作类型、版本、配置"
    "或故障假设，必须在scenario_text中明确决定性条件。场景不得包含候选人能力、"
    "H0/H1、后验、Tree或BRIDGE信息，也不得暗示预期答案。"
)


def parse_v24_scaffold_json(raw: str) -> Dict[str, object]:
    data = _json_object_from_text(raw, "V2.4 shared factual scenario")
    core_concept = str(data["core_concept"]).strip()[:400]
    scenario_text = str(data["scenario_text"]).strip()[:650]
    facts = [
        str(item).strip()[:300] for item in data["stable_facts"]
        if str(item).strip()
    ]
    answerable_scope = str(data["answerable_scope"]).strip()[:500]
    assumptions = [
        str(item).strip()[:260] for item in data["assumptions_to_avoid"]
        if str(item).strip()
    ]
    if not core_concept or not scenario_text or not answerable_scope:
        raise ValueError("V2.4 scaffold text fields cannot be empty")
    if not 1 <= len(facts) <= 4:
        raise ValueError("V2.4 scaffold requires 1-4 stable_facts")
    if not assumptions:
        raise ValueError("V2.4 scaffold requires assumptions_to_avoid")
    if "?" in scenario_text or "？" in scenario_text:
        raise ValueError("V2.4 scenario_text must contain facts, not a question")
    if v21_question_leakage_flags(scenario_text):
        raise ValueError("V2.4 scenario_text leaks strategy information")
    return {
        "core_concept": core_concept,
        "scenario_text": scenario_text.rstrip("。！？?"),
        "stable_facts": facts,
        "answerable_scope": answerable_scope,
        "assumptions_to_avoid": assumptions[:4],
    }


def v24_scaffold_payload(scaffold: Mapping) -> Dict[str, object]:
    return {
        key: scaffold[key]
        for key in (
            "core_concept", "scenario_text", "stable_facts",
            "answerable_scope", "assumptions_to_avoid",
        )
    }


def v24_exact_skill_history(
    dialogue_history: Sequence[Mapping], skill: str,
) -> List[Mapping]:
    """Only the same leaf skill can supply an evidence gap for continuation."""
    return [
        row for row in dialogue_history if str(row.get("skill", "")) == skill
    ][-1:]


def build_v24_shared_scaffold(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    dialogue_history: Sequence[Mapping],
) -> Dict[str, object]:
    """Create one fact-only scenario shared verbatim by Plain and Tree."""
    system = (
        "你是V2.4共享技术场景规划器。只规划事实，不写问题、不选择考查动作。只输出"
        "严格JSON，不得输出Markdown。字段为core_concept字符串、scenario_text字符串、"
        "stable_facts字符串数组、answerable_scope字符串和assumptions_to_avoid字符串数组。"
        + V24_SHARED_FACT_GUARD
    )
    difficulty_guidance = {
        "easy": "给出一个常见且具体的机制或边界，不能靠猜关键词",
        "medium": "给出足够支持一次机制推演、诊断或主要权衡的具体条件",
        "hard": "给出足够支持一次深入故障或边界推演的条件，但不扩展为系统设计题",
    }[difficulty]
    previous_questions = [
        {"skill": str(row.get("skill", "")), "question": str(row.get("question", ""))[:260]}
        for row in dialogue_history[-6:]
    ]
    user = (
        f"目标技能：{skill}\n目标难度：{difficulty}（{difficulty_guidance}）\n"
        f"该能力分支适合观察的技术行为："
        f"{BRANCH_DIAGNOSTIC_LENSES[SKILL_TO_BRANCH[skill]]}\n"
        f"历史问题仅用于避免重复主题：{json.dumps(previous_questions, ensure_ascii=False)}\n"
        f"硬约束：{V24_SHARED_FACT_GUARD}\n"
        "scenario_text必须是一到两句可直接逐字放在两组最终题目前面的中性题面，完整"
        "呈现stable_facts；不得提出问题。事实既要支持典型基础探针，也要支持基于因果"
        "机制、故障边界或核心权衡的单一探针。answerable_scope说明在这些事实下允许得出"
        "什么结论以及哪些结论仍不能唯一确定。"
    )
    return request_judge_json(
        client, model, system, user, parse_v24_scaffold_json,
        max_tokens=800, temperature=0.20,
    )


def parse_v24_probe_json(
    raw: str, fact_count: int,
) -> Dict[str, object]:
    data = _json_object_from_text(raw, "V2.4 probe")
    task = str(data["single_task"]).strip()[:380]
    answer = str(data["answer_outline"]).strip()[:800]
    evidence = str(data["evidence_target"]).strip()[:600]
    answerability = str(data["answerability_check"]).strip()[:700]
    indices = data["required_fact_indices"]
    if not task or not answer or not evidence or not answerability:
        raise ValueError("V2.4 probe text fields cannot be empty")
    if not isinstance(indices, list):
        raise ValueError("V2.4 required_fact_indices must be a list")
    parsed_indices = []
    invalid_indices = [] if indices else ["<empty>"]
    for value in indices:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            invalid_indices.append(str(value)[:80])
            continue
        if 1 <= parsed <= fact_count:
            parsed_indices.append(parsed)
        else:
            invalid_indices.append(parsed)
    parsed_indices = sorted(set(parsed_indices))
    if not parsed_indices:
        # required_fact_indices is audit metadata rather than question content.
        # A model indexing mistake must not discard or stop an otherwise valid
        # generated question; the shared scenario already contains every fact.
        parsed_indices = list(range(1, fact_count + 1))
    if task.count("?") + task.count("？") != 1:
        raise ValueError("V2.4 single_task must contain exactly one question mark")
    if "\n" in task:
        raise ValueError("V2.4 single_task must be one compact sentence")
    if v21_question_leakage_flags(task):
        raise ValueError("V2.4 task leaks strategy information")
    return {
        "single_task": task,
        "answer_outline": answer,
        "required_fact_indices": parsed_indices,
        "required_fact_indices_repaired": bool(invalid_indices),
        "invalid_required_fact_indices": invalid_indices,
        "evidence_target": evidence,
        "answerability_check": answerability,
    }


def design_v24_probe(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    question_context: Mapping,
    shared_scaffold: Mapping,
    rejected_attempts: Sequence[Mapping],
) -> Dict[str, object]:
    """Design one variant-specific task without changing the shared scenario."""
    system = (
        "你是V2.4技术面试探针设计器。共享scenario_text将由程序逐字放在问题开头；"
        "你只能设计紧随其后的一个原子任务，不能添加、修改、重复或省略场景事实。只"
        "输出严格JSON：single_task、answer_outline、required_fact_indices、"
        "evidence_target、answerability_check。single_task必须是一个问句、恰好一个问号、"
        "只要求一个作答动作；answer_outline必须完全由共享事实支持。"
        + V23_SHARED_VALIDITY_GUARD
    )
    if question_context["prompt_variant"] == "tree_v24":
        card = question_context["diagnostic_contrast_card"]
        exact_history = question_context.get("exact_skill_history", [])
        history_instruction = (
            "没有同一叶子技能的历史证据，不得声称承接上一轮。"
            if not exact_history else (
                "仅当无需引入历史中的新事实时，可针对同一技能上次回答的一个未验证边界"
                f"设计任务；历史：{json.dumps(exact_history, ensure_ascii=False)}"
            )
        )
        variant_instruction = (
            "设计能够直接诱发一项可观察区分证据的任务。较低状态可以给出表面合理但缺少"
            "关键机制/边界的回答，较高状态必须通过一次因果推演、故障定位、约束下决策或"
            "核心权衡展示证据。不要问定义，不要把高状态结论写进题面，也不要同时要求"
            "‘判断并解释并给方案’等多个动作；把证据压缩进一个主要作答动作。\n"
            f"较低状态：{card['lower_hypothesis']}\n"
            f"较高状态：{card['upper_hypothesis']}\n"
            f"必须诱发的证据：{card['required_discriminating_evidence']}\n"
            f"{history_instruction}"
        )
    else:
        variant_instruction = (
            "不使用候选人后验或能力假设，设计该技能与难度下最具代表性的普通技术面试"
            "任务。任务仍须具体、可回答且只有一个作答动作。"
        )
    rejected = [
        {
            "question": str(row.get("question", ""))[:320],
            "feedback": str(row.get("feedback", ""))[:600],
        }
        for row in rejected_attempts[-2:]
    ]
    return request_judge_json(
        client=client,
        model=model,
        system=system,
        user=(
            f"目标技能：{skill}\n目标难度：{difficulty}\n"
            f"共享事实场景：{json.dumps(v24_scaffold_payload(shared_scaffold), ensure_ascii=False)}\n"
            f"{variant_instruction}\n"
            "single_task不得复述scenario_text，只能提出一个问题；required_fact_indices"
            "列出作答实际依赖的事实。answerability_check需确认答案未使用共享场景外假设。\n"
            f"显式重试反馈：{json.dumps(rejected, ensure_ascii=False)}"
        ),
        parser=lambda raw: parse_v24_probe_json(
            raw, len(shared_scaffold["stable_facts"])
        ),
        max_tokens=650,
        temperature=0.18,
    )


def generate_v24_shared_fact_question(
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
    shared_scaffold: Mapping,
) -> Tuple[str, Dict[str, float], int, List[Dict]]:
    attempts = []
    rejected = []
    maximum_attempts = max_regenerations + 1 if regenerate_low_quality else 1
    for attempt_index in range(maximum_attempts):
        probe = design_v24_probe(
            client, question_model, skill, difficulty, question_context,
            shared_scaffold, rejected,
        )
        question = f"{shared_scaffold['scenario_text']}。{probe['single_task']}"
        quality_scores, judge_details = evaluate_qwen_question(
            client, question_judge_model, skill, difficulty, question,
            dialogue_history, judge_repeats, question_context,
        )
        passes_quality = (
            quality_scores["overall_question_quality"] >= quality_threshold
            and quality_scores["technical_correctness"] >= 7.0
            and quality_scores["general_question_quality"] >= 7.0
        )
        attempts.append({
            "attempt": attempt_index,
            "shared_factual_scaffold": v24_scaffold_payload(shared_scaffold),
            "shared_scaffold_model_raw_output": shared_scaffold.get("raw", ""),
            "single_task": probe["single_task"],
            "answer_outline": probe["answer_outline"],
            "required_fact_indices": probe["required_fact_indices"],
            "required_fact_indices_repaired": probe[
                "required_fact_indices_repaired"
            ],
            "invalid_required_fact_indices": probe[
                "invalid_required_fact_indices"
            ],
            "evidence_target": probe["evidence_target"],
            "answerability_check": probe["answerability_check"],
            "probe_model_raw_output": probe.get("raw", ""),
            "probe_format_retry_count": probe.get("format_retry_count", 0),
            "question": question,
            "question_model_raw_output": probe.get("raw", ""),
            "quality_scores": quality_scores,
            "question_judge_details": judge_details,
            "generation_pipeline": "shared_facts_then_variant_specific_atomic_probe",
            "technical_validity_guard": True,
            "strategy_leakage_flags": v21_question_leakage_flags(question),
            "explicit_retry_gate_passed": passes_quality,
        })
        if not regenerate_low_quality or passes_quality:
            break
        failed_metrics = [
            name for name, passed in (
                ("overall_question_quality", quality_scores["overall_question_quality"] >= quality_threshold),
                ("technical_correctness", quality_scores["technical_correctness"] >= 7.0),
                ("general_question_quality", quality_scores["general_question_quality"] >= 7.0),
            ) if not passed
        ]
        rejected.append({
            "question": question,
            "feedback": (
                f"未通过显式重试门槛：{', '.join(failed_metrics)}；"
                + "; ".join(row["reason"] for row in judge_details if row["reason"])
            )[:1000],
        })
    accepted = attempts[-1]
    return (
        accepted["question"], accepted["quality_scores"],
        len(attempts) - 1, attempts,
    )


def parse_counterfactual_answers_json(raw: str) -> Dict[str, str]:
    data = _json_object_from_text(raw, "Counterfactual answer")
    lower = str(data["lower_answer"]).strip()
    upper = str(data["upper_answer"]).strip()
    if not lower or not upper:
        raise ValueError("Counterfactual answers cannot be empty")
    return {"lower_answer": lower[:1600], "upper_answer": upper[:1600]}


def parse_discrimination_judge_json(raw: str) -> Dict[str, object]:
    data = _json_object_from_text(raw, "Discrimination judge")
    answer_a_state = str(data["answer_a_state"]).strip().upper()
    answer_b_state = str(data["answer_b_state"]).strip().upper()
    if answer_a_state not in ("H0", "H1") or answer_b_state not in ("H0", "H1"):
        raise ValueError("answer_a_state and answer_b_state must be H0 or H1")
    diagnosticity = float(data["question_diagnosticity"])
    confidence = float(data.get("confidence", 3.0))
    if not 1.0 <= diagnosticity <= 10.0:
        raise ValueError("question_diagnosticity must be within [1,10]")
    if not 1.0 <= confidence <= 5.0:
        raise ValueError("confidence must be within [1,5]")
    return {
        "answer_a_state": answer_a_state,
        "answer_b_state": answer_b_state,
        "question_diagnosticity": diagnosticity,
        "confidence": confidence,
        "reason": str(data.get("reason", ""))[:500],
    }


def evaluate_counterfactual_discrimination(
    client: QwenClient,
    answer_model: str,
    judge_model: str,
    skill: str,
    difficulty: str,
    question: str,
    contrast_card: Mapping,
    repeats: int,
    pair_id: str,
    variant: str,
    responsive_only: bool = False,
) -> Dict[str, object]:
    """Test whether answers elicited by a question reveal adjacent H0/H1 states."""
    if repeats < 1:
        return {"available": False, "reason": "Discrimination repeats disabled"}
    answer_system = (
        "你生成一对受控的模拟候选人回答，用于评价问题的诊断区分度。"
        "两个回答必须针对同一问题、长度大致相当且语言自然；低状态回答严格受H0限制，"
        "高状态回答严格受H1限制。不要在回答中出现H0、H1、能力档位或模拟标签。"
        + (
            "两者都只能回答问题明确要求的任务，不得主动展示问题没有诱发的知识或"
            "为了拉开差距而额外补充无关机制；若问题没有诱发状态差异，两份回答允许"
            "同样缺少该差异。"
            if responsive_only else ""
        )
        + "只输出JSON：lower_answer和upper_answer。"
    )
    answer_user = (
        f"目标技能：{skill}\n目标难度：{difficulty}\n问题：{question}\n"
        f"较低状态：{contrast_card['lower_hypothesis']}\n"
        f"较高状态：{contrast_card['upper_hypothesis']}\n"
        + (
            "严格按问题本身作答，不要参考评审希望观察的证据。"
            if responsive_only else
            f"需要观察的证据：{contrast_card['required_discriminating_evidence']}"
        )
    )
    answer_pair = request_judge_json(
        client, answer_model, answer_system, answer_user,
        parse_counterfactual_answers_json, max_tokens=1000, temperature=0.65,
    )
    lower_answer = str(answer_pair["lower_answer"])
    upper_answer = str(answer_pair["upper_answer"])
    judge_system = (
        "你是盲评诊断区分度评审器。根据问题以及两个相邻能力状态的定义，"
        "独立判断匿名回答A和B分别更符合H0还是H1，并评价该问题使能力差异显现的程度。"
        "不得根据回答长度或文风判断。只输出JSON：answer_a_state与answer_b_state只能是"
        "H0或H1，question_diagnosticity为1到10，confidence为1到5，reason为简短理由。"
    )
    details = []
    base_lower_is_a = stable_seed("v21-discrimination-order", pair_id, variant) % 2 == 0
    for repeat in range(repeats):
        lower_is_a = base_lower_is_a if repeat % 2 == 0 else not base_lower_is_a
        answer_a = lower_answer if lower_is_a else upper_answer
        answer_b = upper_answer if lower_is_a else lower_answer
        judge_user = (
            f"目标技能：{skill}\n目标难度：{difficulty}\n问题：{question}\n"
            f"H0定义：{contrast_card['lower_hypothesis']}\n"
            f"H1定义：{contrast_card['upper_hypothesis']}\n"
            f"回答A：{answer_a}\n回答B：{answer_b}"
        )
        judged = request_judge_json(
            client, judge_model, judge_system, judge_user,
            parse_discrimination_judge_json, max_tokens=500,
        )
        truth_a = "H0" if lower_is_a else "H1"
        truth_b = "H1" if lower_is_a else "H0"
        correct_count = int(judged["answer_a_state"] == truth_a) + int(
            judged["answer_b_state"] == truth_b
        )
        judged.update({
            "repeat": repeat,
            "lower_answer_display_position": "A" if lower_is_a else "B",
            "predicted_lower_answer_state": (
                judged["answer_a_state"] if lower_is_a else judged["answer_b_state"]
            ),
            "predicted_upper_answer_state": (
                judged["answer_b_state"] if lower_is_a else judged["answer_a_state"]
            ),
            "state_accuracy": correct_count / 2.0,
            "both_states_correct": correct_count == 2,
        })
        details.append(judged)
    return {
        "available": True,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "lower_answer": lower_answer,
        "upper_answer": upper_answer,
        "answer_generation_raw_output": answer_pair.get("raw", ""),
        "answer_generation_format_retry_count": answer_pair.get("format_retry_count", 0),
        "judge_repeats": repeats,
        "responsive_only": responsive_only,
        "state_classification_accuracy": quality_mean([
            float(row["state_accuracy"]) for row in details
        ]),
        "both_states_correct_rate": quality_mean([
            float(row["both_states_correct"]) for row in details
        ]),
        "mean_question_diagnosticity": quality_mean([
            float(row["question_diagnosticity"]) for row in details
        ]),
        "mean_confidence": quality_mean([
            float(row["confidence"]) for row in details
        ]),
        "details": details,
    }


def parse_pairwise_judge_json(raw: str) -> Dict[str, object]:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Pairwise judge output does not contain a JSON object")
    data = json.loads(cleaned[start:end + 1])
    winner = str(data["winner"]).strip().upper()
    if winner not in ("A", "B", "TIE"):
        raise ValueError(f"Pairwise winner must be A, B, or TIE, got {winner!r}")
    confidence = float(data.get("confidence", 3.0))
    if not 1.0 <= confidence <= 5.0:
        raise ValueError("Pairwise confidence must be within [1,5]")
    return {
        "winner": winner,
        "confidence": confidence,
        "reason": str(data.get("reason", ""))[:500],
    }


def evaluate_question_pair(
    client: QwenClient,
    model: str,
    skill: str,
    difficulty: str,
    tree_question: str,
    plain_question: str,
    dialogue_history: Sequence[Mapping],
    question_context: Mapping,
    repeats: int,
    pair_id: str,
    tree_variant: str = "prompt_tree_v2",
    plain_variant: str = "prompt_plain_v2",
    strict_technical_gate: bool = False,
) -> Dict[str, object]:
    """Blind, order-randomized pairwise preference evaluation."""
    if repeats < 1:
        raise ValueError("pairwise judge repeats must be at least one")
    blind_context = {
        key: value for key, value in question_context.items()
        if key not in ("prompt_variant", "propagation")
    }
    if strict_technical_gate:
        system = (
            "你是严格的盲评技术面试问题比较器。先执行技术安全门：逐题检查是否聚焦"
            "目标技能、是否包含回答所需条件、是否存在错误或无根据的唯一最优前提、"
            "是否只有一个任务。若一题未通过安全门，它不能仅凭诊断性获胜；若两题均"
            "未通过则判TIE。只有两题均通过后，才比较对当前相邻能力状态的诊断价值、"
            "自适应相关性、清晰度和非重复性。不得臆造问题与历史之间不存在的联系，"
            "reason必须指出问题中的具体证据。你不知道问题来自哪种策略。只输出JSON："
            "winner必须为A、B或TIE，confidence为1到5，reason为简短理由。"
        )
    else:
        system = (
            "你是盲评技术面试问题比较器。比较两个问题在当前能力状态下的自适应诊断"
            "质量，优先判断能否区分给定的较低/较高能力假设及是否承接历史证据；同时"
            "要求技术正确、清晰且只有一个主要任务。你不知道问题来自哪种策略。只输出"
            "JSON：winner必须为A、B或TIE，confidence为1到5，reason为简短理由。"
        )
    details = []
    base_tree_is_a = stable_seed("pair-order", pair_id) % 2 == 0
    for repeat in range(repeats):
        tree_is_a = base_tree_is_a if repeat % 2 == 0 else not base_tree_is_a
        question_a = tree_question if tree_is_a else plain_question
        question_b = plain_question if tree_is_a else tree_question
        user = (
            f"目标技能：{skill}\n目标难度：{difficulty}\n"
            f"诊断上下文：{json.dumps(blind_context, ensure_ascii=False)}\n"
            f"历史对话：{json.dumps(list(dialogue_history)[-3:], ensure_ascii=False)}\n"
            f"问题A：{question_a}\n问题B：{question_b}"
        )
        judged = request_judge_json(
            client, model, system, user, parse_pairwise_judge_json, max_tokens=350,
        )
        displayed_winner = str(judged["winner"])
        if displayed_winner == "TIE":
            canonical_winner = "tie"
        elif (displayed_winner == "A") == tree_is_a:
            canonical_winner = tree_variant
        else:
            canonical_winner = plain_variant
        judged.update({
            "repeat": repeat,
            "tree_display_position": "A" if tree_is_a else "B",
            "canonical_winner": canonical_winner,
        })
        details.append(judged)
    tree_wins = sum(row["canonical_winner"] == tree_variant for row in details)
    plain_wins = sum(row["canonical_winner"] == plain_variant for row in details)
    ties = len(details) - tree_wins - plain_wins
    majority = (
        tree_variant if tree_wins > plain_wins
        else plain_variant if plain_wins > tree_wins
        else "tie"
    )
    return {
        "judge_repeats": repeats,
        "tree_wins": tree_wins,
        "plain_wins": plain_wins,
        "ties": ties,
        "tree_preference_score": (tree_wins + 0.5 * ties) / repeats,
        "majority_winner": majority,
        "strict_technical_gate": strict_technical_gate,
        "details": details,
    }


def select_diagnostic_challenge_skill(
    model: BayesianAbilityModel,
    recent: Sequence[str],
) -> Tuple[str, str, Dict[str, float]]:
    """Choose an uncertainty frontier without forcing domain coverage."""
    candidates = []
    job_variance = model.job_variance()
    total_variance = model.total_variance()
    max_variance = max(model.skill_variance(skill) for skill in SKILLS)
    raw = []
    for skill in SKILLS:
        difficulty = closest_difficulty(model.skill_mean(skill))
        components = model.risk_components(
            skill, difficulty, recent, job_variance, total_variance,
        )
        siblings = CAPABILITY_TREE[SKILL_TO_CLUSTER[skill]][SKILL_TO_BRANCH[skill]]
        sibling_mean = sum(model.skill_mean(name) for name in siblings) / len(siblings)
        disagreement = min(abs(model.skill_mean(skill) - sibling_mean) / 2.0, 1.0)
        uncertainty = model.skill_variance(skill) / max(max_variance, _EPS)
        linkage = float(any(
            SKILL_TO_BRANCH.get(name) == SKILL_TO_BRANCH[skill] for name in recent
        ))
        raw.append((skill, difficulty, components, uncertainty, disagreement, linkage))
    max_utility = max(item[2]["utility"] for item in raw)
    for skill, difficulty, components, uncertainty, disagreement, linkage in raw:
        normalized_utility = components["utility"] / max(max_utility, _EPS)
        score = (
            0.45 * uncertainty + 0.25 * disagreement
            + 0.20 * normalized_utility + 0.10 * linkage
        ) / components["question_cost"]
        enriched = dict(components)
        enriched.update({
            "diagnostic_challenge_score": score,
            "challenge_uncertainty": uncertainty,
            "challenge_sibling_disagreement": disagreement,
            "challenge_history_linkage": linkage,
        })
        candidates.append((score, skill, difficulty, enriched))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, skill, difficulty, components = candidates[0]
    return skill, difficulty, components


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
    for name in (*QUESTION_QUALITY_DIMENSIONS, *COMPOSITE_QUALITY_METRICS):
        values = [float(row[name]) for row in rows] if name in COMPOSITE_QUALITY_METRICS else [
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
        "schema_version": 5,
        "capability_tree": {
            "root_nodes": 1,
            "domain_nodes": len(CLUSTERS),
            "subcapability_nodes": sum(len(value) for value in CAPABILITY_TREE.values()),
            "leaf_nodes": len(SKILLS),
        },
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
                propagation, _policy = strategy_spec(strategy)
                model = BayesianAbilityModel(propagation=propagation)
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
                        strategy,
                        model,
                        skill,
                        components,
                        grouped[(strategy, profile.candidate_id)],
                    )
                    job_variance_before = model.job_variance()
                    total_variance_before = model.total_variance()
                    hierarchy_variance_before = model.hierarchy_variance(skill)
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
                    total_variance_after = model.total_variance()
                    hierarchy_variance_after = model.hierarchy_variance(skill)
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
                        "branch": SKILL_TO_BRANCH[skill],
                        "capability_path": list(SKILL_PATHS[skill]),
                        "difficulty": difficulty,
                        "question_prompt_variant": question_context["prompt_variant"],
                        "question_context": question_context,
                        "question": question,
                        "question_quality_scores": {
                            name: quality_scores[name]
                            for name in QUESTION_QUALITY_DIMENSIONS
                        },
                        "overall_question_quality": quality_scores["overall_question_quality"],
                        "general_question_quality": quality_scores["general_question_quality"],
                        "adaptive_diagnostic_quality": quality_scores["adaptive_diagnostic_quality"],
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
                        "posterior_path": model.posterior_path(skill),
                        "posterior_job_mean": model.job_mean(),
                        "posterior_job_std": math.sqrt(model.job_variance()),
                        "job_variance_before": job_variance_before,
                        "job_variance_after": job_variance_after,
                        "realized_job_variance_reduction": (
                            job_variance_before - job_variance_after
                        ),
                        "realized_global_variance_reduction": (
                            total_variance_before - total_variance_after
                        ),
                        "realized_hierarchy_variance_reduction": (
                            hierarchy_variance_before - hierarchy_variance_after
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
                "propagation": strategy_spec(strategy)[0],
                "skill_coverage": len({row["skill"] for row in rows}) / len(SKILLS),
                "cluster_coverage": len({row["cluster"] for row in rows}) / len(CLUSTERS),
                "branch_coverage": len({row["branch"] for row in rows}) / 27,
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
            "branch_coverage": sum(row["branch_coverage"] for row in ability_rows) / len(ability_rows),
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
            "tree_shape": {"root": 1, "domains": 9, "subcapabilities": 27, "leaves": 108},
        },
        "summary": question_quality_summary,
        "ability_state_diagnostics": ability_diagnostics,
    }
    all_turn_rows = [row for group in grouped.values() for row in group]
    if {"bridge", "random"}.issubset(strategies):
        result["paired_end_to_end_quality"] = paired_question_quality_comparison(
            all_turn_rows, "bridge", "random"
        )
    elif {"tree_bridge", "tree_random"}.issubset(strategies):
        result["paired_end_to_end_quality"] = paired_question_quality_comparison(
            all_turn_rows, "tree_bridge", "tree_random"
        )
    write_csv(output_dir / "candidate_results.csv", candidate_rows)
    atomic_json(output_dir / "summary.json", result)
    run_question_quality_analysis(
        turns_path, output_dir / "quality_analysis", print_summary=False,
    )
    return result


# =============================================================================
# Fixed-target prompt ablation
# =============================================================================

def run_prompt_ablation(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    n_candidates: int,
    questions: int,
    question_judge_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    regenerate_low_quality: bool,
    seed: int,
    request_delay: float,
) -> Dict:
    """Compare plain and tree-adaptive prompts under exactly paired targets.

    At a candidate/turn unit, both variants receive the same capability path,
    difficulty, Gaussian posterior, and immutable history snapshot.  One shared
    canonical answer is added only after both questions have been judged, so a
    variant cannot change the other variant's current or future input state.
    """
    output_dir = output_root / "exp4_prompt_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.json"
    turns_path = output_dir / "turns.jsonl"
    setup_path = output_dir / "setup.json"
    profiles = generate_profiles(n_candidates, seed)
    variants = ("prompt_plain", "prompt_tree")
    setup = {
        "schema_version": 1,
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
        "prompt_variants": list(variants),
        "pairing": "same path, skill, difficulty, posterior, history, and candidate state",
    }
    if setup_path.exists():
        saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if saved_setup != setup:
            raise ValueError(
                "Existing exp4_prompt_ablation/setup.json does not match this run; "
                "use the original arguments or a different --output directory"
            )
    else:
        atomic_json(setup_path, setup)
    serialized_profiles = [asdict(profile) for profile in profiles]
    if profiles_path.exists():
        if json.loads(profiles_path.read_text(encoding="utf-8")) != serialized_profiles:
            raise ValueError("Existing prompt-ablation profiles do not match the requested setup")
    else:
        atomic_json(profiles_path, serialized_profiles)

    existing = load_jsonl(turns_path)
    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    seen_keys = set()
    for row in existing:
        key = (str(row["strategy"]), int(row["candidate_id"]), int(row["turn"]))
        if key in seen_keys:
            raise ValueError(f"Duplicate prompt-ablation record: {key}")
        if key[0] not in variants:
            raise ValueError(f"Unexpected prompt-ablation strategy: {key[0]}")
        seen_keys.add(key)
        grouped[(key[0], key[1])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["turn"]))
    for profile in profiles:
        counts = [len(grouped[(variant, profile.candidate_id)]) for variant in variants]
        if counts[0] != counts[1]:
            raise ValueError(
                f"Incomplete paired prompt-ablation turn for candidate {profile.candidate_id}; "
                "restore the JSONL to the last complete pair or use another output directory"
            )
        for variant in variants:
            turns = [int(row["turn"]) for row in grouped[(variant, profile.candidate_id)]]
            if turns != list(range(len(turns))) or len(turns) > questions:
                raise ValueError(f"Non-contiguous saved turns for {variant}/{profile.candidate_id}")

    client = QwenClient(api_key, base_url)
    total = n_candidates * len(variants) * questions
    with tqdm(total=total, initial=len(existing), desc="Fixed-target prompt pairs") as progress:
        for profile in profiles:
            model = BayesianAbilityModel(propagation="tree")
            per_skill_count = defaultdict(int)
            tree_rows = grouped[("prompt_tree", profile.candidate_id)]
            shared_history = []
            for row in tree_rows:
                model.update(
                    row["skill"], row["ability_observation_score"],
                    observation_std=row["ability_observation_std"],
                )
                per_skill_count[row["skill"]] += 1
                shared_history.append({
                    "turn": row["turn"],
                    "skill": row["skill"],
                    "question": row["canonical_question"],
                    "answer": row["answer"],
                })

            completed_turns = len(tree_rows)
            for turn in range(completed_turns, questions):
                target_index = stable_seed("prompt-target", seed, profile.candidate_id, turn) % len(SKILLS)
                skill = SKILLS[target_index]
                difficulty = closest_difficulty(model.skill_mean(skill))
                components = model.risk_components(skill, difficulty, [
                    row["skill"] for row in shared_history[-6:]
                ])
                base_context = build_question_context(
                    "bridge", model, skill, components, shared_history
                )
                history_hash = hashlib.sha256(
                    json.dumps(shared_history, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                pair_id = f"c{profile.candidate_id:03d}-t{turn:03d}"
                pending = []
                for variant in variants:
                    question_context = dict(base_context)
                    question_context["prompt_variant"] = (
                        "bridge_adaptive" if variant == "prompt_tree" else "baseline"
                    )
                    question, quality_scores, regeneration_count, question_attempts = (
                        generate_qualified_question(
                            client=client,
                            question_model=question_model,
                            question_judge_model=question_judge_model,
                            skill=skill,
                            difficulty=difficulty,
                            dialogue_history=shared_history,
                            judge_repeats=question_judge_repeats,
                            quality_threshold=quality_threshold,
                            max_regenerations=max_regenerations,
                            regenerate_low_quality=regenerate_low_quality,
                            question_context=question_context,
                        )
                    )
                    pending.append((variant, question_context, question, quality_scores,
                                    regeneration_count, question_attempts))

                canonical_question = next(
                    item[2] for item in pending if item[0] == "prompt_tree"
                )
                answer = generate_qwen_answer(
                    client, candidate_model, profile, skill, canonical_question
                )
                job_variance_before = model.job_variance()
                total_variance_before = model.total_variance()
                hierarchy_variance_before = model.hierarchy_variance(skill)
                ability_score, ability_observation_std = simulate_score(
                    profile=profile,
                    skill=skill,
                    difficulty=difficulty,
                    observation_index=per_skill_count[skill],
                    seed=seed,
                )
                model.update(skill, ability_score, observation_std=ability_observation_std)
                shared_history.append({
                    "turn": turn,
                    "skill": skill,
                    "question": canonical_question,
                    "answer": answer,
                })
                per_skill_count[skill] += 1

                for (variant, question_context, question, quality_scores,
                     regeneration_count, question_attempts) in pending:
                    record = {
                        "created_at": utc_now(),
                        "strategy": variant,
                        "candidate_id": profile.candidate_id,
                        "level": profile.level,
                        "turn": turn,
                        "pair_id": pair_id,
                        "history_snapshot_sha256": history_hash,
                        "skill": skill,
                        "cluster": SKILL_TO_CLUSTER[skill],
                        "branch": SKILL_TO_BRANCH[skill],
                        "capability_path": list(SKILL_PATHS[skill]),
                        "difficulty": difficulty,
                        "question_prompt_variant": question_context["prompt_variant"],
                        "question_context": question_context,
                        "question": question,
                        "question_quality_scores": {
                            name: quality_scores[name] for name in QUESTION_QUALITY_DIMENSIONS
                        },
                        "general_question_quality": quality_scores["general_question_quality"],
                        "adaptive_diagnostic_quality": quality_scores["adaptive_diagnostic_quality"],
                        "overall_question_quality": quality_scores["overall_question_quality"],
                        "quality_threshold": quality_threshold,
                        "quality_gate_enabled": regenerate_low_quality,
                        "quality_threshold_met": quality_scores["overall_question_quality"] >= quality_threshold,
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": question_attempts,
                        "canonical_question": canonical_question,
                        "answer": answer,
                        "assigned_skill_theta": profile.skill_theta[skill],
                        "ability_observation_source": "deterministic_latent_simulation",
                        "ability_observation_score": ability_score,
                        "ability_observation_std": ability_observation_std,
                        "posterior_skill_mean": model.skill_mean(skill),
                        "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                        "posterior_job_mean": model.job_mean(),
                        "posterior_job_std": math.sqrt(model.job_variance()),
                        "realized_job_variance_reduction": job_variance_before - model.job_variance(),
                        "realized_global_variance_reduction": total_variance_before - model.total_variance(),
                        "realized_hierarchy_variance_reduction": (
                            hierarchy_variance_before - model.hierarchy_variance(skill)
                        ),
                        "selector": components,
                    }
                    append_jsonl(turns_path, record)
                    grouped[(variant, profile.candidate_id)].append(record)
                    progress.update(1)
                time.sleep(max(request_delay, 0.0))

    rows = load_jsonl(turns_path)
    summary = {
        "setup": setup,
        "summary": {
            variant: summarize_question_records([
                row for row in rows if row["strategy"] == variant
            ])
            for variant in variants
        },
        "paired_prompt_comparison": paired_question_quality_comparison(
            rows, "prompt_tree", "prompt_plain"
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    run_question_quality_analysis(
        turns_path, output_dir / "quality_analysis", print_summary=False,
    )
    return summary


def flatten_v2_prompt_pairs(pairs: Sequence[Mapping]) -> List[Dict]:
    rows = []
    for pair in pairs:
        for variant, generated in pair["variants"].items():
            scores = generated["quality_scores"]
            rows.append({
                "created_at": pair["created_at"],
                "strategy": variant,
                "candidate_id": pair["candidate_id"],
                "level": pair["level"],
                "turn": pair["turn"],
                "pair_id": pair["pair_id"],
                "history_snapshot_sha256": pair["history_snapshot_sha256"],
                "skill": pair["skill"],
                "cluster": pair["cluster"],
                "branch": pair["branch"],
                "capability_path": pair["capability_path"],
                "difficulty": pair["difficulty"],
                "question_prompt_variant": generated["question_context"]["prompt_variant"],
                "question_context": generated["question_context"],
                "question": generated["question"],
                "question_quality_scores": {
                    name: scores[name] for name in QUESTION_QUALITY_DIMENSIONS
                },
                "general_question_quality": scores["general_question_quality"],
                "adaptive_diagnostic_quality": scores["adaptive_diagnostic_quality"],
                "overall_question_quality": scores["overall_question_quality"],
                "quality_threshold": pair["quality_threshold"],
                "quality_gate_enabled": pair["quality_gate_enabled"],
                "quality_threshold_met": (
                    scores["overall_question_quality"] >= pair["quality_threshold"]
                ),
                "regeneration_count": generated["regeneration_count"],
                "question_generation_attempts": generated["question_generation_attempts"],
                "canonical_question": pair["canonical_question"],
                "answer": pair["answer"],
                "assigned_skill_theta": pair["assigned_skill_theta"],
                "ability_observation_source": "deterministic_latent_simulation",
                "ability_observation_score": pair["ability_observation_score"],
                "ability_observation_std": pair["ability_observation_std"],
                "posterior_skill_mean": pair["posterior_skill_mean"],
                "posterior_skill_std": pair["posterior_skill_std"],
                "posterior_job_mean": pair["posterior_job_mean"],
                "posterior_job_std": pair["posterior_job_std"],
                "realized_job_variance_reduction": pair["realized_job_variance_reduction"],
                "realized_global_variance_reduction": pair["realized_global_variance_reduction"],
                "realized_hierarchy_variance_reduction": pair[
                    "realized_hierarchy_variance_reduction"
                ],
                "selector": pair["selector"],
                "pairwise_tree_preference_score": pair["pairwise_evaluation"][
                    "tree_preference_score"
                ],
                "pairwise_majority_winner": pair["pairwise_evaluation"][
                    "majority_winner"
                ],
            })
            discrimination = generated.get("counterfactual_discrimination", {})
            if discrimination.get("available"):
                rows[-1].update({
                    "counterfactual_state_accuracy": discrimination[
                        "state_classification_accuracy"
                    ],
                    "counterfactual_both_states_correct_rate": discrimination[
                        "both_states_correct_rate"
                    ],
                    "counterfactual_question_diagnosticity": discrimination[
                        "mean_question_diagnosticity"
                    ],
                    "counterfactual_judge_confidence": discrimination[
                        "mean_confidence"
                    ],
                })
            final_attempt = generated.get("question_generation_attempts", [])[-1:]
            rows[-1]["strategy_leakage_flags"] = (
                final_attempt[0].get("strategy_leakage_flags", [])
                if final_attempt else []
            )
    return rows


def summarize_pairwise_preferences(
    pairs: Sequence[Mapping],
    tree_variant: str = "prompt_tree_v2",
    plain_variant: str = "prompt_plain_v2",
) -> Dict[str, object]:
    if not pairs:
        return {"available": False, "reason": "No V2 prompt pairs"}
    candidate_scores: Dict[object, List[float]] = defaultdict(list)
    repeat_winners: List[str] = []
    majority_winners: List[str] = []
    confidences: List[float] = []
    order_a = 0
    per_pair_counts = []
    for pair in pairs:
        evaluation = pair["pairwise_evaluation"]
        candidate_scores[pair["candidate_id"]].append(
            float(evaluation["tree_preference_score"])
        )
        majority_winners.append(str(evaluation["majority_winner"]))
        for detail in evaluation["details"]:
            repeat_winners.append(str(detail["canonical_winner"]))
            confidences.append(float(detail["confidence"]))
            order_a += int(detail["tree_display_position"] == "A")
        per_pair_counts.append([
            sum(row["canonical_winner"] == tree_variant for row in evaluation["details"]),
            sum(row["canonical_winner"] == "tie" for row in evaluation["details"]),
            sum(row["canonical_winner"] == plain_variant for row in evaluation["details"]),
        ])
    per_candidate = [quality_mean(values) for values in candidate_scores.values()]
    centered = [value - 0.5 for value in per_candidate]
    ci_low, ci_high = bootstrap_mean_ci(
        per_candidate, seed=stable_seed("v2-pairwise-bootstrap", len(pairs))
    )
    std = quality_sample_std(centered)
    repeat_count = sum(per_pair_counts[0])
    vote_agreement = quality_mean([
        max(counts) / sum(counts) for counts in per_pair_counts
    ])
    unanimous_rate = quality_mean([
        float(max(counts) == sum(counts)) for counts in per_pair_counts
    ])
    if repeat_count > 1:
        observed_agreement = quality_mean([
            (sum(count * count for count in counts) - repeat_count)
            / (repeat_count * (repeat_count - 1))
            for counts in per_pair_counts
        ])
        category_totals = [sum(counts[index] for counts in per_pair_counts) for index in range(3)]
        total_votes = len(per_pair_counts) * repeat_count
        expected_agreement = sum((count / total_votes) ** 2 for count in category_totals)
        fleiss_kappa = (
            (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
            if expected_agreement < 1.0 - _EPS else 0.0
        )
    else:
        fleiss_kappa = None
    return {
        "available": True,
        "unit": "candidate mean across matched targets",
        "n_pairs": len(pairs),
        "n_candidates": len(per_candidate),
        "judge_repeats_per_pair": len(repeat_winners) / len(pairs),
        "tree_preference_score": quality_mean(per_candidate),
        "tree_preference_bootstrap_95_ci_low": ci_low,
        "tree_preference_bootstrap_95_ci_high": ci_high,
        "preference_effect_over_chance": quality_mean(centered),
        "paired_effect_size_dz": quality_mean(centered) / std if std > 0 else 0.0,
        "paired_sign_flip_p": paired_sign_flip_p(
            centered, seed=stable_seed("v2-pairwise-sign-flip", len(pairs))
        ),
        "repeat_vote_tree_rate": repeat_winners.count(tree_variant) / len(repeat_winners),
        "repeat_vote_tie_rate": repeat_winners.count("tie") / len(repeat_winners),
        "repeat_vote_plain_rate": repeat_winners.count(plain_variant) / len(repeat_winners),
        "pair_majority_tree_rate": majority_winners.count(tree_variant) / len(pairs),
        "pair_majority_tie_rate": majority_winners.count("tie") / len(pairs),
        "pair_majority_plain_rate": majority_winners.count(plain_variant) / len(pairs),
        "mean_judge_confidence": quality_mean(confidences),
        "tree_displayed_as_a_rate": order_a / len(repeat_winners),
        "mean_pair_vote_agreement": vote_agreement,
        "unanimous_pair_rate": unanimous_rate,
        "fleiss_kappa_three_way": fleiss_kappa,
        "note": "Question order is deterministically randomized per pair and judge repeat.",
    }


COUNTERFACTUAL_METRICS = (
    "state_classification_accuracy",
    "both_states_correct_rate",
    "mean_question_diagnosticity",
)


def summarize_counterfactual_discrimination(
    pairs: Sequence[Mapping],
    tree_variant: str,
    plain_variant: str,
) -> Dict[str, object]:
    usable = [
        pair for pair in pairs
        if all(
            pair["variants"].get(variant, {}).get(
                "counterfactual_discrimination", {}
            ).get("available")
            for variant in (tree_variant, plain_variant)
        )
    ]
    if not usable:
        return {"available": False, "reason": "Counterfactual discrimination disabled"}
    by_variant: Dict[str, Dict[str, float]] = {}
    for variant in (tree_variant, plain_variant):
        evaluations = [
            pair["variants"][variant]["counterfactual_discrimination"]
            for pair in usable
        ]
        by_variant[variant] = {
            f"{metric}_mean": quality_mean([
                float(row[metric]) for row in evaluations
            ])
            for metric in COUNTERFACTUAL_METRICS
        }
        by_variant[variant]["mean_confidence"] = quality_mean([
            float(row["mean_confidence"]) for row in evaluations
        ])
        per_question_agreement = []
        for row in evaluations:
            signatures = [
                (
                    str(detail["predicted_lower_answer_state"]),
                    str(detail["predicted_upper_answer_state"]),
                )
                for detail in row["details"]
            ]
            per_question_agreement.append(
                max(signatures.count(signature) for signature in set(signatures))
                / len(signatures)
            )
        by_variant[variant]["mean_state_assignment_agreement"] = quality_mean(
            per_question_agreement
        )
        by_variant[variant]["unanimous_state_assignment_rate"] = quality_mean([
            float(value >= 1.0 - _EPS) for value in per_question_agreement
        ])
        diagnosticity_ratings = [
            [float(detail["question_diagnosticity"]) for detail in row["details"]]
            for row in evaluations
        ]
        if diagnosticity_ratings and len(diagnosticity_ratings[0]) >= 2:
            by_variant[variant]["diagnosticity_judge_icc_1_1"] = judge_repeat_icc(
                diagnosticity_ratings
            )
            by_variant[variant]["diagnosticity_judge_icc_1_k"] = judge_repeat_icc_average(
                diagnosticity_ratings
            )
        else:
            by_variant[variant]["diagnosticity_judge_icc_1_1"] = None
            by_variant[variant]["diagnosticity_judge_icc_1_k"] = None

    candidate_values: Dict[Tuple[object, str, str], List[float]] = defaultdict(list)
    for pair in usable:
        for variant in (tree_variant, plain_variant):
            evaluation = pair["variants"][variant]["counterfactual_discrimination"]
            for metric in COUNTERFACTUAL_METRICS:
                candidate_values[(pair["candidate_id"], variant, metric)].append(
                    float(evaluation[metric])
                )
    candidate_ids = sorted({pair["candidate_id"] for pair in usable}, key=str)
    paired = {}
    for metric in COUNTERFACTUAL_METRICS:
        deltas = [
            quality_mean(candidate_values[(candidate_id, tree_variant, metric)])
            - quality_mean(candidate_values[(candidate_id, plain_variant, metric)])
            for candidate_id in candidate_ids
        ]
        low, high = quality_bootstrap_ci(
            deltas, seed=stable_seed("v21-counterfactual-bootstrap", metric)
        )
        delta_std = quality_sample_std(deltas)
        paired[metric] = {
            "mean_delta": quality_mean(deltas),
            "bootstrap_95_ci_low": low,
            "bootstrap_95_ci_high": high,
            "paired_effect_size_dz": (
                quality_mean(deltas) / delta_std if delta_std > _EPS else 0.0
            ),
            "tree_win_rate": quality_mean([float(value > _EPS) for value in deltas]),
            "tie_rate": quality_mean([float(abs(value) <= _EPS) for value in deltas]),
            "tree_loss_rate": quality_mean([float(value < -_EPS) for value in deltas]),
            "paired_sign_flip_p": paired_sign_flip_p(
                deltas, seed=stable_seed("v21-counterfactual-sign-flip", metric)
            ),
        }
    adjusted = holm_adjust({
        metric: float(values["paired_sign_flip_p"])
        for metric, values in paired.items()
    })
    for metric, value in adjusted.items():
        paired[metric]["holm_adjusted_p"] = value
    return {
        "available": True,
        "unit": "candidate mean across matched targets",
        "n_pairs": len(usable),
        "n_candidates": len(candidate_ids),
        "by_variant": by_variant,
        "paired_tree_minus_plain": paired,
        "note": (
            "H0/H1 answers are controlled simulations, not human responses. "
            "This is a mechanism check and not a substitute for human validation."
        ),
    }


def flatten_counterfactual_pairs(
    pairs: Sequence[Mapping],
) -> List[Dict[str, object]]:
    rows = []
    for pair in pairs:
        for variant, generated in pair["variants"].items():
            evaluation = generated.get("counterfactual_discrimination", {})
            if not evaluation.get("available"):
                continue
            rows.append({
                "candidate_id": pair["candidate_id"],
                "turn": pair["turn"],
                "pair_id": pair["pair_id"],
                "strategy": variant,
                "skill": pair["skill"],
                "difficulty": pair["difficulty"],
                "state_classification_accuracy": evaluation[
                    "state_classification_accuracy"
                ],
                "both_states_correct_rate": evaluation["both_states_correct_rate"],
                "mean_question_diagnosticity": evaluation[
                    "mean_question_diagnosticity"
                ],
                "mean_confidence": evaluation["mean_confidence"],
            })
    return rows


def run_prompt_ablation_v2(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    n_candidates: int,
    questions: int,
    question_judge_repeats: int,
    pairwise_judge_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    regenerate_low_quality: bool,
    seed: int,
    request_delay: float,
) -> Dict:
    """V2 fixed-target ablation with contrast cards and blind pairwise judging."""
    output_dir = output_root / "exp5_prompt_ablation_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.json"
    pairs_path = output_dir / "pairs.jsonl"
    turns_path = output_dir / "turns.jsonl"
    setup_path = output_dir / "setup.json"
    profiles = generate_profiles(n_candidates, seed)
    variants = ("prompt_plain_v2", "prompt_tree_v2")
    setup = {
        "schema_version": 1,
        "seed": seed,
        "question_model": question_model,
        "question_judge_model": question_judge_model,
        "candidate_model": candidate_model,
        "question_judge_repeats": question_judge_repeats,
        "pairwise_judge_repeats": pairwise_judge_repeats,
        "quality_threshold": quality_threshold,
        "max_regenerations": max_regenerations,
        "regenerate_low_quality": regenerate_low_quality,
        "ability_update_source": "deterministic_latent_simulation",
        "candidates": n_candidates,
        "questions": questions,
        "prompt_variants": list(variants),
        "generation_pipeline": "qwen-turbo draft then qwen-turbo budget-matched revision",
        "target_sampling": "diagnostic uncertainty frontier; identical target for both variants",
        "pairwise_order": "blind deterministic randomization per repeat",
        "primary_metric": "adaptive_diagnostic_quality",
        "general_quality_role": "non-inferiority safeguard; recommended margin -0.10",
    }
    if setup_path.exists():
        saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if saved_setup != setup:
            raise ValueError(
                "Existing V2 setup does not match this run; use the original "
                "arguments or a different --output directory"
            )
    else:
        atomic_json(setup_path, setup)
    serialized_profiles = [asdict(profile) for profile in profiles]
    if profiles_path.exists():
        if json.loads(profiles_path.read_text(encoding="utf-8")) != serialized_profiles:
            raise ValueError("Existing V2 profiles do not match the requested seed/setup")
    else:
        atomic_json(profiles_path, serialized_profiles)

    existing = load_jsonl(pairs_path)
    grouped: Dict[object, List[Dict]] = defaultdict(list)
    seen = set()
    for pair in existing:
        key = (pair["candidate_id"], pair["turn"])
        if key in seen:
            raise ValueError(f"Duplicate V2 pair record: {key}")
        seen.add(key)
        if set(pair["variants"]) != set(variants):
            raise ValueError(f"Incomplete V2 pair record: {key}")
        grouped[pair["candidate_id"]].append(pair)
    for candidate_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["turn"]))
        turns = [int(row["turn"]) for row in rows]
        if turns != list(range(len(turns))) or len(turns) > questions:
            raise ValueError(f"Non-contiguous V2 pairs for candidate {candidate_id}")

    client = QwenClient(api_key, base_url)
    total = n_candidates * questions
    with tqdm(total=total, initial=len(existing), desc="V2 fixed-target prompt pairs") as progress:
        for profile in profiles:
            model = BayesianAbilityModel(propagation="tree")
            per_skill_count: Dict[str, int] = defaultdict(int)
            history = []
            completed = grouped[profile.candidate_id]
            for pair in completed:
                model.update(
                    pair["skill"], pair["ability_observation_score"],
                    observation_std=pair["ability_observation_std"],
                )
                per_skill_count[pair["skill"]] += 1
                history.append({
                    "turn": pair["turn"],
                    "skill": pair["skill"],
                    "question": pair["canonical_question"],
                    "answer": pair["answer"],
                    "ability_observation_score": pair["ability_observation_score"],
                    "ability_observation_std": pair["ability_observation_std"],
                })

            for turn in range(len(completed), questions):
                recent = [row["skill"] for row in history[-6:]]
                skill, difficulty, components = select_diagnostic_challenge_skill(
                    model, recent,
                )
                base_context = build_question_context(
                    "bridge", model, skill, components, history,
                )
                base_context["diagnostic_contrast_card"] = build_diagnostic_contrast_card(
                    model, skill, history,
                )
                history_hash = hashlib.sha256(
                    json.dumps(history, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                pair_id = f"v2-c{profile.candidate_id:03d}-t{turn:03d}"
                generated: Dict[str, Dict] = {}
                generation_order = list(variants)
                if stable_seed("v2-generation-order", pair_id) % 2:
                    generation_order.reverse()
                for variant in generation_order:
                    context = dict(base_context)
                    context["prompt_variant"] = (
                        "tree_v2" if variant == "prompt_tree_v2" else "plain_v2"
                    )
                    question, scores, regeneration_count, attempts = (
                        generate_two_stage_question(
                            client, question_model, question_judge_model,
                            skill, difficulty, history, question_judge_repeats,
                            quality_threshold, max_regenerations,
                            regenerate_low_quality, context,
                        )
                    )
                    generated[variant] = {
                        "question": question,
                        "quality_scores": scores,
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": attempts,
                        "question_context": context,
                    }

                pairwise = evaluate_question_pair(
                    client, question_judge_model, skill, difficulty,
                    generated["prompt_tree_v2"]["question"],
                    generated["prompt_plain_v2"]["question"],
                    history, base_context, pairwise_judge_repeats, pair_id,
                )
                canonical_question = generated["prompt_tree_v2"]["question"]
                answer = generate_qwen_answer(
                    client, candidate_model, profile, skill, canonical_question,
                )
                job_before = model.job_variance()
                total_before = model.total_variance()
                hierarchy_before = model.hierarchy_variance(skill)
                ability_score, ability_std = simulate_score(
                    profile, skill, difficulty, per_skill_count[skill], seed,
                )
                model.update(skill, ability_score, observation_std=ability_std)
                per_skill_count[skill] += 1
                pair = {
                    "created_at": utc_now(),
                    "candidate_id": profile.candidate_id,
                    "level": profile.level,
                    "turn": turn,
                    "pair_id": pair_id,
                    "history_snapshot_sha256": history_hash,
                    "generation_order": generation_order,
                    "skill": skill,
                    "cluster": SKILL_TO_CLUSTER[skill],
                    "branch": SKILL_TO_BRANCH[skill],
                    "capability_path": list(SKILL_PATHS[skill]),
                    "difficulty": difficulty,
                    "quality_threshold": quality_threshold,
                    "quality_gate_enabled": regenerate_low_quality,
                    "variants": generated,
                    "pairwise_evaluation": pairwise,
                    "canonical_question": canonical_question,
                    "answer": answer,
                    "assigned_skill_theta": profile.skill_theta[skill],
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                    "posterior_skill_mean": model.skill_mean(skill),
                    "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                    "posterior_job_mean": model.job_mean(),
                    "posterior_job_std": math.sqrt(model.job_variance()),
                    "realized_job_variance_reduction": job_before - model.job_variance(),
                    "realized_global_variance_reduction": total_before - model.total_variance(),
                    "realized_hierarchy_variance_reduction": (
                        hierarchy_before - model.hierarchy_variance(skill)
                    ),
                    "selector": components,
                }
                append_jsonl(pairs_path, pair)
                grouped[profile.candidate_id].append(pair)
                history.append({
                    "turn": turn,
                    "skill": skill,
                    "question": canonical_question,
                    "answer": answer,
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                })
                progress.update(1)
                time.sleep(max(request_delay, 0.0))

    pairs = load_jsonl(pairs_path)
    rows = flatten_v2_prompt_pairs(pairs)
    write_jsonl_atomic(turns_path, rows)
    result = {
        "warning": (
            "Candidates are Qwen simulations, not humans. Pairwise order is blind "
            "and randomized; all generated questions are retained by default."
        ),
        "setup": setup,
        "summary": {
            variant: summarize_question_records([
                row for row in rows if row["strategy"] == variant
            ])
            for variant in variants
        },
        "paired_absolute_quality": paired_question_quality_comparison(
            rows, "prompt_tree_v2", "prompt_plain_v2",
        ),
        "blind_pairwise_preference": summarize_pairwise_preferences(pairs),
    }
    atomic_json(output_dir / "summary.json", result)
    run_question_quality_analysis(
        turns_path, output_dir / "quality_analysis", print_summary=False,
    )
    return result


def compact_judge_reliability(
    rows: Sequence[Mapping], variants: Sequence[str]
) -> Dict[str, Dict[str, object]]:
    """Expose the paper-facing repeat SD/ICC fields in the top-level summary."""
    output = {}
    for variant in variants:
        summary = summarize_quality_rows([
            row for row in rows if row["strategy"] == variant
        ])
        output[variant] = {}
        for metric in (
            "technical_correctness", "general_question_quality",
            "adaptive_diagnostic_quality", "overall_question_quality",
        ):
            for suffix in (
                "mean_judge_repeat_std", "judge_icc_1_1", "judge_icc_1_k",
            ):
                key = f"{metric}_{suffix}"
                value = float(summary[key])
                output[variant][key] = value if math.isfinite(value) else None
    return output


def run_prompt_ablation_v21(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    n_candidates: int,
    questions: int,
    question_judge_repeats: int,
    pairwise_judge_repeats: int,
    diagnostic_discrimination_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    regenerate_low_quality: bool,
    seed: int,
    request_delay: float,
) -> Dict:
    """V2.1 fixed-target ablation with validity-first diagnostic blueprints."""
    output_dir = output_root / "exp6_prompt_ablation_v21"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.json"
    pairs_path = output_dir / "pairs.jsonl"
    turns_path = output_dir / "turns.jsonl"
    setup_path = output_dir / "setup.json"
    profiles = generate_profiles(n_candidates, seed)
    plain_variant = "prompt_plain_v21"
    tree_variant = "prompt_tree_v21"
    variants = (plain_variant, tree_variant)
    setup = {
        "schema_version": 2,
        "method_version": "V2.1",
        "seed": seed,
        "question_model": question_model,
        "question_judge_model": question_judge_model,
        "candidate_model": candidate_model,
        "question_judge_repeats": question_judge_repeats,
        "pairwise_judge_repeats": pairwise_judge_repeats,
        "diagnostic_discrimination_repeats": diagnostic_discrimination_repeats,
        "quality_threshold": quality_threshold,
        "max_regenerations": max_regenerations,
        "regenerate_low_quality": regenerate_low_quality,
        "ability_update_source": "deterministic_latent_simulation",
        "candidates": n_candidates,
        "questions": questions,
        "prompt_variants": list(variants),
        "generation_pipeline": (
            "qwen-turbo structured blueprint then qwen-turbo shared "
            "technical-validity-guarded realization"
        ),
        "shared_validity_guard": V21_SHARED_VALIDITY_GUARD,
        "target_sampling": (
            "diagnostic uncertainty frontier; identical path, target, difficulty, "
            "posterior state, candidate state, and history for both variants"
        ),
        "pairwise_order": "blind deterministic randomization per repeat",
        "primary_metric": "blind_pairwise_adaptive_preference",
        "key_secondary_metric": "adaptive_diagnostic_quality",
        "mechanism_metric": "counterfactual_h0_h1_discrimination",
        "general_quality_role": "non-inferiority safeguard; margin -0.10",
        "technical_correctness_role": "validity safeguard; target delta >= -0.05",
    }
    if setup_path.exists():
        saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if saved_setup != setup:
            raise ValueError(
                "Existing V2.1 setup does not match this run; use the original "
                "arguments or a different --output directory"
            )
    else:
        atomic_json(setup_path, setup)
    serialized_profiles = [asdict(profile) for profile in profiles]
    if profiles_path.exists():
        if json.loads(profiles_path.read_text(encoding="utf-8")) != serialized_profiles:
            raise ValueError("Existing V2.1 profiles do not match the requested seed/setup")
    else:
        atomic_json(profiles_path, serialized_profiles)

    existing = load_jsonl(pairs_path)
    grouped: Dict[object, List[Dict]] = defaultdict(list)
    seen = set()
    for pair in existing:
        key = (pair["candidate_id"], pair["turn"])
        if key in seen:
            raise ValueError(f"Duplicate V2.1 pair record: {key}")
        seen.add(key)
        if set(pair["variants"]) != set(variants):
            raise ValueError(f"Incomplete V2.1 pair record: {key}")
        grouped[pair["candidate_id"]].append(pair)
    for candidate_id, candidate_pairs in grouped.items():
        candidate_pairs.sort(key=lambda row: int(row["turn"]))
        turns = [int(row["turn"]) for row in candidate_pairs]
        if turns != list(range(len(turns))) or len(turns) > questions:
            raise ValueError(f"Non-contiguous V2.1 pairs for candidate {candidate_id}")

    client = QwenClient(api_key, base_url)
    total = n_candidates * questions
    with tqdm(
        total=total,
        initial=len(existing),
        desc="V2.1 fixed-target prompt pairs",
    ) as progress:
        for profile in profiles:
            model = BayesianAbilityModel(propagation="tree")
            per_skill_count: Dict[str, int] = defaultdict(int)
            history = []
            completed = grouped[profile.candidate_id]
            for pair in completed:
                model.update(
                    pair["skill"], pair["ability_observation_score"],
                    observation_std=pair["ability_observation_std"],
                )
                per_skill_count[pair["skill"]] += 1
                history.append({
                    "turn": pair["turn"],
                    "skill": pair["skill"],
                    "question": pair["canonical_question"],
                    "answer": pair["answer"],
                    "ability_observation_score": pair["ability_observation_score"],
                    "ability_observation_std": pair["ability_observation_std"],
                })

            for turn in range(len(completed), questions):
                recent = [row["skill"] for row in history[-6:]]
                skill, difficulty, components = select_diagnostic_challenge_skill(
                    model, recent,
                )
                base_context = build_question_context(
                    "bridge", model, skill, components, history,
                )
                contrast_card = build_diagnostic_contrast_card(model, skill, history)
                base_context["diagnostic_contrast_card"] = contrast_card
                history_hash = hashlib.sha256(
                    json.dumps(history, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                pair_id = f"v21-c{profile.candidate_id:03d}-t{turn:03d}"
                generated: Dict[str, Dict] = {}
                generation_order = list(variants)
                if stable_seed("v21-generation-order", pair_id) % 2:
                    generation_order.reverse()
                for variant in generation_order:
                    context = dict(base_context)
                    context["prompt_variant"] = (
                        "tree_v21" if variant == tree_variant else "plain_v21"
                    )
                    question, scores, regeneration_count, attempts = (
                        generate_blueprint_question(
                            client, question_model, question_judge_model,
                            skill, difficulty, history, question_judge_repeats,
                            quality_threshold, max_regenerations,
                            regenerate_low_quality, context,
                        )
                    )
                    discrimination = evaluate_counterfactual_discrimination(
                        client=client,
                        answer_model=candidate_model,
                        judge_model=question_judge_model,
                        skill=skill,
                        difficulty=difficulty,
                        question=question,
                        contrast_card=contrast_card,
                        repeats=diagnostic_discrimination_repeats,
                        pair_id=pair_id,
                        variant=variant,
                    )
                    generated[variant] = {
                        "question": question,
                        "quality_scores": scores,
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": attempts,
                        "question_context": context,
                        "counterfactual_discrimination": discrimination,
                    }

                pairwise = evaluate_question_pair(
                    client, question_judge_model, skill, difficulty,
                    generated[tree_variant]["question"],
                    generated[plain_variant]["question"],
                    history, base_context, pairwise_judge_repeats, pair_id,
                    tree_variant=tree_variant, plain_variant=plain_variant,
                )
                canonical_question = generated[tree_variant]["question"]
                answer = generate_qwen_answer(
                    client, candidate_model, profile, skill, canonical_question,
                )
                job_before = model.job_variance()
                total_before = model.total_variance()
                hierarchy_before = model.hierarchy_variance(skill)
                ability_score, ability_std = simulate_score(
                    profile, skill, difficulty, per_skill_count[skill], seed,
                )
                model.update(skill, ability_score, observation_std=ability_std)
                per_skill_count[skill] += 1
                pair = {
                    "created_at": utc_now(),
                    "method_version": "V2.1",
                    "candidate_id": profile.candidate_id,
                    "level": profile.level,
                    "turn": turn,
                    "pair_id": pair_id,
                    "history_snapshot_sha256": history_hash,
                    "generation_order": generation_order,
                    "skill": skill,
                    "cluster": SKILL_TO_CLUSTER[skill],
                    "branch": SKILL_TO_BRANCH[skill],
                    "capability_path": list(SKILL_PATHS[skill]),
                    "difficulty": difficulty,
                    "quality_threshold": quality_threshold,
                    "quality_gate_enabled": regenerate_low_quality,
                    "variants": generated,
                    "pairwise_evaluation": pairwise,
                    "canonical_question": canonical_question,
                    "answer": answer,
                    "assigned_skill_theta": profile.skill_theta[skill],
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                    "posterior_skill_mean": model.skill_mean(skill),
                    "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                    "posterior_job_mean": model.job_mean(),
                    "posterior_job_std": math.sqrt(model.job_variance()),
                    "realized_job_variance_reduction": job_before - model.job_variance(),
                    "realized_global_variance_reduction": total_before - model.total_variance(),
                    "realized_hierarchy_variance_reduction": (
                        hierarchy_before - model.hierarchy_variance(skill)
                    ),
                    "selector": components,
                }
                append_jsonl(pairs_path, pair)
                grouped[profile.candidate_id].append(pair)
                history.append({
                    "turn": turn,
                    "skill": skill,
                    "question": canonical_question,
                    "answer": answer,
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                })
                progress.update(1)
                time.sleep(max(request_delay, 0.0))

    pairs = load_jsonl(pairs_path)
    rows = flatten_v2_prompt_pairs(pairs)
    write_jsonl_atomic(turns_path, rows)
    discrimination = summarize_counterfactual_discrimination(
        pairs, tree_variant, plain_variant,
    )
    result = {
        "warning": (
            "Candidates and H0/H1 answers are controlled Qwen simulations, not "
            "humans. All generated questions are retained by default."
        ),
        "setup": setup,
        "summary": {
            variant: summarize_question_records([
                row for row in rows if row["strategy"] == variant
            ])
            for variant in variants
        },
        "paired_absolute_quality": paired_question_quality_comparison(
            rows, tree_variant, plain_variant,
        ),
        "blind_pairwise_preference": summarize_pairwise_preferences(
            pairs, tree_variant, plain_variant,
        ),
        "counterfactual_discrimination": discrimination,
        "judge_reliability": compact_judge_reliability(rows, variants),
        "validity_audit": {
            variant: {
                "strategy_leakage_rate": quality_mean([
                    float(bool(row.get("strategy_leakage_flags")))
                    for row in rows if row["strategy"] == variant
                ]),
                "technical_correctness_below_7_rate": quality_mean([
                    float(row["question_quality_scores"]["technical_correctness"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "general_quality_below_7_rate": quality_mean([
                    float(row["general_question_quality"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "quality_gate_enabled": regenerate_low_quality,
            }
            for variant in variants
        },
    }
    atomic_json(output_dir / "summary.json", result)
    atomic_json(output_dir / "counterfactual_discrimination_summary.json", discrimination)
    write_quality_csv(
        output_dir / "counterfactual_discrimination_pairs.csv",
        flatten_counterfactual_pairs(pairs),
    )
    run_question_quality_analysis(
        turns_path, output_dir / "quality_analysis", print_summary=False,
    )
    atomic_json(output_dir / "key_metrics.json", prompt_key_metrics(result))
    return result


def run_prompt_ablation_v22(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    n_candidates: int,
    questions: int,
    question_judge_repeats: int,
    pairwise_judge_repeats: int,
    diagnostic_discrimination_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    regenerate_low_quality: bool,
    seed: int,
    request_delay: float,
) -> Dict:
    """V2.2 fixed-target ablation with minimal answerable diagnostic probes."""
    output_dir = output_root / "exp7_prompt_ablation_v22"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.json"
    pairs_path = output_dir / "pairs.jsonl"
    turns_path = output_dir / "turns.jsonl"
    setup_path = output_dir / "setup.json"
    profiles = generate_profiles(n_candidates, seed)
    plain_variant = "prompt_plain_v22"
    tree_variant = "prompt_tree_v22"
    variants = (plain_variant, tree_variant)
    setup = {
        "schema_version": 3,
        "method_version": "V2.2",
        "seed": seed,
        "question_model": question_model,
        "question_judge_model": question_judge_model,
        "candidate_model": candidate_model,
        "question_judge_repeats": question_judge_repeats,
        "pairwise_judge_repeats": pairwise_judge_repeats,
        "diagnostic_discrimination_repeats": diagnostic_discrimination_repeats,
        "quality_threshold": quality_threshold,
        "max_regenerations": max_regenerations,
        "regenerate_low_quality": regenerate_low_quality,
        "ability_update_source": "deterministic_latent_simulation",
        "candidates": n_candidates,
        "questions": questions,
        "prompt_variants": list(variants),
        "generation_pipeline": (
            "qwen-turbo atomic blueprint then qwen-turbo answerability-first "
            "technical-validity-guarded realization"
        ),
        "shared_validity_guard": V22_SHARED_VALIDITY_GUARD,
        "target_sampling": (
            "diagnostic uncertainty frontier; identical path, target, difficulty, "
            "posterior state, candidate state, and history for both variants"
        ),
        "pairwise_order": "blind deterministic randomization per repeat",
        "primary_metric": "blind_pairwise_adaptive_preference",
        "key_secondary_metric": "adaptive_diagnostic_quality",
        "mechanism_metric": "counterfactual_h0_h1_discrimination",
        "general_quality_role": "non-inferiority safeguard; margin -0.10",
        "technical_correctness_role": "validity safeguard; target delta >= -0.05",
        "v21_preserved": True,
        "v22_change": (
            "one atomic probe, 1-3 stable facts, 1-2 answer points, explicit "
            "answerability outline, forbidden assumptions, and lower temperature"
        ),
    }
    if setup_path.exists():
        saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if saved_setup != setup:
            raise ValueError(
                "Existing V2.2 setup does not match this run; use the original "
                "arguments or a different --output directory"
            )
    else:
        atomic_json(setup_path, setup)
    serialized_profiles = [asdict(profile) for profile in profiles]
    if profiles_path.exists():
        if json.loads(profiles_path.read_text(encoding="utf-8")) != serialized_profiles:
            raise ValueError("Existing V2.2 profiles do not match the requested seed/setup")
    else:
        atomic_json(profiles_path, serialized_profiles)

    existing = load_jsonl(pairs_path)
    grouped: Dict[object, List[Dict]] = defaultdict(list)
    seen = set()
    for pair in existing:
        key = (pair["candidate_id"], pair["turn"])
        if key in seen:
            raise ValueError(f"Duplicate V2.2 pair record: {key}")
        seen.add(key)
        if set(pair["variants"]) != set(variants):
            raise ValueError(f"Incomplete V2.2 pair record: {key}")
        grouped[pair["candidate_id"]].append(pair)
    for candidate_id, candidate_pairs in grouped.items():
        candidate_pairs.sort(key=lambda row: int(row["turn"]))
        turns = [int(row["turn"]) for row in candidate_pairs]
        if turns != list(range(len(turns))) or len(turns) > questions:
            raise ValueError(f"Non-contiguous V2.2 pairs for candidate {candidate_id}")

    client = QwenClient(api_key, base_url)
    total = n_candidates * questions
    with tqdm(
        total=total,
        initial=len(existing),
        desc="V2.2 answerability-first prompt pairs",
    ) as progress:
        for profile in profiles:
            model = BayesianAbilityModel(propagation="tree")
            per_skill_count: Dict[str, int] = defaultdict(int)
            history = []
            completed = grouped[profile.candidate_id]
            for pair in completed:
                model.update(
                    pair["skill"], pair["ability_observation_score"],
                    observation_std=pair["ability_observation_std"],
                )
                per_skill_count[pair["skill"]] += 1
                history.append({
                    "turn": pair["turn"],
                    "skill": pair["skill"],
                    "question": pair["canonical_question"],
                    "answer": pair["answer"],
                    "ability_observation_score": pair["ability_observation_score"],
                    "ability_observation_std": pair["ability_observation_std"],
                })

            for turn in range(len(completed), questions):
                recent = [row["skill"] for row in history[-6:]]
                skill, difficulty, components = select_diagnostic_challenge_skill(
                    model, recent,
                )
                base_context = build_question_context(
                    "bridge", model, skill, components, history,
                )
                contrast_card = build_diagnostic_contrast_card(model, skill, history)
                base_context["diagnostic_contrast_card"] = contrast_card
                history_hash = hashlib.sha256(
                    json.dumps(history, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                pair_id = f"v22-c{profile.candidate_id:03d}-t{turn:03d}"
                generated: Dict[str, Dict] = {}
                generation_order = list(variants)
                if stable_seed("v22-generation-order", pair_id) % 2:
                    generation_order.reverse()
                for variant in generation_order:
                    context = dict(base_context)
                    context["prompt_variant"] = (
                        "tree_v22" if variant == tree_variant else "plain_v22"
                    )
                    question, scores, regeneration_count, attempts = (
                        generate_v22_blueprint_question(
                            client, question_model, question_judge_model,
                            skill, difficulty, history, question_judge_repeats,
                            quality_threshold, max_regenerations,
                            regenerate_low_quality, context,
                        )
                    )
                    discrimination = evaluate_counterfactual_discrimination(
                        client=client,
                        answer_model=candidate_model,
                        judge_model=question_judge_model,
                        skill=skill,
                        difficulty=difficulty,
                        question=question,
                        contrast_card=contrast_card,
                        repeats=diagnostic_discrimination_repeats,
                        pair_id=pair_id,
                        variant=variant,
                    )
                    generated[variant] = {
                        "question": question,
                        "quality_scores": scores,
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": attempts,
                        "question_context": context,
                        "counterfactual_discrimination": discrimination,
                    }

                pairwise = evaluate_question_pair(
                    client, question_judge_model, skill, difficulty,
                    generated[tree_variant]["question"],
                    generated[plain_variant]["question"],
                    history, base_context, pairwise_judge_repeats, pair_id,
                    tree_variant=tree_variant, plain_variant=plain_variant,
                )
                canonical_question = generated[tree_variant]["question"]
                answer = generate_qwen_answer(
                    client, candidate_model, profile, skill, canonical_question,
                )
                job_before = model.job_variance()
                total_before = model.total_variance()
                hierarchy_before = model.hierarchy_variance(skill)
                ability_score, ability_std = simulate_score(
                    profile, skill, difficulty, per_skill_count[skill], seed,
                )
                model.update(skill, ability_score, observation_std=ability_std)
                per_skill_count[skill] += 1
                pair = {
                    "created_at": utc_now(),
                    "method_version": "V2.2",
                    "candidate_id": profile.candidate_id,
                    "level": profile.level,
                    "turn": turn,
                    "pair_id": pair_id,
                    "history_snapshot_sha256": history_hash,
                    "generation_order": generation_order,
                    "skill": skill,
                    "cluster": SKILL_TO_CLUSTER[skill],
                    "branch": SKILL_TO_BRANCH[skill],
                    "capability_path": list(SKILL_PATHS[skill]),
                    "difficulty": difficulty,
                    "quality_threshold": quality_threshold,
                    "quality_gate_enabled": regenerate_low_quality,
                    "variants": generated,
                    "pairwise_evaluation": pairwise,
                    "canonical_question": canonical_question,
                    "answer": answer,
                    "assigned_skill_theta": profile.skill_theta[skill],
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                    "posterior_skill_mean": model.skill_mean(skill),
                    "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                    "posterior_job_mean": model.job_mean(),
                    "posterior_job_std": math.sqrt(model.job_variance()),
                    "realized_job_variance_reduction": job_before - model.job_variance(),
                    "realized_global_variance_reduction": total_before - model.total_variance(),
                    "realized_hierarchy_variance_reduction": (
                        hierarchy_before - model.hierarchy_variance(skill)
                    ),
                    "selector": components,
                }
                append_jsonl(pairs_path, pair)
                grouped[profile.candidate_id].append(pair)
                history.append({
                    "turn": turn,
                    "skill": skill,
                    "question": canonical_question,
                    "answer": answer,
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                })
                progress.update(1)
                time.sleep(max(request_delay, 0.0))

    pairs = load_jsonl(pairs_path)
    rows = flatten_v2_prompt_pairs(pairs)
    write_jsonl_atomic(turns_path, rows)
    discrimination = summarize_counterfactual_discrimination(
        pairs, tree_variant, plain_variant,
    )
    result = {
        "warning": (
            "Candidates and H0/H1 answers are controlled Qwen simulations, not "
            "humans. All generated questions are retained by default."
        ),
        "setup": setup,
        "summary": {
            variant: summarize_question_records([
                row for row in rows if row["strategy"] == variant
            ])
            for variant in variants
        },
        "paired_absolute_quality": paired_question_quality_comparison(
            rows, tree_variant, plain_variant,
        ),
        "blind_pairwise_preference": summarize_pairwise_preferences(
            pairs, tree_variant, plain_variant,
        ),
        "counterfactual_discrimination": discrimination,
        "judge_reliability": compact_judge_reliability(rows, variants),
        "validity_audit": {
            variant: {
                "strategy_leakage_rate": quality_mean([
                    float(bool(row.get("strategy_leakage_flags")))
                    for row in rows if row["strategy"] == variant
                ]),
                "technical_correctness_below_7_rate": quality_mean([
                    float(row["question_quality_scores"]["technical_correctness"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "general_quality_below_7_rate": quality_mean([
                    float(row["general_question_quality"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "quality_gate_enabled": regenerate_low_quality,
            }
            for variant in variants
        },
    }
    atomic_json(output_dir / "summary.json", result)
    atomic_json(output_dir / "counterfactual_discrimination_summary.json", discrimination)
    write_quality_csv(
        output_dir / "counterfactual_discrimination_pairs.csv",
        flatten_counterfactual_pairs(pairs),
    )
    run_question_quality_analysis(
        turns_path, output_dir / "quality_analysis", print_summary=False,
    )
    atomic_json(output_dir / "key_metrics.json", prompt_key_metrics(result))
    return result


def run_prompt_ablation_v23(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    n_candidates: int,
    questions: int,
    question_judge_repeats: int,
    pairwise_judge_repeats: int,
    diagnostic_discrimination_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    regenerate_low_quality: bool,
    seed: int,
    request_delay: float,
) -> Dict:
    """V2.3 ablation with one neutral technical scaffold per matched pair."""
    output_dir = output_root / "exp8_prompt_ablation_v23"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.json"
    pairs_path = output_dir / "pairs.jsonl"
    turns_path = output_dir / "turns.jsonl"
    setup_path = output_dir / "setup.json"
    profiles = generate_profiles(n_candidates, seed)
    plain_variant = "prompt_plain_v23"
    tree_variant = "prompt_tree_v23"
    variants = (plain_variant, tree_variant)
    setup = {
        "schema_version": 4,
        "method_version": "V2.3",
        "seed": seed,
        "question_model": question_model,
        "question_judge_model": question_judge_model,
        "candidate_model": candidate_model,
        "question_judge_repeats": question_judge_repeats,
        "pairwise_judge_repeats": pairwise_judge_repeats,
        "diagnostic_discrimination_repeats": diagnostic_discrimination_repeats,
        "quality_threshold": quality_threshold,
        "max_regenerations": max_regenerations,
        "regenerate_low_quality": regenerate_low_quality,
        "ability_update_source": "deterministic_latent_simulation",
        "candidates": n_candidates,
        "questions": questions,
        "prompt_variants": list(variants),
        "generation_pipeline": (
            "one shared neutral qwen-turbo technical scaffold then one "
            "budget-matched qwen-turbo realization per variant"
        ),
        "shared_validity_guard": V23_SHARED_VALIDITY_GUARD,
        "history_policy": (
            "all prior questions prevent repetition; only same-branch history may "
            "influence Tree diagnostic realization"
        ),
        "target_sampling": (
            "diagnostic uncertainty frontier; identical path, target, difficulty, "
            "posterior state, candidate state, history, facts, and probe menu"
        ),
        "pairwise_order": (
            "blind deterministic randomization with strict technical-validity gate"
        ),
        "primary_metric": "blind_pairwise_adaptive_preference",
        "key_secondary_metric": "adaptive_diagnostic_quality",
        "mechanism_metric": "counterfactual_question_diagnosticity",
        "classification_metric_role": "ceiling-effect audit only",
        "general_quality_role": "non-inferiority safeguard; margin -0.10",
        "technical_correctness_role": "validity safeguard; target delta >= -0.05",
        "v21_and_v22_preserved": True,
    }
    if setup_path.exists():
        saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if saved_setup != setup:
            raise ValueError(
                "Existing V2.3 setup does not match this run; use the original "
                "arguments or a different --output directory"
            )
    else:
        atomic_json(setup_path, setup)
    serialized_profiles = [asdict(profile) for profile in profiles]
    if profiles_path.exists():
        if json.loads(profiles_path.read_text(encoding="utf-8")) != serialized_profiles:
            raise ValueError("Existing V2.3 profiles do not match the requested seed/setup")
    else:
        atomic_json(profiles_path, serialized_profiles)

    existing = load_jsonl(pairs_path)
    grouped: Dict[object, List[Dict]] = defaultdict(list)
    seen = set()
    for pair in existing:
        key = (pair["candidate_id"], pair["turn"])
        if key in seen:
            raise ValueError(f"Duplicate V2.3 pair record: {key}")
        seen.add(key)
        if set(pair["variants"]) != set(variants):
            raise ValueError(f"Incomplete V2.3 pair record: {key}")
        grouped[pair["candidate_id"]].append(pair)
    for candidate_id, candidate_pairs in grouped.items():
        candidate_pairs.sort(key=lambda row: int(row["turn"]))
        turns = [int(row["turn"]) for row in candidate_pairs]
        if turns != list(range(len(turns))) or len(turns) > questions:
            raise ValueError(f"Non-contiguous V2.3 pairs for candidate {candidate_id}")

    client = QwenClient(api_key, base_url)
    total = n_candidates * questions
    with tqdm(
        total=total,
        initial=len(existing),
        desc="V2.3 shared-scaffold prompt pairs",
    ) as progress:
        for profile in profiles:
            model = BayesianAbilityModel(propagation="tree")
            per_skill_count: Dict[str, int] = defaultdict(int)
            history = []
            completed = grouped[profile.candidate_id]
            for pair in completed:
                model.update(
                    pair["skill"], pair["ability_observation_score"],
                    observation_std=pair["ability_observation_std"],
                )
                per_skill_count[pair["skill"]] += 1
                history.append({
                    "turn": pair["turn"],
                    "skill": pair["skill"],
                    "question": pair["canonical_question"],
                    "answer": pair["answer"],
                    "ability_observation_score": pair["ability_observation_score"],
                    "ability_observation_std": pair["ability_observation_std"],
                })

            for turn in range(len(completed), questions):
                recent = [row["skill"] for row in history[-6:]]
                skill, difficulty, components = select_diagnostic_challenge_skill(
                    model, recent,
                )
                base_context = build_question_context(
                    "bridge", model, skill, components, history,
                )
                same_branch_history = v23_same_branch_history(history, skill)
                contrast_card = build_diagnostic_contrast_card(
                    model, skill, same_branch_history,
                )
                base_context["diagnostic_contrast_card"] = contrast_card
                base_context["same_branch_history"] = [
                    {
                        "skill": str(row.get("skill", "")),
                        "question": str(row.get("question", ""))[:300],
                        "answer": str(row.get("answer", ""))[:450],
                    }
                    for row in same_branch_history
                ]
                history_hash = hashlib.sha256(
                    json.dumps(history, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                pair_id = f"v23-c{profile.candidate_id:03d}-t{turn:03d}"
                shared_scaffold = build_v23_shared_scaffold(
                    client, question_model, skill, difficulty, history,
                )
                generated: Dict[str, Dict] = {}
                generation_order = list(variants)
                if stable_seed("v23-generation-order", pair_id) % 2:
                    generation_order.reverse()
                for variant in generation_order:
                    context = dict(base_context)
                    context["prompt_variant"] = (
                        "tree_v23" if variant == tree_variant else "plain_v23"
                    )
                    question, scores, regeneration_count, attempts = (
                        generate_v23_shared_scaffold_question(
                            client, question_model, question_judge_model,
                            skill, difficulty, history, question_judge_repeats,
                            quality_threshold, max_regenerations,
                            regenerate_low_quality, context, shared_scaffold,
                        )
                    )
                    discrimination = evaluate_counterfactual_discrimination(
                        client=client,
                        answer_model=candidate_model,
                        judge_model=question_judge_model,
                        skill=skill,
                        difficulty=difficulty,
                        question=question,
                        contrast_card=contrast_card,
                        repeats=diagnostic_discrimination_repeats,
                        pair_id=pair_id,
                        variant=variant,
                    )
                    generated[variant] = {
                        "question": question,
                        "quality_scores": scores,
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": attempts,
                        "question_context": context,
                        "counterfactual_discrimination": discrimination,
                    }

                pairwise = evaluate_question_pair(
                    client, question_judge_model, skill, difficulty,
                    generated[tree_variant]["question"],
                    generated[plain_variant]["question"],
                    history, base_context, pairwise_judge_repeats, pair_id,
                    tree_variant=tree_variant, plain_variant=plain_variant,
                    strict_technical_gate=True,
                )
                canonical_question = generated[tree_variant]["question"]
                answer = generate_qwen_answer(
                    client, candidate_model, profile, skill, canonical_question,
                )
                job_before = model.job_variance()
                total_before = model.total_variance()
                hierarchy_before = model.hierarchy_variance(skill)
                ability_score, ability_std = simulate_score(
                    profile, skill, difficulty, per_skill_count[skill], seed,
                )
                model.update(skill, ability_score, observation_std=ability_std)
                per_skill_count[skill] += 1
                pair = {
                    "created_at": utc_now(),
                    "method_version": "V2.3",
                    "candidate_id": profile.candidate_id,
                    "level": profile.level,
                    "turn": turn,
                    "pair_id": pair_id,
                    "history_snapshot_sha256": history_hash,
                    "generation_order": generation_order,
                    "skill": skill,
                    "cluster": SKILL_TO_CLUSTER[skill],
                    "branch": SKILL_TO_BRANCH[skill],
                    "capability_path": list(SKILL_PATHS[skill]),
                    "difficulty": difficulty,
                    "quality_threshold": quality_threshold,
                    "quality_gate_enabled": regenerate_low_quality,
                    "shared_technical_scaffold": v23_scaffold_payload(
                        shared_scaffold
                    ),
                    "shared_scaffold_model_raw_output": shared_scaffold.get("raw", ""),
                    "shared_scaffold_format_retry_count": shared_scaffold.get(
                        "format_retry_count", 0
                    ),
                    "variants": generated,
                    "pairwise_evaluation": pairwise,
                    "canonical_question": canonical_question,
                    "answer": answer,
                    "assigned_skill_theta": profile.skill_theta[skill],
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                    "posterior_skill_mean": model.skill_mean(skill),
                    "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                    "posterior_job_mean": model.job_mean(),
                    "posterior_job_std": math.sqrt(model.job_variance()),
                    "realized_job_variance_reduction": job_before - model.job_variance(),
                    "realized_global_variance_reduction": total_before - model.total_variance(),
                    "realized_hierarchy_variance_reduction": (
                        hierarchy_before - model.hierarchy_variance(skill)
                    ),
                    "selector": components,
                }
                append_jsonl(pairs_path, pair)
                grouped[profile.candidate_id].append(pair)
                history.append({
                    "turn": turn,
                    "skill": skill,
                    "question": canonical_question,
                    "answer": answer,
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                })
                progress.update(1)
                time.sleep(max(request_delay, 0.0))

    pairs = load_jsonl(pairs_path)
    rows = flatten_v2_prompt_pairs(pairs)
    write_jsonl_atomic(turns_path, rows)
    discrimination = summarize_counterfactual_discrimination(
        pairs, tree_variant, plain_variant,
    )
    result = {
        "warning": (
            "Candidates and H0/H1 answers are controlled Qwen simulations, not "
            "humans. All generated questions are retained by default."
        ),
        "setup": setup,
        "summary": {
            variant: summarize_question_records([
                row for row in rows if row["strategy"] == variant
            ])
            for variant in variants
        },
        "paired_absolute_quality": paired_question_quality_comparison(
            rows, tree_variant, plain_variant,
        ),
        "blind_pairwise_preference": summarize_pairwise_preferences(
            pairs, tree_variant, plain_variant,
        ),
        "counterfactual_discrimination": discrimination,
        "judge_reliability": compact_judge_reliability(rows, variants),
        "validity_audit": {
            variant: {
                "strategy_leakage_rate": quality_mean([
                    float(bool(row.get("strategy_leakage_flags")))
                    for row in rows if row["strategy"] == variant
                ]),
                "technical_correctness_below_7_rate": quality_mean([
                    float(row["question_quality_scores"]["technical_correctness"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "general_quality_below_7_rate": quality_mean([
                    float(row["general_question_quality"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "quality_gate_enabled": regenerate_low_quality,
            }
            for variant in variants
        },
    }
    atomic_json(output_dir / "summary.json", result)
    atomic_json(output_dir / "counterfactual_discrimination_summary.json", discrimination)
    write_quality_csv(
        output_dir / "counterfactual_discrimination_pairs.csv",
        flatten_counterfactual_pairs(pairs),
    )
    run_question_quality_analysis(
        turns_path, output_dir / "quality_analysis", print_summary=False,
    )
    atomic_json(output_dir / "key_metrics.json", prompt_key_metrics(result))
    return result


def run_prompt_ablation_v24(
    output_root: Path,
    api_key: str,
    base_url: str,
    question_model: str,
    question_judge_model: str,
    candidate_model: str,
    n_candidates: int,
    questions: int,
    question_judge_repeats: int,
    pairwise_judge_repeats: int,
    diagnostic_discrimination_repeats: int,
    quality_threshold: float,
    max_regenerations: int,
    regenerate_low_quality: bool,
    seed: int,
    request_delay: float,
) -> Dict:
    """V2.4 ablation: identical facts, independently designed atomic probes."""
    output_dir = output_root / "exp9_prompt_ablation_v24"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.json"
    pairs_path = output_dir / "pairs.jsonl"
    turns_path = output_dir / "turns.jsonl"
    setup_path = output_dir / "setup.json"
    profiles = generate_profiles(n_candidates, seed)
    plain_variant = "prompt_plain_v24"
    tree_variant = "prompt_tree_v24"
    variants = (plain_variant, tree_variant)
    setup = {
        "schema_version": 5,
        "method_version": "V2.4",
        "seed": seed,
        "question_model": question_model,
        "question_judge_model": question_judge_model,
        "candidate_model": candidate_model,
        "question_judge_repeats": question_judge_repeats,
        "pairwise_judge_repeats": pairwise_judge_repeats,
        "diagnostic_discrimination_repeats": diagnostic_discrimination_repeats,
        "quality_threshold": quality_threshold,
        "max_regenerations": max_regenerations,
        "regenerate_low_quality": regenerate_low_quality,
        "ability_update_source": "deterministic_latent_simulation",
        "candidates": n_candidates,
        "questions": questions,
        "prompt_variants": list(variants),
        "generation_pipeline": (
            "one shared fact-only qwen-turbo scenario then one budget-matched "
            "variant-specific atomic probe per arm; scenario is concatenated verbatim"
        ),
        "shared_validity_guard": V24_SHARED_FACT_GUARD,
        "history_policy": (
            "all prior questions prevent repetition; only exact-leaf history may "
            "define an optional Tree evidence gap"
        ),
        "target_sampling": (
            "diagnostic uncertainty frontier; identical path, target, difficulty, "
            "posterior state, candidate state, history, and factual scenario; probe "
            "design is the intended treatment"
        ),
        "pairwise_order": (
            "blind deterministic randomization with strict technical-validity gate"
        ),
        "primary_metric": "blind_pairwise_adaptive_preference",
        "key_secondary_metric": "adaptive_diagnostic_quality",
        "mechanism_metric": "responsive_counterfactual_question_diagnosticity",
        "classification_metric_role": "ceiling-effect audit only",
        "general_quality_role": "non-inferiority safeguard; margin -0.10",
        "technical_correctness_role": "validity safeguard; target delta >= -0.05",
        "earlier_prompt_modes_preserved": True,
    }
    if setup_path.exists():
        saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if saved_setup != setup:
            raise ValueError(
                "Existing V2.4 setup does not match this run; use the original "
                "arguments or a different --output directory"
            )
    else:
        atomic_json(setup_path, setup)
    serialized_profiles = [asdict(profile) for profile in profiles]
    if profiles_path.exists():
        if json.loads(profiles_path.read_text(encoding="utf-8")) != serialized_profiles:
            raise ValueError("Existing V2.4 profiles do not match the requested seed/setup")
    else:
        atomic_json(profiles_path, serialized_profiles)

    existing = load_jsonl(pairs_path)
    grouped: Dict[object, List[Dict]] = defaultdict(list)
    seen = set()
    for pair in existing:
        key = (pair["candidate_id"], pair["turn"])
        if key in seen:
            raise ValueError(f"Duplicate V2.4 pair record: {key}")
        seen.add(key)
        if set(pair["variants"]) != set(variants):
            raise ValueError(f"Incomplete V2.4 pair record: {key}")
        grouped[pair["candidate_id"]].append(pair)
    for candidate_id, candidate_pairs in grouped.items():
        candidate_pairs.sort(key=lambda row: int(row["turn"]))
        turns = [int(row["turn"]) for row in candidate_pairs]
        if turns != list(range(len(turns))) or len(turns) > questions:
            raise ValueError(f"Non-contiguous V2.4 pairs for candidate {candidate_id}")

    client = QwenClient(api_key, base_url)
    total = n_candidates * questions
    with tqdm(
        total=total,
        initial=len(existing),
        desc="V2.4 shared-facts prompt pairs",
    ) as progress:
        for profile in profiles:
            model = BayesianAbilityModel(propagation="tree")
            per_skill_count: Dict[str, int] = defaultdict(int)
            history = []
            completed = grouped[profile.candidate_id]
            for pair in completed:
                model.update(
                    pair["skill"], pair["ability_observation_score"],
                    observation_std=pair["ability_observation_std"],
                )
                per_skill_count[pair["skill"]] += 1
                history.append({
                    "turn": pair["turn"],
                    "skill": pair["skill"],
                    "question": pair["canonical_question"],
                    "answer": pair["answer"],
                    "ability_observation_score": pair["ability_observation_score"],
                    "ability_observation_std": pair["ability_observation_std"],
                })

            for turn in range(len(completed), questions):
                recent = [row["skill"] for row in history[-6:]]
                skill, difficulty, components = select_diagnostic_challenge_skill(
                    model, recent,
                )
                base_context = build_question_context(
                    "bridge", model, skill, components, history,
                )
                exact_history = v24_exact_skill_history(history, skill)
                contrast_card = build_diagnostic_contrast_card(
                    model, skill, exact_history,
                )
                base_context["diagnostic_contrast_card"] = contrast_card
                base_context["exact_skill_history"] = [
                    {
                        "skill": str(row.get("skill", "")),
                        "question": str(row.get("question", ""))[:300],
                        "answer": str(row.get("answer", ""))[:450],
                    }
                    for row in exact_history
                ]
                history_hash = hashlib.sha256(
                    json.dumps(history, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                pair_id = f"v24-c{profile.candidate_id:03d}-t{turn:03d}"
                shared_scaffold = build_v24_shared_scaffold(
                    client, question_model, skill, difficulty, history,
                )
                generated: Dict[str, Dict] = {}
                generation_order = list(variants)
                if stable_seed("v24-generation-order", pair_id) % 2:
                    generation_order.reverse()
                for variant in generation_order:
                    context = dict(base_context)
                    context["prompt_variant"] = (
                        "tree_v24" if variant == tree_variant else "plain_v24"
                    )
                    question, scores, regeneration_count, attempts = (
                        generate_v24_shared_fact_question(
                            client, question_model, question_judge_model,
                            skill, difficulty, history, question_judge_repeats,
                            quality_threshold, max_regenerations,
                            regenerate_low_quality, context, shared_scaffold,
                        )
                    )
                    discrimination = evaluate_counterfactual_discrimination(
                        client=client,
                        answer_model=candidate_model,
                        judge_model=question_judge_model,
                        skill=skill,
                        difficulty=difficulty,
                        question=question,
                        contrast_card=contrast_card,
                        repeats=diagnostic_discrimination_repeats,
                        pair_id=pair_id,
                        variant=variant,
                        responsive_only=True,
                    )
                    generated[variant] = {
                        "question": question,
                        "quality_scores": scores,
                        "regeneration_count": regeneration_count,
                        "question_generation_attempts": attempts,
                        "question_context": context,
                        "counterfactual_discrimination": discrimination,
                    }

                pairwise = evaluate_question_pair(
                    client, question_judge_model, skill, difficulty,
                    generated[tree_variant]["question"],
                    generated[plain_variant]["question"],
                    history, base_context, pairwise_judge_repeats, pair_id,
                    tree_variant=tree_variant, plain_variant=plain_variant,
                    strict_technical_gate=True,
                )
                canonical_question = generated[tree_variant]["question"]
                answer = generate_qwen_answer(
                    client, candidate_model, profile, skill, canonical_question,
                )
                job_before = model.job_variance()
                total_before = model.total_variance()
                hierarchy_before = model.hierarchy_variance(skill)
                ability_score, ability_std = simulate_score(
                    profile, skill, difficulty, per_skill_count[skill], seed,
                )
                model.update(skill, ability_score, observation_std=ability_std)
                per_skill_count[skill] += 1
                pair = {
                    "created_at": utc_now(),
                    "method_version": "V2.4",
                    "candidate_id": profile.candidate_id,
                    "level": profile.level,
                    "turn": turn,
                    "pair_id": pair_id,
                    "history_snapshot_sha256": history_hash,
                    "generation_order": generation_order,
                    "skill": skill,
                    "cluster": SKILL_TO_CLUSTER[skill],
                    "branch": SKILL_TO_BRANCH[skill],
                    "capability_path": list(SKILL_PATHS[skill]),
                    "difficulty": difficulty,
                    "quality_threshold": quality_threshold,
                    "quality_gate_enabled": regenerate_low_quality,
                    "shared_factual_scaffold": v24_scaffold_payload(shared_scaffold),
                    "shared_scaffold_model_raw_output": shared_scaffold.get("raw", ""),
                    "shared_scaffold_format_retry_count": shared_scaffold.get(
                        "format_retry_count", 0
                    ),
                    "variants": generated,
                    "pairwise_evaluation": pairwise,
                    "canonical_question": canonical_question,
                    "answer": answer,
                    "assigned_skill_theta": profile.skill_theta[skill],
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                    "posterior_skill_mean": model.skill_mean(skill),
                    "posterior_skill_std": math.sqrt(model.skill_variance(skill)),
                    "posterior_job_mean": model.job_mean(),
                    "posterior_job_std": math.sqrt(model.job_variance()),
                    "realized_job_variance_reduction": job_before - model.job_variance(),
                    "realized_global_variance_reduction": total_before - model.total_variance(),
                    "realized_hierarchy_variance_reduction": (
                        hierarchy_before - model.hierarchy_variance(skill)
                    ),
                    "selector": components,
                }
                append_jsonl(pairs_path, pair)
                grouped[profile.candidate_id].append(pair)
                history.append({
                    "turn": turn,
                    "skill": skill,
                    "question": canonical_question,
                    "answer": answer,
                    "ability_observation_score": ability_score,
                    "ability_observation_std": ability_std,
                })
                progress.update(1)
                time.sleep(max(request_delay, 0.0))

    pairs = load_jsonl(pairs_path)
    rows = flatten_v2_prompt_pairs(pairs)
    write_jsonl_atomic(turns_path, rows)
    discrimination = summarize_counterfactual_discrimination(
        pairs, tree_variant, plain_variant,
    )
    result = {
        "warning": (
            "Candidates and H0/H1 answers are controlled Qwen simulations, not "
            "humans. All generated questions are retained by default."
        ),
        "setup": setup,
        "summary": {
            variant: summarize_question_records([
                row for row in rows if row["strategy"] == variant
            ])
            for variant in variants
        },
        "paired_absolute_quality": paired_question_quality_comparison(
            rows, tree_variant, plain_variant,
        ),
        "blind_pairwise_preference": summarize_pairwise_preferences(
            pairs, tree_variant, plain_variant,
        ),
        "counterfactual_discrimination": discrimination,
        "judge_reliability": compact_judge_reliability(rows, variants),
        "validity_audit": {
            variant: {
                "strategy_leakage_rate": quality_mean([
                    float(bool(row.get("strategy_leakage_flags")))
                    for row in rows if row["strategy"] == variant
                ]),
                "technical_correctness_below_7_rate": quality_mean([
                    float(row["question_quality_scores"]["technical_correctness"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "general_quality_below_7_rate": quality_mean([
                    float(row["general_question_quality"] < 7.0)
                    for row in rows if row["strategy"] == variant
                ]),
                "quality_gate_enabled": regenerate_low_quality,
            }
            for variant in variants
        },
    }
    atomic_json(output_dir / "summary.json", result)
    atomic_json(output_dir / "counterfactual_discrimination_summary.json", discrimination)
    write_quality_csv(
        output_dir / "counterfactual_discrimination_pairs.csv",
        flatten_counterfactual_pairs(pairs),
    )
    run_question_quality_analysis(
        turns_path, output_dir / "quality_analysis", print_summary=False,
    )
    atomic_json(output_dir / "key_metrics.json", prompt_key_metrics(result))
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


def quality_value(row: Mapping, metric: str) -> float:
    if metric in row:
        return float(row[metric])
    scores = row["question_quality_scores"]
    if metric in scores:
        return float(scores[metric])
    if metric == "general_question_quality":
        return quality_mean([float(scores[name]) for name in GENERAL_QUALITY_DIMENSIONS])
    if metric == "adaptive_diagnostic_quality":
        return quality_mean([float(scores[name]) for name in ADAPTIVE_QUALITY_DIMENSIONS])
    if metric == "overall_question_quality":
        return quality_mean([float(scores[name]) for name in QUESTION_QUALITY_DIMENSIONS])
    raise KeyError(metric)


def judge_detail_value(detail: Mapping, metric: str) -> float:
    if metric in QUESTION_QUALITY_DIMENSIONS:
        return float(detail[metric])
    if metric == "general_question_quality":
        return quality_mean([float(detail[name]) for name in GENERAL_QUALITY_DIMENSIONS])
    if metric == "adaptive_diagnostic_quality":
        return quality_mean([float(detail[name]) for name in ADAPTIVE_QUALITY_DIMENSIONS])
    if metric == "overall_question_quality":
        return quality_mean([float(detail[name]) for name in QUESTION_QUALITY_DIMENSIONS])
    raise KeyError(metric)


def judge_repeat_icc(ratings: Sequence[Sequence[float]]) -> float:
    """One-way random-effects ICC(1,1) across repeated blind judge calls."""
    matrix = [list(map(float, row)) for row in ratings if len(row) >= 2]
    if len(matrix) < 2:
        return float("nan")
    repeats = min(len(row) for row in matrix)
    matrix = [row[:repeats] for row in matrix]
    row_means = [quality_mean(row) for row in matrix]
    grand = quality_mean([value for row in matrix for value in row])
    between_ms = repeats * sum((value - grand) ** 2 for value in row_means) / (len(matrix) - 1)
    within_ms = sum(
        (value - row_mean) ** 2
        for row, row_mean in zip(matrix, row_means) for value in row
    ) / (len(matrix) * (repeats - 1))
    denominator = between_ms + (repeats - 1) * within_ms
    return (between_ms - within_ms) / denominator if denominator > _EPS else 0.0


def judge_repeat_icc_average(ratings: Sequence[Sequence[float]]) -> float:
    """One-way random-effects reliability of the mean, ICC(1,k)."""
    matrix = [list(map(float, row)) for row in ratings if len(row) >= 2]
    if len(matrix) < 2:
        return float("nan")
    repeats = min(len(row) for row in matrix)
    matrix = [row[:repeats] for row in matrix]
    row_means = [quality_mean(row) for row in matrix]
    grand = quality_mean([value for row in matrix for value in row])
    between_ms = repeats * sum((value - grand) ** 2 for value in row_means) / (len(matrix) - 1)
    within_ms = sum(
        (value - row_mean) ** 2
        for row, row_mean in zip(matrix, row_means) for value in row
    ) / (len(matrix) * (repeats - 1))
    return (between_ms - within_ms) / between_ms if between_ms > _EPS else 0.0


def holm_adjust(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm family-wise error correction with monotone adjusted p-values."""
    finite = [(name, float(value)) for name, value in pvalues.items() if not math.isnan(float(value))]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: Dict[str, float] = {name: float("nan") for name in pvalues}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


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

    for name in COMPOSITE_QUALITY_METRICS:
        values = [quality_value(row, name) for row in rows]
        result.update(quality_metric_stats(values, name))

    overall = [quality_value(row, "overall_question_quality") for row in rows]
    first = [first_attempt_quality(row) for row in rows]
    thresholds = [float(row.get("quality_threshold", 7.0)) for row in rows]
    regenerations = [int(row["regeneration_count"]) for row in rows]
    attempts = [len(row["question_generation_attempts"]) for row in rows]
    final_pass = [
        bool(row.get("quality_threshold_met", score >= threshold))
        for row, score, threshold in zip(rows, overall, thresholds)
    ]
    gate_enabled = [bool(row.get("quality_gate_enabled", True)) for row in rows]
    result.update(quality_metric_stats(first, "first_attempt_overall_quality"))

    repeat_metrics = (*active_dimensions, *COMPOSITE_QUALITY_METRICS)
    for metric in repeat_metrics:
        per_question = []
        for row in rows:
            details = row["question_generation_attempts"][-1].get(
                "question_judge_details", []
            )
            ratings = [
                judge_detail_value(detail, metric) for detail in details
                if all(name in detail for name in QUESTION_QUALITY_DIMENSIONS)
            ]
            if ratings:
                per_question.append(ratings)
        result[f"{metric}_mean_judge_repeat_std"] = (
            quality_mean([quality_sample_std(values) for values in per_question])
            if per_question else 0.0
        )
        result[f"{metric}_judge_icc_1_1"] = judge_repeat_icc(per_question)
        result[f"{metric}_judge_icc_1_k"] = judge_repeat_icc_average(per_question)

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
        "average_within_question_judge_std": result.get(
            "overall_question_quality_mean_judge_repeat_std", 0.0
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
    candidate_rows: Dict[Tuple[str, object], List[Mapping]] = defaultdict(list)
    for row in rows:
        candidate_rows[(str(row["strategy"]), row["candidate_id"])].append(row)
    first_ids = {
        candidate_id for strategy, candidate_id in candidate_rows
        if strategy == first
    }
    second_ids = {
        candidate_id for strategy, candidate_id in candidate_rows
        if strategy == second
    }
    common = sorted(first_ids & second_ids, key=str)
    if not common:
        return {
            "available": False,
            "reason": f"No candidates shared by {first} and {second}",
        }
    metric_results = {}
    for metric in (*QUESTION_QUALITY_DIMENSIONS, *COMPOSITE_QUALITY_METRICS):
        deltas = [
            quality_mean([quality_value(row, metric) for row in candidate_rows[(first, candidate_id)]])
            - quality_mean([quality_value(row, metric) for row in candidate_rows[(second, candidate_id)]])
            for candidate_id in common
        ]
        ci_low, ci_high = quality_bootstrap_ci(
            deltas, seed=stable_seed("paired-quality", first, second, metric)
        )
        delta_std = quality_sample_std(deltas)
        metric_results[metric] = {
            "first_candidate_mean": quality_mean([
                quality_mean([quality_value(row, metric) for row in candidate_rows[(first, candidate_id)]])
                for candidate_id in common
            ]),
            "second_candidate_mean": quality_mean([
                quality_mean([quality_value(row, metric) for row in candidate_rows[(second, candidate_id)]])
                for candidate_id in common
            ]),
            "mean_delta": quality_mean(deltas),
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
            "paired_effect_size_dz": quality_mean(deltas) / delta_std if delta_std > 0 else 0.0,
            "first_win_rate": quality_mean([float(value > _EPS) for value in deltas]),
            "tie_rate": quality_mean([float(abs(value) <= _EPS) for value in deltas]),
            "first_loss_rate": quality_mean([float(value < -_EPS) for value in deltas]),
            "paired_sign_flip_p": paired_sign_flip_p(
                deltas,
                seed=stable_seed("paired-quality-sign-flip", first, second, metric),
            ),
        }
    adjusted = holm_adjust({
        metric: float(values["paired_sign_flip_p"])
        for metric, values in metric_results.items()
    })
    for metric, value in adjusted.items():
        metric_results[metric]["holm_adjusted_p"] = value
    overall = metric_results["overall_question_quality"]
    general = metric_results["general_question_quality"]
    technical = metric_results["technical_correctness"]
    return {
        "available": True,
        "comparison": f"{first}_minus_{second}",
        "unit": "candidate mean across turns",
        "n_paired_candidates": len(common),
        "metrics": metric_results,
        "mean_overall_quality_delta": overall["mean_delta"],
        "bootstrap_95_ci_low": overall["bootstrap_95_ci_low"],
        "bootstrap_95_ci_high": overall["bootstrap_95_ci_high"],
        "paired_effect_size_dz": overall["paired_effect_size_dz"],
        "first_win_rate": overall["first_win_rate"],
        "tie_rate": overall["tie_rate"],
        "first_loss_rate": overall["first_loss_rate"],
        "general_quality_safeguard": {
            "noninferiority_margin": -0.10,
            "two_sided_bootstrap_ci_above_margin": (
                general["bootstrap_95_ci_low"] > -0.10
            ),
            "mean_delta": general["mean_delta"],
            "bootstrap_95_ci_low": general["bootstrap_95_ci_low"],
        },
        "technical_correctness_safeguard": {
            "development_target": -0.05,
            "mean_delta_meets_target": technical["mean_delta"] >= -0.05,
            "mean_delta": technical["mean_delta"],
            "bootstrap_95_ci_low": technical["bootstrap_95_ci_low"],
        },
        "note": (
            "Both prompt variants use the same capability path, skill, difficulty, "
            "posterior state, dialogue history, and candidate state."
            if (first, second) in (
                ("prompt_tree", "prompt_plain"),
                ("prompt_tree_v2", "prompt_plain_v2"),
                ("prompt_tree_v21", "prompt_plain_v21"),
                ("prompt_tree_v22", "prompt_plain_v22"),
                ("prompt_tree_v23", "prompt_plain_v23"),
                ("prompt_tree_v24", "prompt_plain_v24"),
            )
            else "Strategies select different skills and difficulties, so this paired "
                 "delta measures end-to-end sequence quality rather than generator "
                 "quality alone."
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
        "general_question_quality": quality_value(row, "general_question_quality"),
        "adaptive_diagnostic_quality": quality_value(row, "adaptive_diagnostic_quality"),
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


def run_question_quality_analysis(
    input_path: Path,
    output_dir: Path,
    print_summary: bool = True,
) -> Dict:
    rows = load_question_quality_jsonl(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_strategy = grouped_quality_summaries(rows, ["strategy"])
    by_difficulty = grouped_quality_summaries(rows, ["strategy", "difficulty"])
    by_cluster = grouped_quality_summaries(rows, ["strategy", "cluster"])
    by_branch = grouped_quality_summaries(rows, ["strategy", "branch"]) if all(
        "branch" in row for row in rows
    ) else []
    available_strategies = {str(row["strategy"]) for row in rows}
    comparison_pairs = [
        pair for pair in (
            ("bridge", "random"),
            ("tree_bridge", "tree_random"),
            ("tree_bridge", "flat_bridge"),
            ("tree_bridge", "independent_bridge"),
            ("prompt_tree", "prompt_plain"),
            ("prompt_tree_v2", "prompt_plain_v2"),
            ("prompt_tree_v21", "prompt_plain_v21"),
            ("prompt_tree_v22", "prompt_plain_v22"),
            ("prompt_tree_v23", "prompt_plain_v23"),
            ("prompt_tree_v24", "prompt_plain_v24"),
        )
        if set(pair).issubset(available_strategies)
    ]
    comparisons = {
        f"{first}_vs_{second}": paired_question_quality_comparison(rows, first, second)
        for first, second in comparison_pairs
    }
    primary_key = next(iter(comparisons), None)
    comparison = comparisons[primary_key] if primary_key else {
        "available": False,
        "reason": "No configured paired strategy comparison is present",
    }
    bridge_random = comparisons.get("bridge_vs_random", {
        "available": False,
        "reason": "bridge and random are not both present",
    })
    report = {
        "input": str(input_path.resolve()),
        "n_records": len(rows),
        "strategies": sorted({row["strategy"] for row in rows}),
        "overall": summarize_quality_rows(rows),
        "by_strategy": by_strategy,
        "bridge_vs_random": bridge_random,
        "primary_paired_comparison": comparison,
        "paired_comparisons": comparisons,
        "interpretation_notes": [
            "Question-quality scores never enter BayesianAbilityModel.",
            "Turns from the same candidate are repeated measures; candidate means are paired.",
            "Different strategies select different skill/difficulty mixtures; inspect stratified CSV files.",
            "Fixed-target prompt variants hold path, difficulty, history, and state constant.",
        ],
    }
    atomic_json(output_dir / "question_quality_summary.json", report)
    write_quality_csv(output_dir / "question_quality_by_strategy.csv", by_strategy)
    write_quality_csv(output_dir / "question_quality_by_difficulty.csv", by_difficulty)
    write_quality_csv(output_dir / "question_quality_by_cluster.csv", by_cluster)
    write_quality_csv(output_dir / "question_quality_by_branch.csv", by_branch)
    paired_rows = []
    for comparison_name, paired in comparisons.items():
        for metric, values in paired.get("metrics", {}).items():
            paired_rows.append({
                "comparison": comparison_name,
                "metric": metric,
                "n_paired_candidates": paired["n_paired_candidates"],
                **values,
            })
    write_quality_csv(output_dir / "paired_quality_comparisons.csv", paired_rows)
    write_quality_csv(
        output_dir / "question_quality_turns.csv",
        [flatten_quality_turn(row) for row in rows],
    )
    if comparison.get("available"):
        key_quality = {
            "comparison": comparison["comparison"],
            "n_paired_candidates": comparison["n_paired_candidates"],
            "metrics": {
                metric: comparison["metrics"][metric]
                for metric in (
                    "adaptive_diagnostic_quality", "technical_correctness",
                    "general_question_quality", "overall_question_quality",
                )
            },
            "general_quality_safeguard": comparison["general_quality_safeguard"],
            "technical_correctness_safeguard": comparison[
                "technical_correctness_safeguard"
            ],
        }
    else:
        key_quality = comparison
    atomic_json(output_dir / "key_quality_metrics.json", key_quality)
    if print_summary:
        print("\nQuestion quality by strategy")
        for row in by_strategy:
            print(
                f"{row['strategy']}: n={row['n_turns']}, "
                f"overall={row['overall_question_quality_mean']:.3f}, "
                f"general={row['general_question_quality_mean']:.3f}, "
                f"adaptive={row['adaptive_diagnostic_quality_mean']:.3f}"
            )
        if comparison.get("available"):
            print("\nKey paired deltas (first - second)")
            for metric in (
                "adaptive_diagnostic_quality", "technical_correctness",
                "general_question_quality", "overall_question_quality",
            ):
                values = comparison["metrics"][metric]
                print(
                    f"{metric}: {values['mean_delta']:+.3f} "
                    f"[{values['bootstrap_95_ci_low']:+.3f}, "
                    f"{values['bootstrap_95_ci_high']:+.3f}]"
                )
        print(f"\nSaved analysis to: {output_dir.resolve()}")
    return report


# =============================================================================
# CLI
# =============================================================================

def prompt_key_metrics(result: Mapping) -> Dict[str, object]:
    """Return the pre-registered prompt metrics without dropping full artifacts."""
    comparison = result.get("paired_absolute_quality", {})
    if not comparison.get("available"):
        comparison = result.get("paired_prompt_comparison", {})
    if not comparison.get("available"):
        return {"available": False, "reason": "No paired prompt comparison"}

    setup = result.get("setup", {})
    variants = list(setup.get("prompt_variants", []))
    tree_variant = next((name for name in variants if "tree" in name), "tree")
    plain_variant = next((name for name in variants if "plain" in name), "plain")
    metrics = comparison["metrics"]
    adaptive = metrics["adaptive_diagnostic_quality"]
    technical = metrics["technical_correctness"]
    general = metrics["general_question_quality"]
    overall = metrics["overall_question_quality"]
    pairwise = result.get("blind_pairwise_preference", {})
    discrimination = result.get("counterfactual_discrimination", {})
    reliability = result.get("judge_reliability", {}).get(tree_variant, {})
    audit = result.get("validity_audit", {}).get(tree_variant, {})

    def delta(values: Mapping) -> Dict[str, float]:
        return {
            "tree_minus_plain": float(values["mean_delta"]),
            "bootstrap_95_ci_low": float(values["bootstrap_95_ci_low"]),
            "bootstrap_95_ci_high": float(values["bootstrap_95_ci_high"]),
        }

    key: Dict[str, object] = {
        "available": True,
        "method_version": setup.get("method_version", "legacy"),
        "sample": {
            "candidates": int(setup.get("candidates", comparison["n_paired_candidates"])),
            "questions_per_candidate": int(setup.get("questions", 0)),
            "paired_candidates": int(comparison["n_paired_candidates"]),
            "question_pairs": int(pairwise.get("n_pairs", 0)),
        },
        "primary_blind_tree_preference": {
            "value": float(pairwise.get("tree_preference_score", float("nan"))),
            "bootstrap_95_ci_low": float(
                pairwise.get("tree_preference_bootstrap_95_ci_low", float("nan"))
            ),
            "bootstrap_95_ci_high": float(
                pairwise.get("tree_preference_bootstrap_95_ci_high", float("nan"))
            ),
            "development_target": 0.70,
            "meets_point_target": (
                float(pairwise.get("tree_preference_score", float("nan"))) >= 0.70
            ),
        },
        "adaptive_diagnostic_quality": {
            **delta(adaptive),
            "development_target": 0.15,
            "meets_point_target": float(adaptive["mean_delta"]) >= 0.15,
        },
        "technical_correctness": {
            **delta(technical),
            "development_target": -0.05,
            "meets_point_target": float(technical["mean_delta"]) >= -0.05,
        },
        "general_question_quality": {
            **delta(general),
            "noninferiority_margin": -0.10,
            "ci_above_margin": float(general["bootstrap_95_ci_low"]) > -0.10,
        },
        "overall_question_quality": delta(overall),
        "tree_validity_audit": {
            "strategy_leakage_rate": audit.get("strategy_leakage_rate"),
            "technical_correctness_below_7_rate": audit.get(
                "technical_correctness_below_7_rate"
            ),
            "general_quality_below_7_rate": audit.get(
                "general_quality_below_7_rate"
            ),
        },
        "tree_judge_reliability": {
            "adaptive_diagnostic_quality_icc_1_k": reliability.get(
                "adaptive_diagnostic_quality_judge_icc_1_k"
            ),
            "overall_question_quality_icc_1_k": reliability.get(
                "overall_question_quality_judge_icc_1_k"
            ),
        },
        "study_stage": (
            "development only; after prompt changes use a new seed and new sample "
            "for confirmation"
        ),
    }
    if discrimination.get("available"):
        by_variant = discrimination["by_variant"]
        paired = discrimination["paired_tree_minus_plain"]
        state = paired["state_classification_accuracy"]
        diagnosticity = paired["mean_question_diagnosticity"]
        key["counterfactual_h0_h1"] = {
            "tree_state_accuracy": float(
                by_variant[tree_variant]["state_classification_accuracy_mean"]
            ),
            "plain_state_accuracy": float(
                by_variant[plain_variant]["state_classification_accuracy_mean"]
            ),
            "state_accuracy_delta": float(state["mean_delta"]),
            "diagnosticity_delta": float(diagnosticity["mean_delta"]),
            "classification_ceiling": (
                float(by_variant[tree_variant]["state_classification_accuracy_mean"])
                >= 1.0 - _EPS
                and float(
                    by_variant[plain_variant]["state_classification_accuracy_mean"]
                ) >= 1.0 - _EPS
            ),
            "note": "controlled simulation mechanism check, not human validation",
        }
    return key


def _format_ci(values: Mapping, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"[{float(values['bootstrap_95_ci_low']) * scale:+.2f}, "
        f"{float(values['bootstrap_95_ci_high']) * scale:+.2f}]{suffix}"
    )


def print_prompt_key_metrics(title: str, result: Mapping) -> None:
    key = prompt_key_metrics(result)
    print(f"\n{title}")
    if not key.get("available"):
        print(key.get("reason", "No key metrics available"))
        return
    sample = key["sample"]
    print(
        f"Sample: {sample['paired_candidates']} candidates, "
        f"{sample['question_pairs']} matched question pairs"
    )
    primary = key["primary_blind_tree_preference"]
    print(
        "Blind Tree preference: "
        f"{primary['value'] * 100:.2f}% {_format_ci(primary, percent=True)} "
        f"target>=70% [{'PASS' if primary['meets_point_target'] else 'FAIL'}]"
    )
    for label, key_name, target_text, pass_name in (
        (
            "Adaptive diagnostic quality", "adaptive_diagnostic_quality",
            "target>=+0.15", "meets_point_target",
        ),
        (
            "Technical correctness", "technical_correctness",
            "target>=-0.05", "meets_point_target",
        ),
        (
            "General question quality", "general_question_quality",
            "CI lower>-0.10", "ci_above_margin",
        ),
    ):
        values = key[key_name]
        print(
            f"{label}: {values['tree_minus_plain']:+.3f} {_format_ci(values)} "
            f"{target_text} [{'PASS' if values[pass_name] else 'FAIL'}]"
        )
    overall = key["overall_question_quality"]
    print(
        f"Overall quality: {overall['tree_minus_plain']:+.3f} "
        f"{_format_ci(overall)}"
    )
    mechanism = key.get("counterfactual_h0_h1")
    if mechanism:
        print(
            "H0/H1 state accuracy: "
            f"Tree={mechanism['tree_state_accuracy'] * 100:.2f}%, "
            f"Plain={mechanism['plain_state_accuracy'] * 100:.2f}%, "
            f"delta={mechanism['state_accuracy_delta'] * 100:+.2f} pp"
            f"{' [CEILING]' if mechanism['classification_ceiling'] else ''}"
        )
        print(
            "Counterfactual diagnosticity delta: "
            f"{mechanism['diagnosticity_delta']:+.3f}"
        )
    audit = key["tree_validity_audit"]
    if audit["strategy_leakage_rate"] is not None:
        print(
            "Tree validity: "
            f"leakage={audit['strategy_leakage_rate'] * 100:.2f}%, "
            f"technical<7={audit['technical_correctness_below_7_rate'] * 100:.2f}%, "
            f"general<7={audit['general_quality_below_7_rate'] * 100:.2f}%"
        )
    print("Stage: development only; do not treat this run as confirmatory evidence.")


def print_compact(title: str, result: Dict, verbose: bool = False) -> None:
    if not verbose and (
        result.get("paired_absolute_quality", {}).get("available")
        or result.get("paired_prompt_comparison", {}).get("available")
    ):
        print_prompt_key_metrics(title, result)
        return
    print(f"\n{title}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=[
            "exp1", "exp2", "selector", "estimator", "structure_ablation",
            "objective_ablation", "robustness", "job_conditioned",
            "synthetic_suite", "qwen", "prompt_ablation",
            "prompt_ablation_v2", "prompt_ablation_v21",
            "prompt_ablation_v22", "prompt_ablation_v23",
            "prompt_ablation_v24", "analyze", "all",
        ],
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
        "--profile-regime", choices=PROFILE_REGIMES, default="hierarchical",
        help="Latent ability data-generating mechanism for selector experiments",
    )
    parser.add_argument(
        "--profile-regimes", nargs="+", choices=PROFILE_REGIMES,
        default=["hierarchical", "weak_hierarchy", "flat_correlated", "independent"],
        help="Data-generating mechanisms for the estimator benchmark",
    )
    parser.add_argument(
        "--misspecification-rate", type=float, default=0.0,
        help="Fraction of leaf placements corrupted in shuffled_tree profiles",
    )
    parser.add_argument(
        "--job-profile", choices=list(JOB_WEIGHT_PROFILES), default="uniform",
    )
    parser.add_argument(
        "--job-profiles", nargs="+", choices=list(JOB_WEIGHT_PROFILES),
        default=["backend", "cloud_sre", "distributed_systems"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGY_SPECS),
        default=["tree_bridge", "flat_bridge", "independent_bridge", "tree_random"],
    )

    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--question-model", default="qwen-turbo")
    parser.add_argument("--question-judge-model", default="qwen-max")
    parser.add_argument("--question-judge-repeats", type=int, default=3)
    parser.add_argument(
        "--pairwise-judge-repeats", type=int, default=3,
        help="Blind A/B judge repeats for prompt_ablation_v2/v21/v22/v23/v24",
    )
    parser.add_argument(
        "--diagnostic-discrimination-repeats", type=int, default=1,
        help=(
            "Blind H0/H1 classification repeats per question for "
            "prompt_ablation_v21/v22/v23/v24; use 0 to disable"
        ),
    )
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
    parser.add_argument(
        "--verbose-results",
        action="store_true",
        help=(
            "Print complete result JSON. By default prompt-ablation modes print "
            "only pre-registered key metrics; full results are always saved."
        ),
    )
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
        "capability_tree": CAPABILITY_TREE,
        "tree_shape": {"root": 1, "domains": 9, "subcapabilities": 27, "leaves": 108},
        "job_domain_weights": JOB_DOMAIN_WEIGHTS,
        "profile_regimes": list(PROFILE_REGIMES),
        "note": (
            "--mode all preserves the legacy synthetic Exp. 1 and Exp. 2. "
            "--mode synthetic_suite runs the publication synthetic suite. "
            "Qwen modes always require explicit selection."
        ),
    }
    atomic_json(output_root / "manifest.json", manifest)

    if args.mode in ("exp1", "all"):
        result = run_exp1(output_root, args.candidates, args.seed)
        print_compact("Exp. 1 complete", result, verbose=args.verbose_results)

    if args.mode in ("exp2", "selector", "all"):
        result = run_exp2(
            output_root=output_root,
            seeds=args.seeds,
            n_candidates=args.candidates,
            questions=args.questions,
            strategies=args.strategies,
            base_seed=args.seed,
            job_profile=args.job_profile,
            profile_regime=args.profile_regime,
            misspecification_rate=args.misspecification_rate,
        )
        print_compact("Exp. 2 complete", result, verbose=args.verbose_results)

    if args.mode in ("estimator", "synthetic_suite"):
        result = run_estimator_benchmark(
            output_root, args.seeds, args.candidates, args.questions, args.seed,
            args.profile_regimes, args.job_profile,
        )
        print_compact(
            "Estimator benchmark complete", result, verbose=args.verbose_results,
        )

    if args.mode in ("structure_ablation", "synthetic_suite"):
        result = run_structure_ablation(
            output_root, args.seeds, args.candidates, args.questions, args.seed,
        )
        print_compact(
            "Structure ablation complete", result, verbose=args.verbose_results,
        )

    if args.mode in ("objective_ablation", "synthetic_suite"):
        result = run_objective_ablation(
            output_root, args.seeds, args.candidates, args.questions, args.seed,
        )
        print_compact(
            "Objective ablation complete", result, verbose=args.verbose_results,
        )

    if args.mode in ("robustness", "synthetic_suite"):
        result = run_robustness_benchmark(
            output_root, args.seeds, args.candidates, args.questions, args.seed,
        )
        print_compact(
            "Robustness benchmark complete", result, verbose=args.verbose_results,
        )

    if args.mode in ("job_conditioned", "synthetic_suite"):
        result = run_job_conditioned_benchmark(
            output_root, args.seeds, args.candidates, args.questions, args.seed,
            args.job_profiles,
        )
        print_compact(
            "Job-conditioned benchmark complete", result,
            verbose=args.verbose_results,
        )

    if args.mode in (
        "qwen", "prompt_ablation", "prompt_ablation_v2",
        "prompt_ablation_v21", "prompt_ablation_v22", "prompt_ablation_v23",
        "prompt_ablation_v24",
    ):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise SystemExit(
                "Qwen mode requires DASHSCOPE_API_KEY in the environment"
            )
        if args.question_judge_repeats < 1:
            raise SystemExit("--question-judge-repeats must be at least one")
        if args.pairwise_judge_repeats < 1:
            raise SystemExit("--pairwise-judge-repeats must be at least one")
        if args.diagnostic_discrimination_repeats < 0:
            raise SystemExit("--diagnostic-discrimination-repeats cannot be negative")
        if not 1.0 <= args.quality_threshold <= 10.0:
            raise SystemExit("--quality-threshold must be within [1,10]")
        if args.max_regenerations < 0:
            raise SystemExit("--max-regenerations must be non-negative")
        question_attempts = (
            args.max_regenerations + 1 if args.regenerate_low_quality else 1
        )
        if args.mode == "qwen":
            calls_per_turn = question_attempts * (1 + args.question_judge_repeats) + 1
            estimated_calls = (
                args.qwen_candidates * len(args.strategies)
                * args.qwen_questions * calls_per_turn
            )
        elif args.mode == "prompt_ablation":
            calls_per_pair = 2 * question_attempts * (1 + args.question_judge_repeats) + 1
            estimated_calls = args.qwen_candidates * args.qwen_questions * calls_per_pair
        elif args.mode == "prompt_ablation_v2":
            calls_per_pair = (
                2 * question_attempts * (2 + args.question_judge_repeats)
                + args.pairwise_judge_repeats + 1
            )
            estimated_calls = args.qwen_candidates * args.qwen_questions * calls_per_pair
        elif args.mode in ("prompt_ablation_v21", "prompt_ablation_v22"):
            discrimination_calls = (
                2 * (1 + args.diagnostic_discrimination_repeats)
                if args.diagnostic_discrimination_repeats > 0 else 0
            )
            calls_per_pair = (
                2 * question_attempts * (2 + args.question_judge_repeats)
                + args.pairwise_judge_repeats + 1 + discrimination_calls
            )
            estimated_calls = args.qwen_candidates * args.qwen_questions * calls_per_pair
        elif args.mode in ("prompt_ablation_v23", "prompt_ablation_v24"):
            discrimination_calls = (
                2 * (1 + args.diagnostic_discrimination_repeats)
                if args.diagnostic_discrimination_repeats > 0 else 0
            )
            calls_per_pair = (
                1 + 2 * question_attempts * (1 + args.question_judge_repeats)
                + args.pairwise_judge_repeats + 1 + discrimination_calls
            )
            estimated_calls = args.qwen_candidates * args.qwen_questions * calls_per_pair
        print(f"Estimated planned API calls (excluding network/format retries): {estimated_calls}")
        if not args.confirm_api_calls:
            raise SystemExit("Re-run with --confirm-api-calls after checking the estimated cost")
        common = {
            "output_root": output_root,
            "api_key": api_key,
            "base_url": args.base_url,
            "question_model": args.question_model,
            "question_judge_model": args.question_judge_model,
            "candidate_model": args.candidate_model,
            "n_candidates": args.qwen_candidates,
            "questions": args.qwen_questions,
            "question_judge_repeats": args.question_judge_repeats,
            "quality_threshold": args.quality_threshold,
            "max_regenerations": args.max_regenerations,
            "regenerate_low_quality": args.regenerate_low_quality,
            "seed": args.seed,
            "request_delay": args.request_delay,
        }
        if args.mode == "qwen":
            result = run_qwen(strategies=args.strategies, **common)
            print_compact(
                "Qwen pilot complete", result, verbose=args.verbose_results,
            )
        elif args.mode == "prompt_ablation":
            result = run_prompt_ablation(**common)
            print_compact(
                "Fixed-target prompt ablation complete", result,
                verbose=args.verbose_results,
            )
        elif args.mode == "prompt_ablation_v2":
            result = run_prompt_ablation_v2(
                pairwise_judge_repeats=args.pairwise_judge_repeats,
                **common,
            )
            print_compact(
                "V2 fixed-target prompt ablation complete", result,
                verbose=args.verbose_results,
            )
        elif args.mode == "prompt_ablation_v21":
            result = run_prompt_ablation_v21(
                pairwise_judge_repeats=args.pairwise_judge_repeats,
                diagnostic_discrimination_repeats=(
                    args.diagnostic_discrimination_repeats
                ),
                **common,
            )
            print_compact(
                "V2.1 fixed-target prompt ablation complete", result,
                verbose=args.verbose_results,
            )
        elif args.mode == "prompt_ablation_v22":
            result = run_prompt_ablation_v22(
                pairwise_judge_repeats=args.pairwise_judge_repeats,
                diagnostic_discrimination_repeats=(
                    args.diagnostic_discrimination_repeats
                ),
                **common,
            )
            print_compact(
                "V2.2 answerability-first prompt ablation complete", result,
                verbose=args.verbose_results,
            )
        elif args.mode == "prompt_ablation_v23":
            result = run_prompt_ablation_v23(
                pairwise_judge_repeats=args.pairwise_judge_repeats,
                diagnostic_discrimination_repeats=(
                    args.diagnostic_discrimination_repeats
                ),
                **common,
            )
            print_compact(
                "V2.3 shared-scaffold prompt ablation complete", result,
                verbose=args.verbose_results,
            )
        else:
            result = run_prompt_ablation_v24(
                pairwise_judge_repeats=args.pairwise_judge_repeats,
                diagnostic_discrimination_repeats=(
                    args.diagnostic_discrimination_repeats
                ),
                **common,
            )
            print_compact(
                "V2.4 shared-facts prompt ablation complete", result,
                verbose=args.verbose_results,
            )


if __name__ == "__main__":
    main()
