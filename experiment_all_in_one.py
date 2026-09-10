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

Examples:
  python experiment_all_in_one.py --mode exp1 --output runs
  python experiment_all_in_one.py --mode exp2 --seeds 20 --output runs
  python experiment_all_in_one.py --mode prompt_ablation \
      --qwen-candidates 3 --qwen-questions 3 --confirm-api-calls --output runs
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


def generate_profiles(n_candidates: int, seed: int) -> List[CandidateProfile]:
    rng = random.Random(seed)
    levels = balanced_levels(n_candidates)
    rng.shuffle(levels)
    profiles = []
    for candidate_id, level in enumerate(levels):
        config = LEVEL_CONFIG[level]
        global_theta = clip(rng.gauss(config["mean"], config["std"]))
        strong = sorted(rng.sample(CLUSTERS, config["strong"]))
        node_theta = {ROOT_NODE: global_theta}
        skill_theta = {}
        for domain, branches in CAPABILITY_TREE.items():
            domain_id = next(
                node.node_id for node in CAPABILITY_NODES
                if node.level == 1 and node.label == domain
            )
            domain_shift = 0.75 if domain in strong else -0.35
            domain_theta = clip(global_theta + domain_shift + rng.gauss(0.0, 0.30))
            node_theta[domain_id] = domain_theta
            for branch, skills in branches.items():
                branch_id = next(
                    node.node_id for node in CAPABILITY_NODES
                    if node.level == 2 and node.label == branch
                    and node.parent_id == domain_id
                )
                branch_theta = clip(domain_theta + rng.gauss(0.0, 0.40))
                node_theta[branch_id] = branch_theta
                for skill in skills:
                    value = clip(branch_theta + rng.gauss(0.0, 0.50))
                    skill_theta[skill] = value
                    node_theta[SKILL_TO_NODE[skill]] = value
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
    """Exact joint Gaussian posterior under tree, flat, or independent priors.

    In ``tree`` mode every one of the 145 nodes is a random variable.  The
    prior is induced by ``theta_child = theta_parent + eta_level``.  Therefore
    covariance between any two nodes is exactly the accumulated innovation
    variance on their shared ancestral path.  Conditioning on a leaf score is
    standard Gaussian conditioning over the full node covariance matrix.
    """

    MODES = ("tree", "flat", "independent")

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
    ) -> None:
        if graph_enabled is not None:
            propagation = "tree" if graph_enabled else "independent"
        if propagation not in self.MODES:
            raise ValueError(f"propagation must be one of {self.MODES}, got {propagation!r}")
        if len(inheritance_stds) != 3:
            raise ValueError("inheritance_stds must contain L1, L2, and leaf standard deviations")
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
                    if i == j:
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
        return sum(JOB_WEIGHTS[skill] * self.skill_mean(skill) for skill in SKILLS)

    def job_variance(self) -> float:
        total = 0.0
        for first, wf in JOB_WEIGHTS.items():
            i = self._skill_index(first)
            for second, ws in JOB_WEIGHTS.items():
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
            JOB_WEIGHTS[name] * self.covariance[self._skill_index(name)][j]
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
        question_cost = 1.0 + 0.15 * repeat_count
        utility = (
            0.55 * relative_decision
            + 0.25 * relative_global
            + 0.20 * relative_hierarchy
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
    "independent_bridge": ("independent", "bridge"),
    "uncertainty": ("tree", "uncertainty"),
    "fixed": ("tree", "fixed"),
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


def strategy_for_candidate(
    profile: CandidateProfile,
    strategy: str,
    questions: int,
    seed: int,
) -> Dict:
    propagation, _policy = strategy_spec(strategy)
    model = BayesianAbilityModel(propagation=propagation)
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
            "branch": SKILL_TO_BRANCH[skill],
            "capability_path": list(SKILL_PATHS[skill]),
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
        "propagation": propagation,
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
            comparisons[f"{bridge_name}_vs_{baseline}"] = {
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
    run_question_quality_analysis(turns_path, output_dir / "quality_analysis")
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
    run_question_quality_analysis(turns_path, output_dir / "quality_analysis")
    return summary


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
        }
    overall = metric_results["overall_question_quality"]
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
        "note": (
            "Both prompt variants use the same capability path, skill, difficulty, "
            "posterior state, dialogue history, and candidate state."
            if (first, second) == ("prompt_tree", "prompt_plain")
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


def run_question_quality_analysis(input_path: Path, output_dir: Path) -> Dict:
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
            "Fixed-target prompt_tree vs prompt_plain rows hold path, difficulty, history, and state constant.",
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
    print("\nQuestion quality by strategy")
    for row in by_strategy:
        print(
            f"{row['strategy']}: n={row['n_turns']}, "
            f"overall={row['overall_question_quality_mean']:.4f}, "
            f"general={row['general_question_quality_mean']:.4f}, "
            f"adaptive={row['adaptive_diagnostic_quality_mean']:.4f}, "
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
        "--mode", choices=[
            "exp1", "exp2", "qwen", "prompt_ablation", "analyze", "all",
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
        "capability_tree": CAPABILITY_TREE,
        "tree_shape": {"root": 1, "domains": 9, "subcapabilities": 27, "leaves": 108},
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

    if args.mode in ("qwen", "prompt_ablation"):
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
        if args.mode == "qwen":
            calls_per_turn = question_attempts * (1 + args.question_judge_repeats) + 1
            estimated_calls = (
                args.qwen_candidates * len(args.strategies)
                * args.qwen_questions * calls_per_turn
            )
        else:
            calls_per_pair = 2 * question_attempts * (1 + args.question_judge_repeats) + 1
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
            print_compact("Qwen pilot complete", result)
        else:
            result = run_prompt_ablation(**common)
            print_compact("Fixed-target prompt ablation complete", result)


if __name__ == "__main__":
    main()
