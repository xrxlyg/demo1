"""
Bayesian ability model for adaptive technical interviews.

The model keeps a joint Gaussian posterior over technical skills.  The prior
covariance is derived from SkillGraph distance, so evidence observed for one
skill is propagated to related skills.  Competency dimensions use independent
Gaussian posteriors.

All internal values are normalized to [0, 1].  Public update methods accept
the original 0--10 evaluator scores by default.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple


logger = logging.getLogger(__name__)
_EPS = 1e-12


@dataclass
class SkillAbility:
    """Marginal posterior for one skill or competency dimension."""

    mean: float = 0.5
    variance: float = 0.35 ** 2
    observations: int = 0

    @property
    def uncertainty(self) -> float:
        """Posterior standard deviation (kept for API compatibility)."""
        return math.sqrt(max(self.variance, 0.0))

    @property
    def std(self) -> float:
        return self.uncertainty

    def information_gain(self, observation_variance: float = 0.12 ** 2) -> float:
        """Expected entropy reduction from one Gaussian observation."""
        r = max(float(observation_variance), _EPS)
        return 0.5 * math.log1p(max(self.variance, 0.0) / r)

    def is_sufficiently_assessed(self, threshold: float = 0.10) -> bool:
        return self.uncertainty < threshold

    def __repr__(self) -> str:
        return (
            f"SkillAbility(mean={self.mean:.3f}, std={self.uncertainty:.3f}, "
            f"n={self.observations})"
        )


class AbilityModel:
    """
    Joint Gaussian Bayesian estimator over technical skills.

    Observation model for a question targeting skill ``s``::

        y_t = theta_s + epsilon_t,  epsilon_t ~ N(0, R_t)

    ``R_t`` can be supplied per answer from repeated-judge disagreement.  If
    omitted, ``observation_std`` is used.  The posterior update is the exact
    linear-Gaussian (Kalman) update, not a running average.
    """

    def __init__(
        self,
        prior_mean: float = 0.5,
        prior_std: float = 0.35,
        observation_std: float = 0.12,
        graph_length_scale: float = 1.5,
        graph_correlation: float = 0.65,
    ) -> None:
        self.prior_mean = float(prior_mean)
        self.prior_variance = max(float(prior_std) ** 2, _EPS)
        self.observation_variance = max(float(observation_std) ** 2, _EPS)
        self.graph_length_scale = max(float(graph_length_scale), _EPS)
        self.graph_correlation = min(max(float(graph_correlation), 0.0), 0.99)

        self.skills: Dict[str, SkillAbility] = {}
        self.dimensions: Dict[str, SkillAbility] = {}

        self._skill_order: List[str] = []
        self._skill_index: Dict[str, int] = {}
        self._skill_mean: List[float] = []
        self._skill_covariance: List[List[float]] = []
        self._graph_signature: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Initialization and graph prior
    # ------------------------------------------------------------------

    def initialize_from_job_analysis(self, job_analysis: Dict) -> None:
        for skill in job_analysis.get("required_skills", []):
            self.add_skill(skill)
        for skill in job_analysis.get("preferred_skills", []):
            self.add_skill(skill)
        for dim in job_analysis.get("competency_dimensions", []):
            name = dim if isinstance(dim, str) else dim.get("dimension", str(dim))
            self.add_dimension(name)

    def add_skill(self, skill: str) -> None:
        if skill in self.skills:
            return
        self.skills[skill] = SkillAbility(
            mean=self.prior_mean,
            variance=self.prior_variance,
        )
        self._skill_index[skill] = len(self._skill_order)
        self._skill_order.append(skill)
        self._skill_mean.append(self.prior_mean)

        old_n = len(self._skill_covariance)
        for row in self._skill_covariance:
            row.append(0.0)
        self._skill_covariance.append([0.0] * old_n + [self.prior_variance])

    def add_dimension(self, dimension: str) -> None:
        if dimension not in self.dimensions:
            self.dimensions[dimension] = SkillAbility(
                mean=self.prior_mean,
                variance=self.prior_variance,
            )

    def configure_skill_graph(self, skill_graph) -> bool:
        """
        Build a graph-distance covariance prior.

        This must happen before the first observation.  Repeated calls with the
        same graph are harmless.  Returns ``True`` when the prior is configured.
        """
        if skill_graph is None:
            return False

        nodes = getattr(skill_graph, "nodes", {})
        for name in nodes:
            self.add_skill(name)

        edges = tuple(sorted(
            (name, child)
            for name, node in nodes.items()
            for child in getattr(node, "children", [])
        ))
        signature = (tuple(sorted(nodes)), edges)
        if signature == self._graph_signature:
            return True
        if any(a.observations > 0 for a in self.skills.values()):
            logger.warning("SkillGraph prior cannot be changed after observations.")
            return False

        adjacency: Dict[str, List[str]] = {name: [] for name in self._skill_order}
        for parent, child in edges:
            adjacency.setdefault(parent, []).append(child)
            adjacency.setdefault(child, []).append(parent)

        n = len(self._skill_order)
        covariance = [[0.0] * n for _ in range(n)]
        for i, source in enumerate(self._skill_order):
            distances = self._shortest_distances(source, adjacency)
            for j, target in enumerate(self._skill_order):
                if i == j:
                    covariance[i][j] = self.prior_variance
                    continue
                distance = distances.get(target)
                if distance is None:
                    covariance[i][j] = 0.0
                else:
                    covariance[i][j] = (
                        self.prior_variance
                        * self.graph_correlation
                        * math.exp(-distance / self.graph_length_scale)
                    )

        self._skill_mean = [self.prior_mean] * n
        self._skill_covariance = covariance
        self._graph_signature = signature
        self._sync_skill_marginals()
        return True

    @staticmethod
    def _shortest_distances(source: str, adjacency: Mapping[str, List[str]]) -> Dict[str, int]:
        distances = {source: 0}
        queue = [source]
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    # ------------------------------------------------------------------
    # Bayesian updates
    # ------------------------------------------------------------------

    def update_skill(
        self,
        skill: str,
        score: float,
        max_score: float = 10.0,
        score_std: Optional[float] = None,
        observation_variance: Optional[float] = None,
    ) -> None:
        """
        Apply one exact Gaussian posterior update.

        ``score_std`` is expressed on the raw score scale.  For example, three
        judge scores with standard deviation 0.8 on a 0--10 scale should pass
        ``score_std=0.8``.  ``observation_variance`` is already normalized and
        takes precedence when supplied.
        """
        self.add_skill(skill)
        if max_score <= 0:
            raise ValueError("max_score must be positive")
        observed = min(max(float(score) / max_score, 0.0), 1.0)
        noise = self._resolve_observation_variance(
            max_score=max_score,
            score_std=score_std,
            observation_variance=observation_variance,
        )

        j = self._skill_index[skill]
        old_mean = list(self._skill_mean)
        old_cov = [list(row) for row in self._skill_covariance]
        innovation_variance = max(old_cov[j][j] + noise, _EPS)
        innovation = observed - old_mean[j]
        gain = [old_cov[i][j] / innovation_variance for i in range(len(old_mean))]

        self._skill_mean = [old_mean[i] + gain[i] * innovation for i in range(len(old_mean))]
        n = len(old_mean)
        new_cov = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                value = old_cov[i][k] - old_cov[i][j] * old_cov[j][k] / innovation_variance
                new_cov[i][k] = value

        # Remove small floating-point asymmetries and keep diagonal positive.
        for i in range(n):
            new_cov[i][i] = max(new_cov[i][i], _EPS)
            for k in range(i + 1, n):
                symmetric = 0.5 * (new_cov[i][k] + new_cov[k][i])
                new_cov[i][k] = symmetric
                new_cov[k][i] = symmetric

        self._skill_covariance = new_cov
        self.skills[skill].observations += 1
        self._sync_skill_marginals()

    def update_dimension(
        self,
        dimension: str,
        score: float,
        max_score: float = 10.0,
        score_std: Optional[float] = None,
        observation_variance: Optional[float] = None,
    ) -> None:
        self.add_dimension(dimension)
        if max_score <= 0:
            raise ValueError("max_score must be positive")
        ability = self.dimensions[dimension]
        observed = min(max(float(score) / max_score, 0.0), 1.0)
        noise = self._resolve_observation_variance(
            max_score=max_score,
            score_std=score_std,
            observation_variance=observation_variance,
        )
        posterior_variance = 1.0 / (1.0 / ability.variance + 1.0 / noise)
        posterior_mean = posterior_variance * (
            ability.mean / ability.variance + observed / noise
        )
        ability.mean = posterior_mean
        ability.variance = max(posterior_variance, _EPS)
        ability.observations += 1

    def _resolve_observation_variance(
        self,
        max_score: float,
        score_std: Optional[float],
        observation_variance: Optional[float],
    ) -> float:
        if observation_variance is not None:
            return max(float(observation_variance), _EPS)
        if score_std is not None:
            return max((float(score_std) / max_score) ** 2, _EPS)
        return self.observation_variance

    def _sync_skill_marginals(self) -> None:
        for name, i in self._skill_index.items():
            self.skills[name].mean = self._skill_mean[i]
            self.skills[name].variance = max(self._skill_covariance[i][i], _EPS)

    # ------------------------------------------------------------------
    # Acquisition statistics used by QuestionSelector
    # ------------------------------------------------------------------

    def get_skill_covariance(self, first: str, second: str) -> float:
        if first not in self._skill_index or second not in self._skill_index:
            return 0.0
        return self._skill_covariance[
            self._skill_index[first]
        ][self._skill_index[second]]

    def decision_variance(self, weights: Mapping[str, float]) -> float:
        """Posterior variance of z = sum_s weights[s] * theta_s."""
        total = 0.0
        for first, wf in weights.items():
            for second, ws in weights.items():
                total += wf * ws * self.get_skill_covariance(first, second)
        return max(total, 0.0)

    def expected_decision_variance_reduction(
        self,
        skill: str,
        weights: Mapping[str, float],
        observation_variance: Optional[float] = None,
    ) -> float:
        """Closed-form expected reduction in Var(w^T theta) after asking skill."""
        if skill not in self._skill_index:
            return 0.0
        noise = max(
            self.observation_variance if observation_variance is None else observation_variance,
            _EPS,
        )
        j = self._skill_index[skill]
        covariance_with_decision = sum(
            weight * self.get_skill_covariance(name, skill)
            for name, weight in weights.items()
        )
        denominator = self._skill_covariance[j][j] + noise
        return max(covariance_with_decision ** 2 / denominator, 0.0)

    def expected_total_variance_reduction(
        self,
        skill: str,
        observation_variance: Optional[float] = None,
    ) -> float:
        """Expected reduction in trace(Sigma) after asking one skill."""
        if skill not in self._skill_index:
            return 0.0
        noise = max(
            self.observation_variance if observation_variance is None else observation_variance,
            _EPS,
        )
        j = self._skill_index[skill]
        denominator = self._skill_covariance[j][j] + noise
        numerator = sum(row[j] ** 2 for row in self._skill_covariance)
        return max(numerator / denominator, 0.0)

    def total_skill_variance(self) -> float:
        return sum(max(self._skill_covariance[i][i], 0.0) for i in range(len(self._skill_order)))

    def get_job_weights(self, skill_graph=None) -> Dict[str, float]:
        """Return normalized weights over job-relevant leaf skills."""
        if not self.skills:
            return {}
        nodes = getattr(skill_graph, "nodes", {}) if skill_graph is not None else {}
        if nodes:
            leaves = [
                name for name, node in nodes.items()
                if not getattr(node, "children", []) and name in self.skills
            ]
            targets = leaves or list(self.skills)
            raw = {
                name: max(float(getattr(nodes.get(name), "weight", 1.0)), 0.0)
                for name in targets
            }
        else:
            raw = {name: 1.0 for name in self.skills}
        denominator = sum(raw.values()) or float(len(raw))
        return {name: value / denominator for name, value in raw.items()}

    # ------------------------------------------------------------------
    # Queries and serialization
    # ------------------------------------------------------------------

    def get_skill_info_gain(self, skill: str) -> float:
        return self.skills.get(skill, SkillAbility()).information_gain(
            self.observation_variance
        )

    def get_dimension_info_gain(self, dimension: str) -> float:
        return self.dimensions.get(dimension, SkillAbility()).information_gain(
            self.observation_variance
        )

    def get_weakest_skills(self, top_n: int = 3) -> List[Tuple[str, float]]:
        return sorted(
            [(name, ability.mean) for name, ability in self.skills.items()],
            key=lambda item: item[1],
        )[:top_n]

    def get_highest_uncertainty_skills(self, top_n: int = 3) -> List[Tuple[str, float]]:
        return sorted(
            [(name, ability.uncertainty) for name, ability in self.skills.items()],
            key=lambda item: -item[1],
        )[:top_n]

    def get_uncovered_skills(self) -> List[str]:
        return [name for name, ability in self.skills.items() if ability.observations == 0]

    def get_uncovered_dimensions(self) -> List[str]:
        return [name for name, ability in self.dimensions.items() if ability.observations == 0]

    def overall_confidence(self) -> float:
        abilities = list(self.skills.values()) + list(self.dimensions.values())
        if not abilities:
            return 0.0
        prior_std = math.sqrt(self.prior_variance)
        avg_ratio = sum(
            min(ability.uncertainty / prior_std, 1.0)
            for ability in abilities
        ) / len(abilities)
        return 1.0 - avg_ratio

    @staticmethod
    def _ability_summary(ability: SkillAbility) -> Dict:
        # Include old and new field names so existing callers remain usable.
        return {
            "mean": round(ability.mean, 4),
            "variance": round(ability.variance, 6),
            "std": round(ability.uncertainty, 4),
            "uncertainty": round(ability.uncertainty, 4),
            "n": ability.observations,
            "observations": ability.observations,
        }

    def summary(self) -> Dict:
        return {
            "model": "joint_gaussian_bayesian",
            "scale": "internal_[0,1]; reported_scores_may_be_rescaled_to_[0,10]",
            "prior_mean": self.prior_mean,
            "prior_std": round(math.sqrt(self.prior_variance), 4),
            "default_observation_std": round(math.sqrt(self.observation_variance), 4),
            "skills": {
                name: self._ability_summary(ability)
                for name, ability in self.skills.items()
            },
            "dimensions": {
                name: self._ability_summary(ability)
                for name, ability in self.dimensions.items()
            },
            "overall_confidence": round(self.overall_confidence(), 4),
            "uncovered_skills": self.get_uncovered_skills(),
            "uncovered_dimensions": self.get_uncovered_dimensions(),
        }
