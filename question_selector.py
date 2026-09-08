"""
Decision-risk question selection for adaptive technical interviews.

This replaces the former weighted heuristic (IG + CG + DM + DR plus graph
bonuses).  A candidate question is valued by the expected Bayesian reduction
in (1) job-decision variance and (2) total skill uncertainty.  Difficulty
mismatch increases expected observation noise, and conversational repetition
is represented as an explicit question cost.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from ability_model import AbilityModel, SkillAbility


logger = logging.getLogger(__name__)
_EPS = 1e-12

DIFFICULTY_TARGET: Dict[str, float] = {
    "easy": 0.30,
    "medium": 0.55,
    "hard": 0.80,
}

# Utility = decision * relative decision-risk reduction
#         + global   * relative trace reduction, divided by question cost.
DEFAULT_WEIGHTS = {"decision": 0.70, "global": 0.30}


@dataclass
class QuestionTarget:
    target: str
    question_type: str = "technical"
    difficulty: str = "medium"
    is_followup: bool = False
    score: float = 0.0
    reasoning: str = ""
    components: Dict = field(default_factory=dict)


def _difficulty_efficiency(mean: float, difficulty: str, bandwidth: float = 0.22) -> float:
    """
    Expected measurement efficiency of an item at the current ability.

    Questions far above or below current ability are treated as noisier rather
    than receiving an arbitrary additive difficulty bonus.
    """
    target = DIFFICULTY_TARGET.get(difficulty, DIFFICULTY_TARGET["medium"])
    z = (mean - target) / max(bandwidth, _EPS)
    return 0.15 + 0.85 * math.exp(-0.5 * z * z)


def _recent_count(skill: str, recent_skills: List[str]) -> int:
    return sum(1 for name in recent_skills if name == skill)


def _dimension_score(
    ability_model: AbilityModel,
    dimension: str,
    target_difficulty: str,
    recent_skills: List[str],
    weights: Mapping[str, float],
    repeat_cost: float,
) -> Tuple[float, Dict]:
    ability = ability_model.dimensions.get(dimension, SkillAbility())
    efficiency = _difficulty_efficiency(ability.mean, target_difficulty)
    noise = ability_model.observation_variance / efficiency
    reduction = ability.variance ** 2 / max(ability.variance + noise, _EPS)
    relative_reduction = reduction / max(ability.variance, _EPS)
    cost = 1.0 + repeat_cost * _recent_count(dimension, recent_skills)
    total = relative_reduction / cost
    return total, {
        "decision_reduction": round(reduction, 6),
        "global_reduction": round(reduction, 6),
        "relative_decision_reduction": round(relative_reduction, 6),
        "relative_global_reduction": round(relative_reduction, 6),
        "difficulty_efficiency": round(efficiency, 6),
        "observation_variance": round(noise, 6),
        "question_cost": round(cost, 6),
        "total": round(total, 6),
    }


def score_question_bayesian(
    skill: str,
    ability_model: AbilityModel,
    recent_skills: List[str],
    target_difficulty: str,
    skill_graph=None,
    weights: Optional[Mapping[str, float]] = None,
    repeat_cost: float = 0.15,
) -> Tuple[float, Dict]:
    """Score a technical target by closed-form expected posterior risk reduction."""
    ability_model.configure_skill_graph(skill_graph)
    ability = ability_model.skills.get(skill, SkillAbility())
    utility_weights = dict(weights or DEFAULT_WEIGHTS)
    job_weights = ability_model.get_job_weights(skill_graph)

    efficiency = _difficulty_efficiency(ability.mean, target_difficulty)
    effective_noise = ability_model.observation_variance / efficiency

    decision_reduction = ability_model.expected_decision_variance_reduction(
        skill,
        job_weights,
        observation_variance=effective_noise,
    )
    global_reduction = ability_model.expected_total_variance_reduction(
        skill,
        observation_variance=effective_noise,
    )
    decision_variance = ability_model.decision_variance(job_weights)
    total_variance = ability_model.total_skill_variance()

    relative_decision = decision_reduction / max(decision_variance, _EPS)
    relative_global = global_reduction / max(total_variance, _EPS)
    cost = 1.0 + repeat_cost * _recent_count(skill, recent_skills)

    total = (
        utility_weights["decision"] * relative_decision
        + utility_weights["global"] * relative_global
    ) / cost

    return total, {
        "decision_reduction": round(decision_reduction, 6),
        "global_reduction": round(global_reduction, 6),
        "relative_decision_reduction": round(relative_decision, 6),
        "relative_global_reduction": round(relative_global, 6),
        "difficulty_efficiency": round(efficiency, 6),
        "observation_variance": round(effective_noise, 6),
        "question_cost": round(cost, 6),
        "total": round(total, 6),
    }


class QuestionSelector:
    """Select the question with maximum expected Bayesian risk reduction."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        repeat_cost: float = 0.15,
        followup_boost: Optional[float] = None,
    ) -> None:
        if weights and {"alpha", "beta", "gamma", "delta"}.intersection(weights):
            logger.warning(
                "Legacy alpha/beta/gamma/delta selector weights are ignored; "
                "use {'decision': ..., 'global': ...}."
            )
            weights = None
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        if "decision" not in self.weights or "global" not in self.weights:
            raise ValueError("selector weights require 'decision' and 'global'")
        weight_sum = self.weights["decision"] + self.weights["global"]
        if weight_sum <= 0:
            raise ValueError("selector weights must have a positive sum")
        self.weights = {
            "decision": self.weights["decision"] / weight_sum,
            "global": self.weights["global"] / weight_sum,
        }
        self.repeat_cost = max(float(repeat_cost), 0.0)
        if followup_boost is not None:
            logger.warning("followup_boost is deprecated; repetition is modeled as cost.")

    def _score(
        self,
        ability_model: AbilityModel,
        target: str,
        question_type: str,
        target_difficulty: str,
        recent_skills: List[str],
        skill_graph=None,
    ) -> Tuple[float, Dict]:
        if question_type == "technical":
            return score_question_bayesian(
                skill=target,
                ability_model=ability_model,
                recent_skills=recent_skills,
                target_difficulty=target_difficulty,
                skill_graph=skill_graph,
                weights=self.weights,
                repeat_cost=self.repeat_cost,
            )
        return _dimension_score(
            ability_model=ability_model,
            dimension=target,
            target_difficulty=target_difficulty,
            recent_skills=recent_skills,
            weights=self.weights,
            repeat_cost=self.repeat_cost,
        )

    def select(
        self,
        ability_model: AbilityModel,
        candidates: List[str],
        question_type: str,
        target_difficulty: str,
        recent_skills: List[str],
        last_skill: Optional[str] = None,
        followup_counts: Optional[Dict[str, int]] = None,
        max_followup: int = 2,
    ) -> QuestionTarget:
        if not candidates:
            return QuestionTarget(
                target="unknown",
                question_type="closing",
                reasoning="No candidates available.",
            )

        followups = followup_counts or {}
        scored = []
        for target in candidates:
            if (
                len(candidates) > 1
                and target == last_skill
                and followups.get(target, 0) >= max_followup
            ):
                continue
            total, components = self._score(
                ability_model,
                target,
                question_type,
                target_difficulty,
                recent_skills,
            )
            scored.append((total, target, components))

        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_target, best_components = scored[0]
        reasoning = (
            f"Bayesian risk reduction argmax '{best_target}' score={best_score:.6f} | "
            f"decision={best_components['relative_decision_reduction']:.4f} "
            f"global={best_components['relative_global_reduction']:.4f} "
            f"efficiency={best_components['difficulty_efficiency']:.4f} "
            f"cost={best_components['question_cost']:.3f}"
        )
        return QuestionTarget(
            target=best_target,
            question_type=question_type,
            difficulty=target_difficulty,
            is_followup=(best_target == last_skill),
            score=best_score,
            reasoning=reasoning,
            components=best_components,
        )

    def select_with_graph(
        self,
        ability_model: AbilityModel,
        skill_graph,
        question_type: str,
        target_difficulty: str,
        recent_skills: List[str],
        current_skill: Optional[str],
        traversal_state,
        followup_counts: Optional[Dict[str, int]] = None,
        max_followup: int = 2,
    ) -> QuestionTarget:
        ability_model.configure_skill_graph(skill_graph)
        traversal_state.last_score = (
            ability_model.skills.get(current_skill, SkillAbility()).mean
            if current_skill else 0.5
        )

        if current_skill:
            next_from_graph, strategy = skill_graph.next_node(current_skill, traversal_state)
        else:
            next_from_graph = skill_graph.topic_switch()
            strategy = "topic_switch_init"

        local: List[str] = []
        if next_from_graph:
            node = skill_graph.get(next_from_graph)
            if node and node.parent:
                local = [child.name for child in skill_graph.children_of(node.parent)]
            else:
                local = [next_from_graph]

        candidates = list(dict.fromkeys(local + [root.name for root in skill_graph.roots()]))
        if not candidates:
            candidates = skill_graph.all_names()

        followups = followup_counts or {}
        scored = []
        for target in candidates:
            if (
                len(candidates) > 1
                and target == current_skill
                and followups.get(target, 0) >= max_followup
            ):
                continue
            total, components = self._score(
                ability_model,
                target,
                question_type,
                target_difficulty,
                recent_skills,
                skill_graph,
            )
            scored.append((total, target, components))

        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_target, best_components = scored[0]
        reasoning = (
            f"[{strategy}] Bayesian risk argmax '{best_target}' score={best_score:.6f} | "
            f"decision={best_components['relative_decision_reduction']:.4f} "
            f"global={best_components['relative_global_reduction']:.4f} "
            f"efficiency={best_components['difficulty_efficiency']:.4f} "
            f"cost={best_components['question_cost']:.3f}"
        )
        logger.info("select_with_graph | best=%s score=%.6f | %s", best_target, best_score, reasoning)
        return QuestionTarget(
            target=best_target,
            question_type=question_type,
            difficulty=target_difficulty,
            is_followup=(best_target == current_skill),
            score=best_score,
            reasoning=reasoning,
            components=best_components,
        )


def score_all(
    ability_model: AbilityModel,
    candidates: List[str],
    question_type: str,
    target_difficulty: str,
    recent_skills: List[str],
    weights: Optional[Dict[str, float]] = None,
    skill_graph=None,
) -> List[Dict]:
    selector = QuestionSelector(weights=weights)
    results = []
    for target in candidates:
        total, components = selector._score(
            ability_model,
            target,
            question_type,
            target_difficulty,
            recent_skills,
            skill_graph,
        )
        ability = (
            ability_model.skills.get(target, SkillAbility())
            if question_type == "technical"
            else ability_model.dimensions.get(target, SkillAbility())
        )
        results.append({
            "skill": target,
            **components,
            "estimate": {
                "mean": round(ability.mean, 4),
                "variance": round(ability.variance, 6),
                "std": round(ability.uncertainty, 4),
                "observations": ability.observations,
            },
        })
    results.sort(key=lambda item: (-item["total"], item["skill"]))
    return results
