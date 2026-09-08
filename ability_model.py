"""
ability_model.py
Candidate ability state model using Bayesian-style mean/uncertainty tracking.
Replaces the flat covered_skills list with a probabilistic ability estimate.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SkillAbility:
    """Ability estimate for a single skill or dimension."""
    mean: float = 0.5        # Estimated ability level [0, 1]
    uncertainty: float = 1.0 # Uncertainty / variance (high = less data)
    observations: int = 0    # Number of evaluations received

    def information_gain(self) -> float:
        """Higher uncertainty = more potential information gain from asking."""
        return self.uncertainty / (1.0 + self.observations * 0.3)

    def is_sufficiently_assessed(self, threshold: float = 0.3) -> bool:
        """True if uncertainty is low enough to consider skill assessed."""
        return self.uncertainty < threshold

    def __repr__(self):
        return f"SkillAbility(mean={self.mean:.2f}, uncertainty={self.uncertainty:.2f}, n={self.observations})"


class AbilityModel:
    """
    Tracks candidate ability across all skills and competency dimensions.

    Replaces:
        state.covered_skills  (List[str])
        state.covered_dimensions (List[Dict])

    With:
        ability_model.skills        (Dict[str, SkillAbility])
        ability_model.dimensions    (Dict[str, SkillAbility])
    """

    # How much a single answer updates the uncertainty
    UNCERTAINTY_DECAY = 0.25
    # Score thresholds for answer quality mapping
    SCORE_TO_ABILITY = {
        (0, 2):  0.15,
        (2, 4):  0.35,
        (4, 6):  0.55,
        (6, 8):  0.75,
        (8, 10): 0.90,
    }

    def __init__(self):
        self.skills: Dict[str, SkillAbility] = {}
        self.dimensions: Dict[str, SkillAbility] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_from_job_analysis(self, job_analysis: Dict) -> None:
        """
        Seed the model with skills/dimensions from job analysis.
        All start at prior: mean=0.5, uncertainty=1.0.
        """
        for skill in job_analysis.get("required_skills", []):
            self.skills[skill] = SkillAbility()

        for skill in job_analysis.get("preferred_skills", []):
            self.skills[skill] = SkillAbility(mean=0.5, uncertainty=0.8)

        for dim in job_analysis.get("competency_dimensions", []):
            dim_name = dim if isinstance(dim, str) else dim.get("dimension", str(dim))
            self.dimensions[dim_name] = SkillAbility()

    def add_skill(self, skill: str) -> None:
        """Lazily register a skill encountered during the interview."""
        if skill not in self.skills:
            self.skills[skill] = SkillAbility()

    def add_dimension(self, dimension: str) -> None:
        if dimension not in self.dimensions:
            self.dimensions[dimension] = SkillAbility()

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_skill(self, skill: str, score: float, max_score: float = 10.0) -> None:
        """
        Update ability estimate for a skill after receiving an answer score.

        Uses a running weighted average:
            new_mean = (old_mean * n + observed) / (n + 1)
            uncertainty *= DECAY

        Args:
            skill: Skill identifier
            score: Raw score from AnswerEvaluatorAgent
            max_score: Maximum possible score (for normalization)
        """
        self.add_skill(skill)
        ability = self.skills[skill]
        observed = score / max_score  # Normalize to [0, 1]
        self._update_ability(ability, observed)

    def update_dimension(self, dimension: str, score: float, max_score: float = 10.0) -> None:
        """Update ability estimate for a competency dimension."""
        self.add_dimension(dimension)
        ability = self.dimensions[dimension]
        observed = score / max_score
        self._update_ability(ability, observed)

    def _update_ability(self, ability: SkillAbility, observed: float) -> None:
        """Bayesian-style update: weighted running mean + uncertainty decay."""
        n = ability.observations
        ability.mean = (ability.mean * n + observed) / (n + 1)
        ability.uncertainty = max(0.05, ability.uncertainty * (1 - self.UNCERTAINTY_DECAY))
        ability.observations += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_skill_info_gain(self, skill: str) -> float:
        """Information gain potential for a skill (higher = more worth asking)."""
        return self.skills.get(skill, SkillAbility()).information_gain()

    def get_dimension_info_gain(self, dimension: str) -> float:
        return self.dimensions.get(dimension, SkillAbility()).information_gain()

    def get_weakest_skills(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Return skills with lowest ability mean (candidate is weakest here)."""
        return sorted(
            [(s, a.mean) for s, a in self.skills.items()],
            key=lambda x: x[1]
        )[:top_n]

    def get_highest_uncertainty_skills(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Return skills where we have the least information."""
        return sorted(
            [(s, a.information_gain()) for s, a in self.skills.items()],
            key=lambda x: -x[1]
        )[:top_n]

    def get_uncovered_skills(self) -> List[str]:
        """Skills with zero observations (never asked about)."""
        return [s for s, a in self.skills.items() if a.observations == 0]

    def get_uncovered_dimensions(self) -> List[str]:
        return [d for d, a in self.dimensions.items() if a.observations == 0]

    def overall_confidence(self) -> float:
        """
        Average certainty across all tracked abilities.
        Returns value in [0, 1] where 1 = fully assessed.
        """
        all_abilities = list(self.skills.values()) + list(self.dimensions.values())
        if not all_abilities:
            return 0.0
        avg_uncertainty = sum(a.uncertainty for a in all_abilities) / len(all_abilities)
        return 1.0 - min(avg_uncertainty, 1.0)

    def summary(self) -> Dict:
        """Structured summary of current ability state."""
        return {
            "skills": {
                s: {"mean": round(a.mean, 2), "uncertainty": round(a.uncertainty, 2), "n": a.observations}
                for s, a in self.skills.items()
            },
            "dimensions": {
                d: {"mean": round(a.mean, 2), "uncertainty": round(a.uncertainty, 2), "n": a.observations}
                for d, a in self.dimensions.items()
            },
            "overall_confidence": round(self.overall_confidence(), 2),
            "uncovered_skills": self.get_uncovered_skills(),
            "uncovered_dimensions": self.get_uncovered_dimensions(),
        }
