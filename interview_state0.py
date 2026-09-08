"""
interview_state.py  (updated)
Extends original InterviewState with AbilityModel, SkillGraph,
and follow-up tracking needed by InterviewPolicy.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from ability_model import AbilityModel
from skill_graph import SkillGraph


@dataclass
class InterviewState:
    """
    Complete interview session state.

    New fields vs original:
        ability_model         → replaces covered_skills / covered_dimensions
        skill_graph           → replaces flat required_skills list
        last_answer_score     → feeds InterviewPolicy follow-up decision
        last_question_skill   → feeds InterviewPolicy follow-up decision
        followup_counts       → per-skill follow-up counter
        question_metadata     → richer per-question record (skill, type, score)
    """

    # ── Existing fields (unchanged) ──────────────────────────────────
    job_analysis: Dict[str, Any] = field(default_factory=dict)
    experience_units: List[Dict[str, Any]] = field(default_factory=list)
    dialogue_history: List[Tuple[str, str]] = field(default_factory=list)
    current_experience_index: int = 0
    interview_phase: str = "opening"
    question_count: int = 0

    # ── Replaced fields ──────────────────────────────────────────────
    # covered_skills: List[str]       → REMOVED, use ability_model.skills
    # covered_dimensions: List[Dict]  → REMOVED, use ability_model.dimensions

    # ── New fields ───────────────────────────────────────────────────
    ability_model: AbilityModel = field(default_factory=AbilityModel)
    """Probabilistic ability state for all tracked skills and dimensions."""

    skill_graph: SkillGraph = field(default_factory=SkillGraph)
    """Skill topology enabling deep-dive and topic-switch navigation."""

    last_answer_score: Optional[float] = None
    """Score from AnswerEvaluatorAgent for the most recent answer."""

    last_question_skill: Optional[str] = None
    """Skill/dimension targeted by the most recent question."""

    last_question_type: Optional[str] = None
    """'technical' or 'competency'"""

    followup_counts: Dict[str, int] = field(default_factory=dict)
    """How many follow-up questions have been asked per skill."""

    question_metadata: List[Dict[str, Any]] = field(default_factory=list)
    """
    Per-question records: {
        'question': str,
        'answer': str,
        'skill': str,
        'question_type': str,   # technical / competency
        'is_followup': bool,
        'difficulty': str,
        'score': float | None,
    }
    """

    # ── Initialization ───────────────────────────────────────────────

    def initialize(self, job_analysis: Dict[str, Any]) -> None:
        """
        Called once after job_analysis is set.
        Seeds AbilityModel and SkillGraph from job description.
        """
        self.job_analysis = job_analysis
        self.ability_model.initialize_from_job_analysis(job_analysis)
        self.skill_graph.build_from_job_analysis(job_analysis)

    # ── Dialogue management ──────────────────────────────────────────

    def add_dialogue(self, question: str, answer: str,
                     skill: Optional[str] = None,
                     question_type: Optional[str] = None,
                     is_followup: bool = False,
                     difficulty: str = "medium") -> None:
        """Record a Q&A exchange with rich metadata."""
        self.dialogue_history.append((question, answer))
        self.question_count += 1
        self.question_metadata.append({
            "question": question,
            "answer": answer,
            "skill": skill,
            "question_type": question_type,
            "is_followup": is_followup,
            "difficulty": difficulty,
            "score": None,  # Filled in after evaluation
        })

    # ── Evaluation feedback loop ─────────────────────────────────────

    def record_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """
        Called after AnswerEvaluatorAgent runs.
        Updates ability model and stores score for policy decision.

        Expected evaluation keys (from evaluator agents):
            评分 (float): numeric score
            评估类型: '技术评估' | '能力评估'
            考察目标: skill or dimension name
        """
        raw_score = evaluation.get("评分", 5.0)
        eval_type = evaluation.get("评估类型", "")
        target = evaluation.get("考察目标", "")

        # Update ability model
        if eval_type == "技术评估":
            self.ability_model.update_skill(target, raw_score)
        elif eval_type == "能力评估":
            self.ability_model.update_dimension(target, raw_score)

        # Update state for policy
        self.last_answer_score = raw_score
        self.last_question_skill = target
        self.last_question_type = "technical" if eval_type == "技术评估" else "competency"

        # Update follow-up counter if this was a follow-up
        if self.question_metadata and self.question_metadata[-1].get("is_followup"):
            self.followup_counts[target] = self.followup_counts.get(target, 0) + 1

        # Back-fill score on the last question record
        if self.question_metadata:
            self.question_metadata[-1]["score"] = raw_score

    # ── Experience navigation (unchanged) ────────────────────────────

    def has_more_experiences(self) -> bool:
        return self.current_experience_index < len(self.experience_units)

    def get_current_experience(self) -> Dict[str, Any]:
        if self.has_more_experiences():
            return self.experience_units[self.current_experience_index]
        return {}

    def move_to_next_experience(self) -> None:
        self.current_experience_index += 1

    # ── Summary / debug ──────────────────────────────────────────────

    def get_progress_summary(self) -> Dict[str, Any]:
        return {
            "phase": self.interview_phase,
            "question_count": self.question_count,
            "ability_state": self.ability_model.summary(),
            "followup_counts": self.followup_counts,
            "last_score": self.last_answer_score,
        }
