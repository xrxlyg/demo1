"""
interview_policy.py

适配新版 question_selector.py：
- 移除 SelectionWeights 引用（新版用 Dict[str, float]）
- QuestionSelector.select() 签名改为：
      select(ability_model, candidates, question_type,
             target_difficulty, recent_skills, last_skill, followup_counts)
- AbilityModel 改为从 question_selector 引入（新版统一定义在那里）
- SkillGraph 保留引用，传给 select_with_graph
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from skill_graph import SkillGraph, TraversalState
from question_selector import (
    QuestionSelector,
    QuestionTarget,
    AbilityModel,
)


FOLLOWUP_SCORE_MIN       = 3.5
FOLLOWUP_SCORE_MAX       = 6.5
MAX_FOLLOWUPS_PER_SKILL  = 2
MIN_TECHNICAL_QUESTIONS  = 3
MIN_COMPETENCY_QUESTIONS = 2


@dataclass
class PolicyInput:
    """All data the policy needs to make a decision."""
    ability_model:       AbilityModel
    skill_graph:         SkillGraph
    dialogue_history:    List[Dict]
    interview_phase:     str
    question_count:      int
    last_answer_score:   Optional[float]     = None
    last_question_skill: Optional[str]       = None
    followup_counts:     Dict[str, int]      = field(default_factory=dict)
    # Fix 1: orchestrator 的 adaptive difficulty 传入，policy 直接使用
    current_difficulty:  str                 = "medium"


class InterviewPolicy:
    """
    Determines what to ask next based on candidate ability state.

    Decision flow:
        1. Resolve current phase (stay / transition)
        2. Check if follow-up is needed (middle-range score)
        3. Call QuestionSelector with the real current_difficulty from PolicyInput
    """

    def __init__(
        self,
        selector:         Optional[QuestionSelector] = None,
        traversal_state:  Optional[TraversalState]   = None,
    ):
        self.selector        = selector or QuestionSelector()
        self.traversal_state = traversal_state or TraversalState()

    def select_next_question(self, policy_input: PolicyInput) -> QuestionTarget:
        inp   = policy_input
        phase = self._resolve_phase(inp)

        # 近期问过的技能（用于 DR 计算）
        recent_skills = self._extract_recent_skills(inp.dialogue_history, window=6)

        # 是否需要追问
        followup_skill = self._should_followup(
            last_score=inp.last_answer_score,
            last_skill=inp.last_question_skill,
            followup_counts=inp.followup_counts,
        )

        # Fix 1: 使用 PolicyInput 中传入的 current_difficulty
        if phase == "competency":
            candidates = list(inp.ability_model.dimensions.keys())
            target = self.selector.select(
                ability_model=inp.ability_model,
                candidates=candidates,
                question_type="competency",
                target_difficulty=inp.current_difficulty,
                recent_skills=recent_skills,
                last_skill=inp.last_question_skill,
                followup_counts=inp.followup_counts,
            )

        elif phase == "closing":
            # 直接返回 closing target，不走 selector
            target = QuestionTarget(
                target="closing",
                question_type="closing",
                difficulty=inp.current_difficulty,
                reasoning="policy: competency phase complete → closing",
            )

        else:
            # technical phase：图感知 argmax
            self.traversal_state.last_score = (
                inp.last_answer_score / 10.0
                if inp.last_answer_score is not None else 0.5
            )
            self.traversal_state.recent_skills   = recent_skills
            self.traversal_state.question_count  = inp.question_count
            self.traversal_state.interview_phase = phase

            target = self.selector.select_with_graph(
                ability_model=inp.ability_model,
                skill_graph=inp.skill_graph,
                question_type="technical",
                target_difficulty=inp.current_difficulty,
                recent_skills=recent_skills,
                current_skill=inp.last_question_skill,
                traversal_state=self.traversal_state,
                followup_counts=inp.followup_counts,
            )

        # 追问覆盖：如果 policy 判断需要追问，强制替换 target
        if followup_skill and not target.is_followup:
            target = QuestionTarget(
                target=followup_skill,
                question_type=target.question_type,
                difficulty=inp.current_difficulty,
                is_followup=True,
                reasoning=f"policy: followup forced for '{followup_skill}'",
            )

        return target

    # ── Phase management ──────────────────────────────────────────────────────

    def _resolve_phase(self, inp: PolicyInput) -> str:
        phase = inp.interview_phase

        if phase == "opening":
            return "technical"

        if phase == "technical":
            return "competency" if self._technical_phase_complete(inp) else "technical"

        if phase == "competency":
            return "closing" if self._competency_phase_complete(inp) else "competency"

        return phase

    def _technical_phase_complete(self, inp: PolicyInput) -> bool:
        if inp.question_count < MIN_TECHNICAL_QUESTIONS:
            return False
        # 还有未评估的技能 → 未完成
        uncovered = [
            s for s, e in inp.ability_model.skills.items()
            if e.observations == 0
        ]
        return len(uncovered) == 0

    def _competency_phase_complete(self, inp: PolicyInput) -> bool:
        if inp.question_count < MIN_COMPETENCY_QUESTIONS:
            return False
        uncovered = [
            d for d, e in inp.ability_model.dimensions.items()
            if e.observations == 0
        ]
        return len(uncovered) == 0

    # ── Follow-up logic ───────────────────────────────────────────────────────

    def _should_followup(
        self,
        last_score:     Optional[float],
        last_skill:     Optional[str],
        followup_counts: Dict[str, int],
    ) -> Optional[str]:
        if last_score is None or last_skill is None:
            return None
        in_middle   = FOLLOWUP_SCORE_MIN <= last_score <= FOLLOWUP_SCORE_MAX
        under_limit = followup_counts.get(last_skill, 0) < MAX_FOLLOWUPS_PER_SKILL
        return last_skill if (in_middle and under_limit) else None

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_recent_skills(
        dialogue_history: List[Dict], window: int = 6
    ) -> List[str]:
        """从对话历史中提取最近 window 轮的技能名称。"""
        recent = []
        for item in reversed(dialogue_history[-window:]):
            skill = item.get("skill") or item.get("target")
            if skill:
                recent.insert(0, skill)
        return recent