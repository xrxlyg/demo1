"""
interview_state.py  (v6 适配版)

适配改动：
1. AbilityModel 从 question_selector 引入（新版统一定义在那里）
   - 移除 ability_model.initialize_from_job_analysis()
   - initialize() 只存 job_analysis，AbilityModel / SkillGraph 由 orchestrator 负责构建
2. SkillGraph 从 skill_graph 引入（新版）
   - 移除 skill_graph.build_from_job_analysis()
   - state 里的 skill_graph 由 orchestrator 在 Stage 2/4 赋值
3. dialogue_history 类型从 List[Tuple] 改为 List[Dict]
   - add_dialogue 存字典，fix 原有元组存储导致的解包错误
4. record_evaluation 移除对 ability_model.update_skill/dimension 的调用
   - 新版 ability_model 更新统一由 orchestrator._update_ability_model() 负责
   - state 只负责记录 last_answer_score / last_question_skill / question_metadata
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from skill_graph import SkillGraph
from question_selector import AbilityModel


@dataclass
class InterviewState:
    """
    Complete interview session state.

    职责划分（v6）：
        InterviewState   — 存储运行时状态（对话历史、评分记录、阶段追踪）
        AbilityModel     — 由 orchestrator 构建并赋值到 state.ability_model
        SkillGraph       — 由 orchestrator 构建并赋值到 state.skill_graph
        AbilityModel更新 — 由 orchestrator._update_ability_model() 负责
    """

    # ── 基础字段 ──────────────────────────────────────────────────────
    job_analysis:             Dict[str, Any]       = field(default_factory=dict)
    experience_units:         List[Dict[str, Any]] = field(default_factory=list)
    dialogue_history:         List[Dict[str, Any]] = field(default_factory=list)  # 存字典
    current_experience_index: int                  = 0
    interview_phase:          str                  = "opening"
    question_count:           int                  = 0

    # ── 由 orchestrator 赋值的组件 ────────────────────────────────────
    ability_model: Optional[AbilityModel] = None
    """由 orchestrator Stage 4 构建后赋值。"""

    skill_graph:   Optional[SkillGraph]   = None
    """由 orchestrator Stage 2 构建后赋值。"""

    # ── Policy 决策所需的状态 ─────────────────────────────────────────
    last_answer_score:   Optional[float] = None
    last_question_skill: Optional[str]   = None
    last_question_type:  Optional[str]   = None
    followup_counts:     Dict[str, int]  = field(default_factory=dict)

    question_metadata: List[Dict[str, Any]] = field(default_factory=list)
    """
    每题记录：{
        'question':      str,
        'answer':        str,
        'skill':         str | None,
        'question_type': str,        # technical / competency / opening
        'is_followup':   bool,
        'difficulty':    str,
        'score':         float | None,
    }
    """

    # ── 初始化 ────────────────────────────────────────────────────────

    def initialize(self, job_analysis: Dict[str, Any]) -> None:
        """
        存储 job_analysis。

        v6 说明：
            AbilityModel 和 SkillGraph 的构建由 orchestrator 负责
            （Stage 2: SkillTreeAgent，Stage 4: AbilityModel init），
            构建完成后 orchestrator 会直接赋值：
                state.ability_model = AbilityModel(...)
                state.skill_graph   = skill_graph
            这里不再调用 build_from_job_analysis / initialize_from_job_analysis。
        """
        self.job_analysis             = job_analysis
        self.experience_units         = job_analysis.get("experience_units", [])
        self.current_experience_index = 0
        self.interview_phase          = "opening"
        self.question_count           = 0
        self.dialogue_history         = []
        self.question_metadata        = []
        self.followup_counts          = {}
        self.last_answer_score        = None
        self.last_question_skill      = None
        self.last_question_type       = None

    # ── 对话记录 ──────────────────────────────────────────────────────

    def add_dialogue(
        self,
        question:      str,
        answer:        str,
        skill:         Optional[str] = None,
        question_type: Optional[str] = None,
        is_followup:   bool          = False,
        difficulty:    str           = "medium",
    ) -> None:
        """记录一轮问答，以字典形式存入 dialogue_history。"""
        record = {
            "question":      question,
            "answer":        answer,
            "skill":         skill,
            "question_type": question_type,
            "is_followup":   is_followup,
            "difficulty":    difficulty,
            "score":         None,  # 评估后由 record_evaluation 回填
        }
        self.dialogue_history.append(record)
        self.question_count += 1
        self.question_metadata.append(record)

    # ── 评估回调 ──────────────────────────────────────────────────────

    def record_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """
        AnswerEvaluatorAgent 运行后调用。

        v6 说明：
            AbilityModel 的 update_skill / update_dimension 由
            orchestrator._update_ability_model() 统一负责。
            这里只负责：
                1. 更新 last_answer_score / last_question_skill / last_question_type
                2. 回填最后一条 question_metadata 的 score 字段
                3. 更新 followup_counts
        """
        raw_score = evaluation.get("评分", 5.0)
        eval_type = evaluation.get("评估类型", "")
        target    = evaluation.get("考察目标", "")

        self.last_answer_score   = raw_score
        self.last_question_skill = target
        self.last_question_type  = (
            "technical" if eval_type == "技术评估" else "competency"
        )

        # 如果是追问，更新追问计数
        if self.question_metadata and self.question_metadata[-1].get("is_followup"):
            self.followup_counts[target] = self.followup_counts.get(target, 0) + 1

        # 回填得分
        if self.question_metadata:
            self.question_metadata[-1]["score"] = raw_score

    # ── 经历导航 ──────────────────────────────────────────────────────

    def has_more_experiences(self) -> bool:
        return self.current_experience_index < len(self.experience_units)

    def get_current_experience(self) -> Dict[str, Any]:
        if self.has_more_experiences():
            return self.experience_units[self.current_experience_index]
        return {}

    def move_to_next_experience(self) -> None:
        self.current_experience_index += 1

    # ── 摘要 / 调试 ───────────────────────────────────────────────────

    def get_progress_summary(self) -> Dict[str, Any]:
        ability_summary = (
            self.ability_model.summary() if self.ability_model else {}
        )
        return {
            "phase":          self.interview_phase,
            "question_count": self.question_count,
            "ability_state":  ability_summary,
            "followup_counts": self.followup_counts,
            "last_score":     self.last_answer_score,
        }