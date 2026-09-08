"""
interview_orchestrator.py  (v6)

"""

import os
import json
import logging
from dataclasses import replace
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from interview_state import InterviewState
from interview_policy import InterviewPolicy, PolicyInput

from skill_graph import SkillGraph, TraversalState
from question_selector import (
    QuestionTarget,
    QuestionSelector,
    AbilityModel,
    score_all,
)

from agents.answer_evaluator_agent import AnswerEvaluatorAgent
from agents.technical_question_agent import TechnicalQuestionAgent
from agents.competency_question_agent import CompetencyQuestionAgent
from agents.job_analyst_agent import JobAnalystAgent
from agents.dimension_expert_agent import DimensionExpertAgent
from agents.skill_tree_agent import SkillTreeAgent          # v6 新增

logger = logging.getLogger(__name__)


# ── 常量 ──────────────────────────────────────────────────────────────────────

MAX_TECHNICAL_QUESTIONS   = 12
MAX_COMPETENCY_QUESTIONS  = 8
MAX_TOTAL_QUESTIONS       = 25
EXPERIENCE_COVERAGE_THRESHOLD = 0.70

DIFFICULTY_UP_THRESHOLD   = 8.0
DIFFICULTY_DOWN_THRESHOLD = 5.0
DIFFICULTY_LADDER = ["easy", "medium", "hard"]

OPENING_QUESTION = (
    "请先简单介绍一下您最近的一段工作或项目经历，"
    "以及您在其中承担的主要职责。"
)
CLOSING_STATEMENT = (
    "非常感谢您今天参加面试，我们对您的经历和技能有了很好的了解。"
    "面试到此结束，您有什么想问我的吗？"
)

SKILL_ALIASES: Dict[str, List[str]] = {
    "redis":         ["缓存", "cache", "redis"],
    "kafka":         ["消息队列", "message queue", "mq", "kafka", "消息中间件"],
    "docker":        ["容器", "container", "docker", "镜像"],
    "kubernetes":    ["k8s", "kubernetes", "容器编排", "orchestration"],
    "mysql":         ["mysql", "关系型数据库", "rdb", "sql数据库"],
    "postgresql":    ["postgresql", "postgres", "pg"],
    "mongodb":       ["mongodb", "mongo", "文档数据库", "nosql"],
    "elasticsearch": ["elasticsearch", "es", "全文检索", "搜索引擎"],
    "spring":        ["spring", "spring框架", "ioc"],
    "spring boot":   ["spring boot", "springboot"],
    "分布式系统":    ["分布式", "distributed", "微服务", "microservice"],
    "消息队列":      ["消息队列", "message queue", "mq", "kafka", "rabbitmq", "rocketmq"],
    "数据库":        ["数据库", "database", "db", "mysql", "postgresql", "oracle"],
    "缓存":          ["缓存", "cache", "redis", "memcached"],
    "多线程":        ["多线程", "并发", "concurrent", "thread", "线程池"],
    "系统架构":      ["架构", "architecture", "系统设计", "system design"],
    "性能优化":      ["性能优化", "performance", "优化", "调优", "tuning"],
    "设计模式":      ["设计模式", "design pattern", "pattern"],
    "git":           ["git", "版本控制", "version control", "svn"],
    "ci/cd":         ["ci/cd", "cicd", "持续集成", "continuous integration", "jenkins"],
}


def _skill_fuzzy_match(skill: str, text: str) -> bool:
    skill_lower = skill.lower()
    text_lower  = text.lower()
    if skill_lower in text_lower:
        return True
    aliases = SKILL_ALIASES.get(skill_lower, [])
    return any(alias in text_lower for alias in aliases)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class InterviewOrchestrator:

    def __init__(
        self,
        llm_wrapper,
        skill_pool:       List[str],
        dimension_pool:   List[str],
        dimension_tree:   Dict[str, Any],
        save_training_data: bool = False,
        training_data_dir:  str  = "training_data",
        candidate_name:     str  = "candidate",
        max_questions:      int  = MAX_TOTAL_QUESTIONS,
        enforce_coverage_guard: bool = False,
        # Agent overrides（测试用）
        policy:           Optional[InterviewPolicy]        = None,
        evaluator:        Optional[AnswerEvaluatorAgent]   = None,
        tech_agent:       Optional[TechnicalQuestionAgent] = None,
        comp_agent:       Optional[CompetencyQuestionAgent] = None,
        job_analyst:      Optional[JobAnalystAgent]        = None,
        dim_expert:       Optional[DimensionExpertAgent]   = None,
        skill_tree_agent: Optional[SkillTreeAgent]         = None,
        selector_weights: Optional[Dict[str, float]]       = None,
    ):
        self.llm_wrapper    = llm_wrapper
        self.skill_pool     = skill_pool
        self.dimension_pool = dimension_pool
        self.dimension_tree = dimension_tree
        self.max_questions  = max_questions
        self.enforce_coverage_guard = bool(enforce_coverage_guard)

        self.save_training_data = save_training_data
        self.training_data_dir  = training_data_dir
        self.candidate_name     = candidate_name

        # ── Agents ────────────────────────────────────────────────────────
        self.state            = InterviewState()
        self.policy           = policy           or InterviewPolicy()
        self.evaluator        = evaluator        or AnswerEvaluatorAgent(llm_wrapper)
        self.tech_agent       = tech_agent       or TechnicalQuestionAgent(llm_wrapper)
        self.comp_agent       = comp_agent       or CompetencyQuestionAgent(llm_wrapper)
        self.job_analyst      = job_analyst      or JobAnalystAgent(llm_wrapper)
        self.dim_expert       = dim_expert       or DimensionExpertAgent(llm_wrapper)
        self.skill_tree_agent = skill_tree_agent or SkillTreeAgent(llm_wrapper)

        # ── Formalized components ─────────────────────────────────────────
        self._selector        = QuestionSelector(weights=selector_weights)
        self._skill_graph:    Optional[SkillGraph]   = None
        self._ability_model:  Optional[AbilityModel] = None
        self._traversal_state = TraversalState()

        # ── Runtime state ─────────────────────────────────────────────────
        self._current_question:   str  = ""
        self._finished:           bool = False
        self._current_difficulty: str  = "medium"

        self._experience_skill_hits: Dict[int, set] = {}
        self._skills_asked:     set = set()
        self._dimensions_asked: set = set()

        # ── Public data ───────────────────────────────────────────────────
        self.job_name:            str        = ""
        self.required_skills:     List[str]  = []
        self.required_dimensions: List[Dict] = []
        self.skill_hierarchy:     Dict       = {}

        self.answer_evaluations: List[Dict] = []
        self.training_samples:   List[Dict] = []

        self.skill_coverage:     Dict[str, Any] = {}
        self.dimension_coverage: Dict[str, Any] = {}

    # =========================================================================
    # 公开接口
    # =========================================================================

    def initialize_interview(self, job_profile: Dict[str, Any]) -> None:
        """
        四阶段串联初始化：

        Stage 1 — JobAnalystAgent:
            JD原文 → job_name / required_skills / mapped_dimensions

        Stage 2 — SkillTreeAgent（v6 新增，两阶段建树）:
            Phase 1: JD原文 → 宽树（领域 → 工具）
            Phase 2: 每个工具节点 → 深子树（工具 → 考察主题 → 具体考察点）

        Stage 3 — DimensionExpertAgent:
            mapped_dimensions → 带子维度的维度结构

        Stage 4 — 初始化 SkillGraph / AbilityModel / InterviewState
        """
        # ── Stage 1: JobAnalystAgent ──────────────────────────────────────
        logger.info("Stage 1: JobAnalystAgent analyzing JD...")
        job_analysis = self.job_analyst.run(
            job_profile=job_profile,
            # skill_pool=self.skill_pool,
            dimension_pool=self.dimension_pool,
        )

        self.job_name        = job_analysis.get("job_name", job_profile.get("job_name", "未知岗位"))
        self.required_skills = job_analysis.get("skills", {}).get("list", [])

        logger.info(
            "Stage 1 done | job=%s skills=%s",
            self.job_name, self.required_skills,
        )

        # ── Stage 2: SkillTreeAgent（两阶段建树）────────────────────────
        logger.info("Stage 2: SkillTreeAgent building skill tree (2-phase)...")
        self.skill_hierarchy = self.skill_tree_agent.run(job_analysis)

        if self.skill_hierarchy:
            self._skill_graph = SkillGraph.from_dict(self.skill_hierarchy)
            logger.info(
                "Stage 2 done | graph nodes=%d, roots=%d, max_depth=%d",
                len(self._skill_graph.nodes),
                len(self._skill_graph.roots()),
                self._compute_max_depth(),
            )
        else:
            logger.warning("Stage 2: empty tree, falling back to flat list.")
            self._skill_graph = SkillGraph.from_flat_list(self.required_skills)
            self.skill_hierarchy = {s: {"weight": 1.0} for s in self.required_skills}
        # ── Stage 3: DimensionExpertAgent ────────────────────────────────
        logger.info("Stage 3: DimensionExpertAgent mapping sub-dimensions...")
        dim_result = self.dim_expert.run(
            job_analysis=job_analysis,
            dimension_tree=self.dimension_tree,
        )

        raw_dims = dim_result.get("dimensions", [])
        if raw_dims:
            self.required_dimensions = raw_dims
        else:
            mapped = job_analysis.get("mapped_dimensions", {}).get("list", [])
            self.required_dimensions = [
                {"dimension": d, "sub_dimensions": self.dimension_tree.get(d, [])}
                for d in mapped
            ]

        logger.info(
            "Stage 3 done | dimensions=%s",
            [d.get("dimension") for d in self.required_dimensions],
        )

        # ── Stage 4: 初始化 AbilityModel / InterviewState ────────────────
        all_graph_skills = self._skill_graph.all_names()
        dim_names        = [d["dimension"] for d in self.required_dimensions]

        # self._ability_model = AbilityModel(
        #     skills=all_graph_skills,
        #     dimensions=dim_names,
        # )
        self._ability_model = AbilityModel()

        self._ability_model.initialize_from_job_analysis({
            "required_skills": all_graph_skills,
            "preferred_skills": [],
            "competency_dimensions": dim_names
        })
        job_analysis_for_state = {
            "job_name":              self.job_name,
            "job_id":                job_profile.get("job_id", ""),
            "recruit_type":          job_profile.get("recruit_type", ""),
            "required_skills":       self.required_skills,
            "preferred_skills":      [],
            "competency_dimensions": self.required_dimensions,
            "raw_text":              job_profile.get("raw_text", ""),
        }
        self.state.initialize(job_analysis_for_state)
        self.state.interview_phase = "opening"

        logger.info(
            "Stage 4 done | ability_model skills=%d dimensions=%d",
            len(all_graph_skills), len(dim_names),
        )

    def generate_opening_question(self) -> str:
        self._current_question = OPENING_QUESTION
        return OPENING_QUESTION

    def process_response(
        self,
        answer:   str,
        question: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict]]:
        if question:
            self._current_question = question

        if self._finished:
            return CLOSING_STATEMENT, None

        evaluation: Optional[Dict] = None
        current_phase = self.state.interview_phase

        # ── Step 1: opening ───────────────────────────────────────────────
        if current_phase == "opening":
            self._handle_opening_response(answer)
            self.state.interview_phase = "technical"

        # ── Step 2: 评估 + 更新 AbilityModel ─────────────────────────────
        else:
            evaluation = self._evaluate_and_record(self._current_question, answer)
            self._adapt_difficulty(self.state.last_answer_score)

        # ── Step 3: 硬终止 ────────────────────────────────────────────────
        if self._should_end_interview():
            self._finish_interview()
            return CLOSING_STATEMENT, evaluation

        # ── Step 4: 图感知 argmax 选题 ────────────────────────────────────
        target = self._select_next_question()

        # ── Step 5: Optional coverage constraint ─────────────────────────
        # Disabled by default so it cannot overwrite the Bayesian argmax and
        # artificially create a coverage advantage in policy comparisons.
        if self.enforce_coverage_guard and not target.is_followup:
            target = self._apply_coverage_guard(target)

        # ── Step 6: Phase 同步 ────────────────────────────────────────────
        self._sync_phase(target)

        if self.state.interview_phase == "closing":
            self._finish_interview()
            return CLOSING_STATEMENT, evaluation

        # ── Step 7: Experience coverage ───────────────────────────────────
        self._maybe_advance_experience(target)

        # ── Step 8: 生成问题文本 ──────────────────────────────────────────
        next_question = self._generate_question(target)

        # ── Step 9: 记录已出题 ────────────────────────────────────────────
        self._record_question_asked(target)

        # ── Step 10: 更新状态 ─────────────────────────────────────────────
        self._current_question         = next_question
        self.state.last_question_skill = target.target
        self.state.last_question_type  = target.question_type

        if target.is_followup:
            self.state.followup_counts[target.target] = (
                self.state.followup_counts.get(target.target, 0) + 1
            )

        self._traversal_state.recent_skills   = self._get_recent_skills(window=6)
        self._traversal_state.question_count  = self.state.question_count
        self._traversal_state.interview_phase = self.state.interview_phase

        self._refresh_coverage_views()

        logger.info(
            "Q#%d | phase=%s skill=%s followup=%s difficulty=%s score=%.4f | %s",
            self.state.question_count + 1,
            self.state.interview_phase,
            target.target, target.is_followup,
            target.difficulty, target.score,
            target.reasoning,
        )

        return next_question, evaluation

    def generate_overall_evaluation(self) -> Dict[str, Any]:
        if not self._ability_model:
            return {}
        skills = self._ability_model.skills
        dims   = self._ability_model.dimensions

        skill_scores = {s: round(e.mean * 10, 1) for s, e in skills.items() if e.observations > 0}
        dim_scores   = {d: round(e.mean * 10, 1) for d, e in dims.items()   if e.observations > 0}

        all_obs = [e for e in list(skills.values()) + list(dims.values()) if e.observations > 0]
        mean    = sum(e.mean for e in all_obs) / len(all_obs) if all_obs else 0.5

        return {
            "candidate_name":   self.candidate_name,
            "job_name":         self.job_name,
            "interview_date":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_level":    self._mean_to_level(mean),
            "overall_score":    round(mean * 10, 1),
            "skill_scores":     skill_scores,
            "dimension_scores": dim_scores,
            "phases_completed": self._infer_phases_completed(),
            "ability_model":    self._ability_model.summary(),
            "skill_graph":      self._skill_graph.summary() if self._skill_graph else {},
            "recommendation":   self._make_recommendation(mean),
        }

    def get_interview_summary(self) -> Dict[str, Any]:
        cov_s = self._compute_skill_coverage()
        cov_d = self._compute_dimension_coverage()
        return {
            "总问题数":      self.state.question_count,
            "讨论经历数":    self.state.current_experience_index + 1,
            "覆盖技能数":    cov_s["covered"],
            "评估维度数":    cov_d["covered"],
            "当前难度":      self._current_difficulty,
            "最终阶段":      self.state.interview_phase,
            "技能覆盖率":    f"{cov_s['rate'] * 100:.0f}%",
            "维度覆盖率":    f"{cov_d['rate'] * 100:.0f}%",
            "SkillGraph":   self._skill_graph.summary() if self._skill_graph else {},
            "AbilityModel": self._ability_model.summary() if self._ability_model else {},
        }

    def get_selector_debug(self) -> List[Dict]:
        """返回当前所有候选技能的评分明细，用于调试。"""
        if not self._ability_model or not self._skill_graph:
            return []
        phase      = self.state.interview_phase
        q_type     = "technical" if phase != "competency" else "competency"
        candidates = (
            self._skill_graph.all_names()
            if q_type == "technical"
            else [d["dimension"] for d in self.required_dimensions]
        )
        return score_all(
            ability_model=self._ability_model,
            candidates=candidates,
            question_type=q_type,
            target_difficulty=self._current_difficulty,
            recent_skills=self._get_recent_skills(window=6),
            skill_graph=self._skill_graph if q_type == "technical" else None,
        )

    # =========================================================================
    # 核心选题（图感知 argmax）
    # =========================================================================

    def _select_next_question(self) -> QuestionTarget:
        phase      = self.state.interview_phase
        recent     = self._get_recent_skills(window=6)
        last_skill = self.state.last_question_skill

        if phase == "competency":
            candidates = [d["dimension"] for d in self.required_dimensions]
            return self._selector.select(
                ability_model=self._ability_model,
                candidates=candidates,
                question_type="competency",
                target_difficulty=self._current_difficulty,
                recent_skills=recent,
                last_skill=last_skill,
                followup_counts=self.state.followup_counts,
            )

        return self._selector.select_with_graph(
            ability_model=self._ability_model,
            skill_graph=self._skill_graph,
            question_type="technical",
            target_difficulty=self._current_difficulty,
            recent_skills=recent,
            current_skill=last_skill,
            traversal_state=self._traversal_state,
            followup_counts=self.state.followup_counts,
        )

    def _get_recent_skills(self, window: int = 6) -> List[str]:
        history = self.state.dialogue_history or []
        recent  = []
        for turn in reversed(history[-window:]):
            skill = turn.get("skill") or turn.get("target")
            if skill:
                recent.insert(0, skill)
        return recent

    # =========================================================================
    # AbilityModel 更新
    # =========================================================================

        # =========================================================================
    # AbilityModel 更新（FIXED）
    # =========================================================================

    def _update_ability_model(
        self, skill: str, question_type: str, raw_score: float
    ) -> None:
        """
        修复点：
        1. 移除重复 update（原来 update 两次）
        2. 正确从 AbilityModel 读取 mean
        3. 不再错误调用 update_skill 作为 getter
        """
        if not self._ability_model:
            return

        # ── Step 1: 更新能力模型 ─────────────────────────────
        if question_type == "technical":
            self._ability_model.update_skill(skill, raw_score)

            if self._skill_graph:
                self._skill_graph.record_observation(skill)

            ability = self._ability_model.skills.get(skill)

        else:
            self._ability_model.update_dimension(skill, raw_score)
            ability = self._ability_model.dimensions.get(skill)

        # ── Step 2: 更新 traversal_state ─────────────────────
        if ability:
            self._traversal_state.last_score = ability.mean

            logger.debug(
                "AbilityModel | %s '%s' → μ=%.3f σ=%.3f (n=%d)",
                question_type,
                skill,
                ability.mean,
                ability.uncertainty,
                ability.observations,
            )

    # =========================================================================
    # Fix 1: 难度自适应
    # =========================================================================

    def _adapt_difficulty(self, score: Optional[float]) -> None:
        if score is None:
            return
        idx = DIFFICULTY_LADDER.index(self._current_difficulty)
        if score > DIFFICULTY_UP_THRESHOLD:
            new_idx = min(idx + 1, len(DIFFICULTY_LADDER) - 1)
        elif score < DIFFICULTY_DOWN_THRESHOLD:
            new_idx = max(idx - 1, 0)
        else:
            return
        if new_idx != idx:
            old = self._current_difficulty
            self._current_difficulty = DIFFICULTY_LADDER[new_idx]
            logger.info("Difficulty: %s → %s (score=%.1f)", old, self._current_difficulty, score)

    # =========================================================================
    # Fix 2: Coverage guard（followup 跳过）
    # =========================================================================

    def _apply_coverage_guard(self, target: QuestionTarget) -> QuestionTarget:
        if target.question_type == "technical":
            asked     = self._skills_asked
            uncovered = [
                s for s, e in (self._ability_model.skills if self._ability_model else {}).items()
                if e.observations == 0 and s not in asked
            ]
        else:
            asked     = self._dimensions_asked
            uncovered = [
                d for d, e in (self._ability_model.dimensions if self._ability_model else {}).items()
                if e.observations == 0 and d not in asked
            ]

        if target.target in asked and uncovered:
            forced = uncovered[0]
            logger.info("Coverage guard: %s → %s", target.target, forced)
            return replace(
                target,
                target=forced,
                is_followup=False,
                reasoning=f"coverage guard: forced '{forced}'",
            )
        return target

    # =========================================================================
    # Fix 3: Experience coverage fuzzy matching
    # =========================================================================

    def _maybe_advance_experience(self, target: QuestionTarget) -> None:
        idx        = self.state.current_experience_index
        experience = self.state.get_current_experience()
        if not experience:
            return
        exp_skills = self._get_experience_skills(experience)
        if not exp_skills:
            if self._experience_skill_hits.get(idx):
                self.state.move_to_next_experience()
            return
        hit_set  = self._experience_skill_hits.get(idx, set())
        coverage = len(hit_set) / len(exp_skills)
        if coverage >= EXPERIENCE_COVERAGE_THRESHOLD:
            self.state.move_to_next_experience()
            logger.info("Advanced experience %d (coverage %.0f%%)", idx + 1, coverage * 100)

    def _get_experience_skills(self, experience: Dict) -> List[str]:
        return list(set(
            experience.get("tech_stack", []) + experience.get("methods", [])
        ))

    def _update_experience_hit(self, target: QuestionTarget) -> None:
        idx        = self.state.current_experience_index
        experience = self.state.get_current_experience()
        if not experience:
            return
        exp_skills = self._get_experience_skills(experience)
        exp_text   = " ".join(
            exp_skills
            + [experience.get("summary", "")]
            + [experience.get("claimed_responsibility", "")]
        )
        if target.target in exp_skills:
            self._experience_skill_hits.setdefault(idx, set()).add(target.target)
            return
        if _skill_fuzzy_match(target.target, exp_text):
            self._experience_skill_hits.setdefault(idx, set()).add(target.target)
            logger.debug("Fuzzy hit: '%s' in experience %d", target.target, idx)

    def _record_question_asked(self, target: QuestionTarget) -> None:
        if target.question_type == "technical":
            self._skills_asked.add(target.target)
        else:
            self._dimensions_asked.add(target.target)
        self._update_experience_hit(target)

    # =========================================================================
    # Fix 4: Coverage = observations > 0
    # =========================================================================

    def _compute_skill_coverage(self) -> Dict[str, Any]:
        if not self._ability_model:
            return {"total": 0, "covered": 0, "rate": 0.0, "uncovered": []}
        skills        = self._ability_model.skills
        total         = len(skills)
        covered_names = [s for s, e in skills.items() if e.observations > 0]
        return {
            "total":     total,
            "covered":   len(covered_names),
            "rate":      round(len(covered_names) / total, 2) if total else 0.0,
            "uncovered": [s for s in skills if s not in covered_names],
        }

    def _compute_dimension_coverage(self) -> Dict[str, Any]:
        if not self._ability_model:
            return {"total": 0, "covered": 0, "rate": 0.0, "uncovered": []}
        dims          = self._ability_model.dimensions
        total         = len(dims)
        covered_names = [d for d, e in dims.items() if e.observations > 0]
        return {
            "total":     total,
            "covered":   len(covered_names),
            "rate":      round(len(covered_names) / total, 2) if total else 0.0,
            "uncovered": [d for d in dims if d not in covered_names],
        }

    def _refresh_coverage_views(self) -> None:
        self.skill_coverage     = self._compute_skill_coverage()
        self.dimension_coverage = self._compute_dimension_coverage()

    # =========================================================================
    # 内部辅助方法
    # =========================================================================

    def _handle_opening_response(self, answer: str) -> None:
        self.state.add_dialogue(
            question=self._current_question,
            answer=answer,
            skill=None,
            question_type="opening",
            is_followup=False,
            difficulty="easy",
        )

    def _evaluate_and_record(self, question: str, answer: str) -> Dict:
        question_type = self.state.last_question_type or "technical"
        target_skill  = self.state.last_question_skill or ""

        try:
            evaluation = self.evaluator.run(
                question=question,
                answer=answer,
                question_type=question_type,
                target_skill_or_dimension=target_skill,
            )
        except Exception as e:
            logger.warning("Evaluator failed: %s. score=5.0", e)
            evaluation = {
                "评分":     5.0,
                "评估类型": "技术评估" if question_type == "technical" else "能力评估",
                "考察目标": target_skill,
                "评语":     "评估失败，使用默认分数",
            }

        self.state.record_evaluation(evaluation)

        raw_score = evaluation.get("评分", 5.0)
        if target_skill:
            self._update_ability_model(target_skill, question_type, raw_score)

        is_followup = (
            self.state.followup_counts.get(target_skill, 0) > 0
            and self.state.last_question_skill == target_skill
        )
        self.state.add_dialogue(
            question=question,
            answer=answer,
            skill=target_skill,
            question_type=question_type,
            is_followup=is_followup,
            difficulty=self._current_difficulty,
        )

        self.answer_evaluations.append(evaluation)

        if self.save_training_data:
            self.training_samples.append({
                "candidate":     self.candidate_name,
                "job_name":      self.job_name,
                "question":      question,
                "answer":        answer,
                "question_type": question_type,
                "skill":         target_skill,
                "difficulty":    self._current_difficulty,
                "score":         raw_score,
                "evaluation":    evaluation,
                # "ability_at_time": (
                #     self._ability_model.get_skill(target_skill).summary()
                #     if question_type == "technical" and self._ability_model else None
                # ),
                "ability_at_time": (
                    {
                        "mean": round(a.mean, 2),
                        "uncertainty": round(a.uncertainty, 2),
                        "n": a.observations,
                    }
                    if (
                        question_type == "technical"
                        and self._ability_model
                        and (a := self._ability_model.skills.get(target_skill))
                    )
                    else None
                ),
            })

        return evaluation

    def _generate_question(self, target: QuestionTarget) -> str:
        experience = self.state.get_current_experience() or {}
        if target.question_type == "technical":
            return self._generate_technical(target, experience)
        return self._generate_competency(target, experience)

    def _generate_technical(self, target: QuestionTarget, experience: Dict) -> str:
        covered = [
            s for s, e in (self._ability_model.skills if self._ability_model else {}).items()
            if e.observations >= 2
        ]
        try:
            return self.tech_agent.run(
                experience_unit=experience,
                covered_tech_points=covered,
                difficulty_level=target.difficulty,
            )
        except Exception as e:
            logger.error("TechnicalQuestionAgent failed: %s", e)
            return f"请介绍一下您在 {target.target} 方面的理解和实践经验。"

    def _generate_competency(self, target: QuestionTarget, experience: Dict) -> str:
        dimension_dict = self._find_dimension_dict(target.target)
        try:
            return self.comp_agent.run(
                experience_unit=experience,
                target_dimension=dimension_dict,
                dialogue_history=self.state.dialogue_history,
                difficulty_level=target.difficulty,
            )
        except Exception as e:
            logger.error("CompetencyQuestionAgent failed: %s", e)
            return f"请描述一个体现您{target.target}能力的具体经历。"

    def _sync_phase(self, target: QuestionTarget) -> None:
        current   = self.state.interview_phase
        next_type = target.question_type
        if next_type == "closing":
            self.state.interview_phase = "closing"
        elif next_type == "competency" and current == "technical":
            self.state.interview_phase = "competency"
            logger.info("Phase transition: technical → competency")
        elif next_type == "technical" and current == "opening":
            self.state.interview_phase = "technical"

    def _should_end_interview(self) -> bool:
        q = self.state.question_count
        if q >= self.max_questions:
            logger.info("Hard limit: %d/%d", q, self.max_questions)
            return True
        phase  = self.state.interview_phase
        meta   = self.state.question_metadata
        tech_q = sum(1 for m in meta if m.get("question_type") == "technical")
        comp_q = sum(1 for m in meta if m.get("question_type") == "competency")
        if phase == "technical"  and tech_q >= MAX_TECHNICAL_QUESTIONS:
            return False
        if phase == "competency" and comp_q >= MAX_COMPETENCY_QUESTIONS:
            return True
        return False

    def _finish_interview(self) -> None:
        self._finished = True
        self._maybe_save_training_data()
        self._refresh_coverage_views()

    def _maybe_save_training_data(self) -> None:
        if not self.save_training_data or not self.training_samples:
            return
        try:
            os.makedirs(self.training_data_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            filename = os.path.join(
                self.training_data_dir,
                f"{date_str}_{self.candidate_name}_{self.job_name}.json",
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.training_samples, f, ensure_ascii=False, indent=2)
            logger.info("Training data saved: %s (%d records)", filename, len(self.training_samples))
        except Exception as e:
            logger.error("Failed to save training data: %s", e)

    def _find_dimension_dict(self, dimension_name: str) -> Dict:
        for dim in self.state.job_analysis.get("competency_dimensions", []):
            if isinstance(dim, dict) and dim.get("dimension") == dimension_name:
                return dim
            if isinstance(dim, str) and dim == dimension_name:
                return {"dimension": dimension_name, "sub_dimensions": []}
        return {"dimension": dimension_name, "sub_dimensions": []}

    def _compute_max_depth(self) -> int:
        if not self._skill_graph:
            return 0
        return max((n.depth for n in self._skill_graph.nodes.values()), default=0)

    def _mean_to_level(self, mean: float) -> str:
        if mean >= 0.78:   return "senior"
        elif mean >= 0.52: return "mid"
        return "junior"

    def _make_recommendation(self, mean: float) -> str:
        if mean >= 0.78:   return "强烈推荐录用"
        elif mean >= 0.60: return "推荐录用"
        elif mean >= 0.45: return "有条件录用，建议进一步评估"
        return "暂不推荐"

    def _infer_phases_completed(self) -> List[str]:
        types_seen = {m.get("question_type") for m in self.state.question_metadata}
        phases = ["opening"]
        if "technical"  in types_seen: phases.append("technical")
        if "competency" in types_seen: phases.append("competency")
        if self._finished:             phases.append("closing")
        return phases

    # ── 属性 ──────────────────────────────────────────────────────────────────

    @property
    def current_question(self) -> str:
        return self._current_question

    @property
    def progress(self) -> Dict[str, Any]:
        summary = self.state.get_progress_summary()
        summary.update({
            "candidate_name":      self.candidate_name,
            "current_difficulty":  self._current_difficulty,
            "skill_coverage":      self._compute_skill_coverage(),
            "dimension_coverage":  self._compute_dimension_coverage(),
            "max_questions":       self.max_questions,
            "skill_graph_summary": self._skill_graph.summary() if self._skill_graph else {},
        })
        return summary
